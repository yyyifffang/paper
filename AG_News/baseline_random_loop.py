import pandas as pd
import numpy as np
import torch
import gc
import re
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoModelForCausalLM, TrainingArguments, Trainer
)
import evaluate
from data_utils import partition_data

# ==========================================
# 參數設定 (與 Active Learning 保持完全一致)
# ==========================================
MAX_ITERATIONS = 5
SAMPLES_PER_ITER = 40
COST_PER_SAMPLE = 3
LAMBDA_PENALTY = 0.0002  # 雖然不停車，但我們一樣記錄 U 值方便事後畫圖比較

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# ==========================================
# 模組 1: 訓練與評估 BERT (完全相同)
# ==========================================
def compute_metrics(eval_pred):
    acc_metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"]}

def train_and_evaluate_bert(train_df, val_df, iter_num):
    print(f"\n[Random Iter {iter_num}] 開始訓練 BERT...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
    
    output_dir = f"./models/random_results_iter_{iter_num}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # 已修復為新版參數
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    acc = eval_results['eval_accuracy']
    
    print(f"[Random Iter {iter_num}] BERT 訓練完成！Validation Accuracy: {acc:.4f}")
    
    # 清空 VRAM
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return acc

# ==========================================
# 模組 2: 隨機挑選樣本 (拔掉 Entropy，直接 Random)
# ==========================================
def select_random_samples(pool_df, top_k=SAMPLES_PER_ITER):
    print("\n瞎子摸象中：隨機抽取樣本 (不看 Entropy)...")
    # 直接使用 pandas 的 sample 函數隨機抽樣
    random_samples = pool_df.sample(n=top_k, random_state=None) 
    return random_samples

# ==========================================
# 模組 3: Llama-3 資料擴增 (完全相同)
# ==========================================
def augment_with_llm(hard_samples_df):
    print("\n啟動 Llama-3 進行資料擴增...")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    augmented_records = []
    
    for idx, row in hard_samples_df.iterrows():
        cat_name = LABEL_MAP[row['label']]
        
        system_prompt = "You are an expert data annotator and data augmentation assistant. Generate text that creates clear, unambiguous decision boundaries."
        user_prompt = f"""
Given the following news text which belongs strictly to the '{cat_name}' category, generate 3 diverse augmented versions.

CRITICAL INSTRUCTIONS to prevent Class Collapse:
1. EXTREME CATEGORY ISOLATION: The augmented text must explicitly and overwhelmingly sound like '{cat_name}'. 
2. AVOID OVERLAP: 
   - If 'Sports', absolutely DO NOT use business jargon (e.g., merger, stock, company, CEO, market) or tech jargon (e.g., software, digital). 
   - If 'World', focus on geopolitics, international relations, governments. Avoid corporate vocabulary.
   - If 'Business', focus on markets, companies, economics.
   - If 'Sci/Tech', focus on technology, science, computers.
3. REMOVE AMBIGUITY: Replace words that confuse '{cat_name}' with another category using strong, specific terms.

Original Text: {row['text']}

Output format:
VARIATION 1: [Text]
VARIATION 2: [Text]
VARIATION 3: [Text]

Do not include any other explanations.
"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        
        input_length = inputs['input_ids'].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        variations = re.findall(r"VARIATION \d+:\s*(.*)", response)
        
        for var in variations:
            augmented_records.append({'text': var.strip(), 'label': row['label']})
            
    # 清空 VRAM (縮排已修正)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return pd.DataFrame(augmented_records)

# ==========================================
# 主迴圈 (Baseline Random Loop)
# ==========================================
def main():
    print("🚀 初始化 Random Baseline 環境...")
    
    dataset = load_dataset("sh0416/ag_news")
    df = dataset['train'].to_pandas()
    df['label'] = df['label'] - 1
    df['text'] = df['title'] + ' - ' + df['description']
    df_clean = df[['text', 'label']]
    
    df_seed, df_val, df_test = partition_data(df_clean, random_seed=42)
    
    used_indices = pd.concat([df_seed, df_val, df_test]).index
    df_pool = df_clean.drop(used_indices).reset_index(drop=True)
    df_pool = df_pool.sample(5000, random_state=42).reset_index(drop=True)
    
    current_train_df = df_seed.copy()
    cumulative_cost = 0
    history = []

    for t in range(MAX_ITERATIONS):
        print(f"\n{'='*50}")
        print(f"🔄 開始 Random Iteration {t} (目前訓練集大小: {len(current_train_df)})")
        print(f"{'='*50}")
        
        # 1. 訓練並取得 Performance
        acc = train_and_evaluate_bert(current_train_df, df_val, t)
        
        # 2. 計算效用函數 (僅記錄，不作為停止條件)
        U = acc - (LAMBDA_PENALTY * cumulative_cost)
        print(f"📊 狀態記錄: P(Acc)={acc:.4f}, Cost={cumulative_cost}, U={U:.4f}")
        
        history.append({'iter': t, 'accuracy': acc, 'cost': cumulative_cost, 'utility': U})
        
        # 🚨 注意這裡！移除了 Early Stopping 判斷，它會盲目地跑滿 5 輪！
        
        if t < MAX_ITERATIONS - 1:
            # 3. 隨機挑選樣本 (取代了 Entropy)
            random_samples_df = select_random_samples(df_pool)
            df_pool = df_pool.drop(random_samples_df.index).reset_index(drop=True)
            
            # 4. LLM 擴增
            augmented_df = augment_with_llm(random_samples_df)
            
            # 5. 更新狀態
            current_train_df = pd.concat([current_train_df, augmented_df], ignore_index=True)
            current_train_df.to_csv(f"data/random_train_iter_{t+1}.csv", index=False)
            
            cumulative_cost += len(augmented_df)
            print(f"✅ 盲目擴增了 {len(augmented_df)} 筆資料，強制進入下一輪。")

    pd.DataFrame(history).to_csv("data/baseline_random_history.csv", index=False)
    print("\n🎉 Random 對照組實驗結束！歷程已儲存至 data/baseline_random_history.csv")

if __name__ == "__main__":
    main()