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
from scipy.stats import entropy
import evaluate
from data_utils import partition_data

# ==========================================
# 參數設定 (Hyperparameters)
# ==========================================
MAX_ITERATIONS = 10           
WARMUP_ITERATIONS = 2        # 前 2 輪使用 Random 暖身，建立基礎認知
SAMPLES_PER_ITER = 40        
LAMBDA_PENALTY = 0.0005      
COST_PER_SAMPLE = 3

TREC6_LABEL_MAP = {
    0: "ABBR (Abbreviation / Acronym)", 
    1: "ENTY (Entity / Objects / Things)", 
    2: "DESC (Description / Definition / Explanation)", 
    3: "HUM (Human / Person / Organization)", 
    4: "LOC (Location / Places)", 
    5: "NUM (Numeric / Dates / Quantities / Prices)"
}

# ==========================================
# 模組 1: 訓練與評估 BERT
# ==========================================
def compute_metrics(eval_pred):
    acc_metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"]}

def train_and_evaluate_bert(train_df, val_df, iter_num):
    print(f"\n[Hybrid Iter {iter_num}] 開始訓練 BERT...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    
    train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)
    
    output_dir = f"./models/hybrid_results_iter_{iter_num}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  
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
    
    print(f"[Hybrid Iter {iter_num}] BERT 訓練完成！Validation Accuracy: {acc:.4f}")
    
    best_model_path = trainer.state.best_model_checkpoint
    
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return acc, best_model_path

# ==========================================
# 模組 2A: Random 隨機抽樣 (Phase 1)
# ==========================================
def select_random_samples(pool_df, top_k=SAMPLES_PER_ITER):
    print("\n[Phase 1: 暖身期] 隨機抽取一般性樣本...")
    return pool_df.sample(n=top_k, random_state=None)

# ==========================================
# 模組 2B: Entropy 困難樣本挑選 (Phase 2)
# ==========================================
def select_hard_samples(pool_df, model_path, top_k=SAMPLES_PER_ITER):
    print("\n[Phase 2: 主動學習] 尋找高 Entropy 的困難邊界樣本...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
    model.eval()
    
    pool_ds = Dataset.from_pandas(pool_df)
    entropies = []
    
    with torch.no_grad():
        for i in range(len(pool_ds)):
            inputs = tokenizer(pool_df.iloc[i]['text'], return_tensors="pt", truncation=True, max_length=64).to("cuda")
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            ent = entropy(probs)
            entropies.append(ent)
            
    pool_df = pool_df.copy()
    pool_df['entropy'] = entropies
    hard_samples = pool_df.sort_values(by='entropy', ascending=False).head(top_k)
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return hard_samples

# ==========================================
# 模組 3: 呼叫 Llama-3 進行資料擴增
# ==========================================
def augment_with_llm(samples_df):
    print("\n啟動 Llama-3 進行客服意圖資料擴增...")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    augmented_records = []
    
    for idx, row in samples_df.iterrows():
        cat_name = TREC6_LABEL_MAP[row['label']]
        
        system_prompt = "You are a real human user typing natural, short questions into an enterprise customer service chatbot or search engine. Do not act like an AI or an encyclopedia."
        user_prompt = f"""
Given the following user question which strictly asks for information about '{cat_name}', generate 3 diverse augmented versions of this question.

CRITICAL INSTRUCTIONS FOR CUSTOMER SERVICE DOMAIN:
1. EXTREME ISOLATION: The augmented questions must strictly ask for '{cat_name}'. 
   - If 'HUM' (Human/Person), ask WHO or WHICH PERSON. Do not ask where.
   - If 'LOC' (Location), ask WHERE or WHICH PLACE. Do not ask who.
   - If 'NUM' (Numeric), ask HOW MANY, WHEN, or HOW MUCH.
   - If 'DESC' (Description), ask WHAT IS or HOW TO.
   - If 'ENTY' (Entity), ask WHAT [OBJECT/THING].
   - If 'ABBR' (Abbreviation), ask WHAT DOES IT STAND FOR.
2. KEEP IT SHORT: Real users ask short questions. Each variation MUST be under 15 words.
3. TONE: Conversational, direct, and slightly informal (like a user typing on a phone).
4. NO HALLUCINATION: Do not answer the question. Only generate the question itself.

Original Question: {row['text']}

Output format:
VARIATION 1: [Short Question]
VARIATION 2: [Short Question]
VARIATION 3: [Short Question]

Do not include any other explanations or introductory text.
"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        
        input_length = inputs['input_ids'].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        variations = re.findall(r"VARIATION \d+:\s*(.*)", response)
        
        for var in variations:
            clean_var = var.strip().replace('"', '').replace("'", "")
            augmented_records.append({'text': clean_var, 'label': row['label']})
            
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return pd.DataFrame(augmented_records)

# ==========================================
# 主迴圈 (Hybrid Learning Loop)
# ==========================================
def main():
    print("初始化 TREC-6 Hybrid Learning 環境...")
    
    dataset = load_dataset("trec", trust_remote_code=True)
    df = dataset['train'].to_pandas()
    df['label'] = df['coarse_label']
    df_clean = df[['text', 'label']]

    # 隨機抽取3次不同的data seed
    test_seeds = [42, 2026, 9527]
    all_histories = []
    
    for seed in test_seeds:
        print(f"\n{'='*60}")
        print(f"使用 Random Seed : {seed} 初始化數據集...")
        print(f"\n{'='*60}")

        df_seed, df_val, df_test = partition_data(df_clean, random_seed=seed)
        
        used_indices = pd.concat([df_seed, df_val, df_test]).index
        df_pool = df_clean.drop(used_indices).reset_index(drop=True)
        df_pool = df_pool.sample(frac=1, random_state=seed).reset_index(drop=True)
    
        current_train_df = df_seed.copy()
        cumulative_cost = 0
        best_U = -float('inf')
        history = []

        for t in range(MAX_ITERATIONS):
            print(f"\n{'='*60}")
            phase_name = "暖身期 (Random)" if t < WARMUP_ITERATIONS else "主動學習期 (Entropy)"
            print(f"開始 Hybrid Iteration {t} - 【{phase_name}】 (目前訓練集大小: {len(current_train_df)})")
            print(f"{'='*60}")
            
            # 1. 訓練並取得 Performance
            acc, best_model_path = train_and_evaluate_bert(current_train_df, df_val, t)
            
            # 2. 計算效用函數 U = P - λ * C
            U = acc - (LAMBDA_PENALTY * cumulative_cost)
            print(f"效用函數評估: P(Acc)={acc:.4f}, Cost={cumulative_cost}, U={U:.4f}")
            
            history.append({'iter': t, 'accuracy': acc, 'cost': cumulative_cost, 'utility': U})
            
            # 3. 動態停止機制 (Stopping Criterion)
            if U < best_U:
                if t < WARMUP_ITERATIONS:
                    print(f"[暖身期保護] 本回合效用 U ({U:.4f}) 雖低於最佳效用 ({best_U:.4f})，但仍在暖身期，強制繼續擴增以建立基礎認知。")
                else:
                    print(f"\n[Early Stopping 觸發] 本回合效用 U ({U:.4f}) 低於最佳效用 ({best_U:.4f})！")
                    print("證明邊際效益已遞減，停止迴圈。")
                    break
            else:
                best_U = U
                
            # 4. 挑選樣本與擴增
            if t < MAX_ITERATIONS - 1:
                if t < WARMUP_ITERATIONS:
                    # 暖身期：使用隨機抽樣
                    samples_df = select_random_samples(df_pool)
                else:
                    # 主動學習期：使用 Entropy 抽樣
                    samples_df = select_hard_samples(df_pool, best_model_path)
                
                df_pool = df_pool.drop(samples_df.index).reset_index(drop=True)
                
                augmented_df = augment_with_llm(samples_df)
                current_train_df = pd.concat([current_train_df, augmented_df], ignore_index=True)
                current_train_df.to_csv(f"data/hybrid_train_seed{seed}_iter_{t+1}_lambda0.0005.csv", index=False)
                
                cumulative_cost += len(augmented_df)
                print(f"成功加入 {len(augmented_df)} 筆擴增資料，準備進入下一輪。")

        for h in history:
            h['seed'] = seed
        all_histories.extend(history)
        print(f"Seed {seed} 的歷程已儲存。")

    # 所有seed跑完存成一個csv
    pd.DataFrame(all_histories).to_csv("data/hybrid_learning_history_all_seeds_lambda0.0005.csv", index=False)
    print("\n所有Seed的歷程已儲存至 data/hybrid_learning_history_all_seeds_lambda0.0005.csv")

if __name__ == "__main__":
    main()