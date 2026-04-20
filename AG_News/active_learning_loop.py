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
MAX_ITERATIONS = 5           # 最多執行幾輪
SAMPLES_PER_ITER = 40        # 每一輪挑選多少困難樣本
LAMBDA_PENALTY = 0.0002      # 效用函數中的成本懲罰係數 (lambda)
COST_PER_SAMPLE = 3          # 每個樣本擴增出 3 筆資料的成本

# 類別對照表
LABEL_MAP = { 0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech" }

# ==========================================
# 模組 1: 訓練與評估 BERT，並回傳準確率與模型路徑
# ==========================================
def compute_metrics(eval_pred):
    acc_metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"]}

def train_and_evaluate_bert(train_df, val_df, iter_num):
    print(f"\n[Iter {iter_num}] 開始訓練 BERT...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    train_ds = Dataset.from_pandas(train_df).map(tokenize_fn, batched=True)
    val_ds = Dataset.from_pandas(val_df).map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

    output_dir = f"./models/results_iter_{iter_num}"

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
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_results = trainer.evaluate()
    acc = eval_results["eval_accuracy"]

    print(f"[Iter {iter_num}] BERT 訓練完成！Validation Accuracy: {acc:.4f}")

    # 儲存最佳模型路徑 用來算entropy
    best_model_path = trainer.state.best_model_checkpoint

    # 清空 VRAM
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()

    return acc, best_model_path

# ==========================================
# 模組 2: 使用訓練好的 BERT 計算 Pool 的 Entropy 並挑選困難樣本
# ==========================================
def select_hard_samples(pool_df, model_path, top_k=SAMPLES_PER_ITER):
    print("\n尋找高 Entropy 的困難樣本...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")
    model.eval()

    pool_ds = Dataset.from_pandas(pool_df)
    entropies = []

    with torch.no_grad():
        for i in range(len(pool_ds)):
            inputs = tokenizer(pool_df.iloc[i]['text'], return_tensors="pt", truncation=True, max_length=128).to("cuda")
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            ent = entropy(probs)
            entropies.append(ent)

    pool_df = pool_df.copy()
    pool_df['entropy'] = entropies
    hard_samples = pool_df.sort_values(by='entropy', ascending=False).head(top_k)

    # 清空 VRAM
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return hard_samples

# ==========================================
# 模組 3: 呼叫 Llama-3 進行資料擴增 (防崩塌 Prompt)
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

        messages =[{"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_prompt} ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        
        input_length = inputs['input_ids'].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        variations = re.findall(r"VARIATION \d+:\s*(.*)", response)

        for var in variations:
            augmented_records.append({"text": var.strip(), "label": row['label']})

    # 清空 VRAM
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return pd.DataFrame(augmented_records)

# ==========================================
# 主迴圈 (Active Learning Loop)
# ==========================================
def main():
    print("🚀 初始化 Active Learning 環境...")

    # 讀取並切分資料
    dataset = load_dataset("sh0416/ag_news")
    df = dataset['train'].to_pandas()
    df['label'] = df['label'] - 1
    df['text'] = df['title'] + ' - ' + df['description']
    df_clean = df[['text', 'label']]
    
    df_seed, df_val, df_test = partition_data(df_clean, random_seed=42)

    # 建立 Unlabeled Pool (扣除掉 Seed、Val、Test 的剩餘資料)
    used_indices = pd.concat([df_seed, df_val, df_test]).index
    df_pool = df_clean.drop(used_indices).reset_index(drop=True)
    # 加速實驗，從 Pool 抽 5000 筆當作未標註池
    df_pool = df_pool.sample(5000, random_state=42).reset_index(drop=True)

    current_train_df = df_seed.copy()
    cumulative_cost = 0
    best_U = -float('inf')

    history = []

    for t in range(MAX_ITERATIONS):
        print(f"\n{'='*50}")
        print(f"🔄 開始 Iteration {t} (目前訓練集大小: {len(current_train_df)})")
        print(f"{'='*50}")

        # 1. 訓練並取得 Performance
        acc, best_model_path = train_and_evaluate_bert(current_train_df, df_val, t)
        P = acc

        # 2. 計算效用函數 U = P - λ * C
        U = P - (LAMBDA_PENALTY * cumulative_cost)
        print(f"📊 效用函數評估: P(Acc)={P:.4f}, Cost={cumulative_cost}, U={U:.4f}")

        history.append({
            "iter": t,
            "accuracy": P,
            "cost": cumulative_cost,
            "utility": U
        })

        # 3. 動態停止機制 (Stopping Criterion)
        if U < best_U:
            print(f"\n🛑 [Early Stopping 觸發] 本回合效用 U ({U:.4f}) 低於最佳效用 ({best_U:.4f})！")
            print("🛑 證明邊際效益已遞減，停止迴圈。")
            break
        else:
            best_U = U

        # 如果還沒達到停止條件，且不是最後一輪，則繼續擴增
        if t < MAX_ITERATIONS - 1:
            # 4. 挑選困難樣本
            hard_samples_df = select_hard_samples(df_pool, best_model_path)

            # 從 Pool 中移除被選中的樣本
            df_pool = df_pool.drop(hard_samples_df.index).reset_index(drop=True)

            # 5. 使用 LLM 擴增資料
            augmented_df = augment_with_llm(hard_samples_df)

            # 6. 更新狀態
            current_train_df = pd.concat([current_train_df, augmented_df], ignore_index=True)
            current_train_df.to_csv(f"data/train_iter_{t+1}.csv", index=False)

            cumulative_cost += len(augmented_df) # 成本增加
            print(f"✅ 成功加入 {len(augmented_df)} 筆擴增資料，準備進入下一輪。")
        
    
    # 儲存實驗歷程
    pd.DataFrame(history).to_csv("data/active_learning_history.csv", index=False)
    print("\n🎉 Active Learning 實驗結束！歷程已儲存至 data/active_learning_history.csv")

if __name__ == "__main__":
    main()