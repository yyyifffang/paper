import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

def load_local_llm(model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    print(f"準備載入模型: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 使用 bfloat16 經度載入，並自動分配到 GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    print("模型載入完成")
    return tokenizer, model

def generate_augmentations(tokenizer, model, text, category_name):
    # 定義 system prompt 與 user prompt
    system_prompt = "You are a real human user typing natural, short questions into an enterprise customer service chatbot or search engine. Do not act like an AI or an encyclopedia."
    
    user_prompt = f"""
Given the following user question which strictly asks for information about '{category_name}', generate 3 diverse augmented versions of this question.

CRITICAL INSTRUCTIONS FOR CUSTOMER SERVICE DOMAIN:
1. EXTREME ISOLATION: The augmented questions must strictly ask for '{category_name}'. 
   - If 'HUM' (Human/Person), ask WHO or WHICH PERSON. Do not ask where.
   - If 'LOC' (Location), ask WHERE or WHICH PLACE. Do not ask who.
   - If 'NUM' (Numeric), ask HOW MANY, WHEN, or HOW MUCH.
   - If 'DESC' (Description), ask WHAT IS or HOW TO.
   - If 'ENTY' (Entity), ask WHAT [OBJECT/THING].
   - If 'ABBR' (Abbreviation), ask WHAT DOES IT STAND FOR.
2. KEEP IT SHORT: Real users ask short questions. Each variation MUST be under 15 words.
3. TONE: Conversational, direct, and slightly informal (like a user typing on a phone).
4. NO HALLUCINATION: Do not answer the question. Only generate the question itself.

Original Question: {text}

Output format:
VARIATION 1: [Short Question]
VARIATION 2: [Short Question]
VARIATION 3: [Short Question]

Do not include any other explanations or introductory text.
"""

    # 1. 使用 llama-3 的 chat template 包裝 prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. 將文字轉換為模型需要的 Tensor 格式 (包含 input_ids 與 attention_mask)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 3. 模型進行推論生成 (短問句）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # 4. 解碼生成的文本 (精準裁切掉輸入 prompt 的長度)
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    # 使用正則表達式提取 三個變體
    variations = re.findall(r"VARIATION \d+:\s*(.*)", response)

    # 清理LLM的雙引號或是單引號
    clean_variations = [var.strip().replace('"', '').replace("'", "") for var in variations]

    # 確保有三個變體，如果不足則補空字符串
    return clean_variations if clean_variations else []

def main():
    trace_label_map = {
        0: "ABBR (Abbreviation / Acronym)", 
        1: "ENTY (Entity / Objects / Things)", 
        2: "DESC (Description / Definition / Explanation)", 
        3: "HUM (Human / Person / Organization)", 
        4: "LOC (Location / Places)", 
        5: "NUM (Numeric / Dates / Quantities / Prices)"
    }

    # 1. 讀取困難樣本
    df_hard = pd.read_csv("hard_samples_trec6.csv")
    print(f"成功讀取 {len(df_hard)} 筆困難樣本。")

    # 2. 載入本地 LLM 
    tokenizer, model = load_local_llm("meta-llama/Meta-Llama-3-8B-Instruct")

    augmented_data = [] # 宣告陣列

    # 3. 逐筆進行資料擴增
    print("\n開始進行 LLM 資料擴增...")
    for index, row in df_hard.iterrows():
        original_text = row['text']
        true_label = row['true_label']
        category_name = trace_label_map[true_label]

        print(f"處理第 {index+1}/40 筆 (真實類別: {category_name})...")

        variations = generate_augmentations(tokenizer, model, original_text, category_name)

        for var in variations:
            augmented_data.append({
                'text': var.strip(),
                'label': true_label,
                'source': 'llm_augmented'
            })

    # 4.儲存擴增結果 
    df_aug = pd.DataFrame(augmented_data)
    output_filename = "augmented_samples.csv"
    df_aug.to_csv(output_filename, index=False)
    print(f"\n擴增完成！共生成 {len(df_aug)} 筆新資料，已儲存至 {output_filename}")

if __name__ == "__main__":
    main()