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
    system_prompt = "You are an expert data annotator and data augmentation assistant for machine learning. Your goal is to generate text that creates clear, unambiguous decision boundaries between text categories."
    
    user_prompt = f"""
Given the following news text which belongs strictly to the '{category_name}' category, generate 3 diverse augmented versions.

CRITICAL INSTRUCTIONS to prevent Class Collapse:
1. EXTREME CATEGORY ISOLATION: The augmented text must explicitly and overwhelmingly sound like '{category_name}'. 
2. AVOID OVERLAP: 
   - If the category is 'Sports', absolutely DO NOT use business jargon (e.g., merger, stock, company, CEO, market) or tech jargon (e.g., software, digital, network). Focus entirely on athletic competitions, scores, players, and matches.
   - If the category is 'World', focus on geopolitics, international relations, governments, and global events. Avoid corporate or technology-centric vocabulary.
   - If the category is 'Business', focus on markets, companies, and economics.
   - If the category is 'Sci/Tech', focus on technology, science, computers, and space.
3. REMOVE AMBIGUITY: If the original text contains words that might confuse '{category_name}' with another category, you MUST replace them with strong, unambiguous terms specific to '{category_name}'.

Original Text: {text}

Output format:
VARIATION 1: [Your unambiguously '{category_name}' text here]
VARIATION 2: [Your unambiguously '{category_name}' text here]
VARIATION 3: [Your unambiguously '{category_name}' text here]

Do not include any other explanations.
"""

    # 1. 使用 llama-3 的 chat template 包裝 prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. 將文字轉換為模型需要的 Tensor 格式 (包含 input_ids 與 attention_mask)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 3. 模型進行推論生成 (使用 **inputs 進行解包)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # 4. 解碼生成的文本 (精準裁切掉輸入 prompt 的長度)
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    # 使用正則表達式提取 三個變體
    variations = re.findall(r"VARIATION \d+:\s*(.*)", response)

    # 確保有三個變體，如果不足則補空字符串
    return variations if variations else []

def main():
    # 類別對照表 (0: World, 1: Sports, 2: Business, 3: Sci/Tech)
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    # 1. 讀取困難樣本
    df_hard = pd.read_csv("hard_samples_with_text.csv")
    print(f"成功讀取 {len(df_hard)} 筆困難樣本。")

    # 2. 載入本地 LLM 
    tokenizer, model = load_local_llm("meta-llama/Meta-Llama-3-8B-Instruct")

    augmented_data = [] # 宣告陣列

    # 3. 逐筆進行資料擴增
    print("\n開始進行 LLM 資料擴增...")
    for index, row in df_hard.iterrows():
        original_text = row['text']
        true_label = row['true_label']
        category_name = label_map[true_label]

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