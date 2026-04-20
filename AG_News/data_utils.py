import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from transformers import BertTokenizer

def partition_data(df, random_seed=42):
    # seed(10/type), val(200/type), test(500/type)
    seed_list, val_list, test_list = [], [], []

    for label, group in df.groupby('label'):
        #1. test (500/type)
        rest, test = train_test_split(group, test_size=500, random_state=random_seed)
        #2. val (200/type)
        rest, val = train_test_split(rest, test_size=200, random_state=random_seed)
        #3. seed 
        _, seed = train_test_split(rest, test_size=10, random_state=random_seed)

        seed_list.append(seed)
        val_list.append(val)
        test_list.append(test)

    df_seed = pd.concat(seed_list).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_val = pd.concat(val_list).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    df_test = pd.concat(test_list).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    return df_seed, df_val, df_test

def prepare_datasets(csv_path='train.csv'):
    """讀資料、清理、切分、轉換為 Hugging Face Tokenized Dataset"""
    print("從 Hugging Face 載入與處理資料...")
    dataset = load_dataset("sh0416/ag_news")
    # 1. 轉換為 pandas DataFrame
    df = dataset['train'].to_pandas()

    # 2. 清理與轉換資料
    df['label'] = df['label'] - 1  # 調整標籤從 0 開始
    df['text'] = df['title'] + ' - ' + df['description']  # 合併title與description
    df_clean = df[['text', 'label']] # 保留需要的欄位
    print(f"原始資料集大小: {len(df_clean)}")

    # 3. 切分資料
    df_seed, df_val, df_test = partition_data(df_clean, random_seed=42)
    print(f"Seed集大小: {len(df_seed)}, Val集大小: {len(df_val)}, Test集大小: {len(df_test)}")
    
    # 4. 轉換為 Hugging Face Dataset
    train_dataset = Dataset.from_pandas(df_seed)
    val_dataset = Dataset.from_pandas(df_val)
    test_dataset = Dataset.from_pandas(df_test)
    
    # 5. Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=128
    )

    # 6.批次處理編碼並移除原始文字欄位
    train_dataset = train_dataset.map(tokenize_function, batched=True).remove_columns(['text'])
    val_dataset = val_dataset.map(tokenize_function, batched=True).remove_columns(['text'])
    test_dataset = test_dataset.map(tokenize_function, batched=True).remove_columns(['text'])

    # 7. 設定格式為 PyTorch tensors
    train_dataset.set_format('torch')
    val_dataset.set_format('torch')
    test_dataset.set_format('torch')

    print("資料處理完成！")
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    train_ds, val_ds, test_ds = prepare_datasets()
    print("Training Dataset sample:", len(train_ds))