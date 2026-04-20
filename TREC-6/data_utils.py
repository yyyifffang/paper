import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset
from transformers import BertTokenizer

def partition_data(df, random_seed=42):
    # seed(10/type), val(20/type), test(30/type)
    seed_list, val_list, test_list = [], [], []

    for label, group in df.groupby('label'):
        #1. test (30/type)
        rest, test = train_test_split(group, test_size=30, random_state=random_seed)
        #2. val (20/type)
        rest, val = train_test_split(rest, test_size=20, random_state=random_seed)
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
    print("從 Hugging Face 載入 TREC-6 智能客服意圖資料集...")
    dataset = load_dataset("trec", trust_remote_code=True)
    # 1. 轉換為 pandas DataFrame
    df = dataset['train'].to_pandas()

    # 2. 清理與轉換資料
    df['label'] = df['coarse_label']
    df_clean = df[['text', 'label']]
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
            max_length=64
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