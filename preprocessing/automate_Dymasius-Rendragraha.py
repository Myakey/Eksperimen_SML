import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

def run_preprocessing(train_path, test_path, output_dir):
    print("Memulai proses data preprocessing...")
    
    # 1. Memuat dataset
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print(f"Data termuat. Train: {df_train.shape}, Test: {df_test.shape}")

    # 2. Menghapus kolom yang tidak relevan
    cols_to_drop = ['Unnamed: 0', 'id']
    df_train = df_train.drop(columns=[col for col in cols_to_drop if col in df_train.columns], errors='ignore')
    df_test = df_test.drop(columns=[col for col in cols_to_drop if col in df_test.columns], errors='ignore')

    # 3. Menangani Missing Values
    median_delay = df_train['Arrival Delay in Minutes'].median()
    df_train['Arrival Delay in Minutes'] = df_train['Arrival Delay in Minutes'].fillna(median_delay)
    df_test['Arrival Delay in Minutes'] = df_test['Arrival Delay in Minutes'].fillna(median_delay)

    # 4. Encoding Data Kategorikal
    class_mapping = {'Eco': 0, 'Eco Plus': 1, 'Business': 2}
    if 'Class' in df_train.columns:
        df_train['Class'] = df_train['Class'].map(class_mapping)
        df_test['Class'] = df_test['Class'].map(class_mapping)

    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'satisfaction']
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df_train.columns:
            df_train[col] = le.fit_transform(df_train[col])
            df_test[col] = le.transform(df_test[col])

    # 5. Standarisasi Fitur Numerik
    num_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    scaler = StandardScaler()
    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_test[num_cols] = scaler.transform(df_test[num_cols])

    # 6. Menyimpan Hasil Preprocessing
    os.makedirs(output_dir, exist_ok=True)
    train_out = os.path.join(output_dir, 'train_preprocessed.csv')
    test_out = os.path.join(output_dir, 'test_preprocessed.csv')
    
    df_train.to_csv(train_out, index=False)
    df_test.to_csv(test_out, index=False)
    print(f"Preprocessing selesai! Data disimpan di: {output_dir}")

if __name__ == "__main__":
    # Menentukan path (asumsi dijalankan dari root repository)
    RAW_TRAIN = 'airplane_satisfaction_raw/train.csv'
    RAW_TEST = 'airplane_satisfaction_raw/test.csv'
    OUTPUT_FOLDER = 'preprocessing/airplane_satisfaction_preprocessing'
    
    run_preprocessing(RAW_TRAIN, RAW_TEST, OUTPUT_FOLDER)