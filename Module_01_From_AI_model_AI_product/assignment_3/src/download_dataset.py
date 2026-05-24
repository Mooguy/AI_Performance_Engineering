import os
import pandas as pd

DATASET_NAME = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
URL = "hf://datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset"

def load_dataset():
    if not os.path.exists(f"data/{DATASET_NAME}"):
        print(f"Dataset not found locally. Downloading from {URL}...")
        df = pd.read_csv(f"{URL}/{DATASET_NAME}")
        df.to_csv(f"data/{DATASET_NAME}", index=False)
    else:
        print(f"Dataset found locally. Loading from {DATASET_NAME}...")
        df = pd.read_csv(f"data/{DATASET_NAME}")

    return df