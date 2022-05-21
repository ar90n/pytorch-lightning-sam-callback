import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def main():
    try:
        vanilla_csv_path = Path(sys.argv[1])
        sam_csv_path = Path(sys.argv[2])
    except IndexError:
        print("Usage: gen_training_chart.py vanilla_csv_path sam_csv_path")
        exit(1)

    vanilla_df = pd.read_csv(vanilla_csv_path)
    sam_df = pd.read_csv(sam_csv_path)

    vanilla_train_loss = vanilla_df[~vanilla_df["train_loss"].isna()]["train_loss"].reset_index(drop=True)
    sam_train_loss = sam_df[~sam_df["train_loss"].isna()]["train_loss"].reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.plot(vanilla_train_loss.index, vanilla_train_loss, label="Vanilla")
    plt.plot(sam_train_loss.index, sam_train_loss, label="SAM")
    plt.legend()
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("training_loss.png")
    plt.show()


    vanilla_val_loss = vanilla_df[~vanilla_df["val_loss"].isna()]["val_loss"].reset_index(drop=True)
    sam_val_loss = sam_df[~sam_df["val_loss"].isna()]["val_loss"].reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.plot(vanilla_val_loss.index, vanilla_val_loss, label="Vanilla")
    plt.plot(sam_val_loss.index, sam_val_loss, label="SAM")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validatoin Loss")
    plt.savefig("validation_loss.png")
    plt.show()

    

if __name__ == "__main__":
    main()