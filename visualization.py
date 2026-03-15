import os
import pandas as pd
import matplotlib.pyplot as plt

from load_data import load_raw_csv, cleanup_dataframe

def plot_label_counts(out_path: str = 'label_counts.png', show: bool = False) -> str:
    df = load_raw_csv()

    df_clean = cleanup_dataframe(df)

    counts = df_clean['label'].fillna('Unknown').value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar')
    plt.title('Number of sentences per label')
    plt.xlabel('Label')
    plt.ylabel('Number of sentences')
    plt.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_path)
    if show:
        plt.show()
    plt.close()

    return out_path

if __name__ == '__main__':
    out_file = plot_label_counts()
    print(f'Saved label counts plot to {out_file}')
