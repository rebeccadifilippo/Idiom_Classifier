import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/idioms_data.csv")

TEXT_COLUMN = "sentence"
LABEL_COLUMN = "label"

# Array of labels used in the dataset
LABELS = ["Metaphor", "Euphemism", "Personification", "Parallelism", "Simile", 
          "Oxymoron", "Paradox", "Hyperbole", "Literal", "Irony"]

def load_raw_csv() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data csv file not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8")

    return df

#group by sentence id and combine tokens into a full sentence
def merge_tokens(tokens):
    tokens = [str(t) for t in tokens if pd.notna(t)]
    sentence = ' '.join(tokens)
    sentence = sentence.replace(' ,', ',').replace(' .', '.').replace(' `` ', '"').replace(" '' ", '"')
    return sentence

def cleanup_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df['id'] = df['id'].ffill()
    df['class'] = df['class'].ffill()

    sentences = df.groupby(['id', 'class'])['token'].apply(merge_tokens).reset_index()
    sentences = sentences.rename(columns={'token': 'sentence', 'class': 'label'})

    return sentences

def main() -> None:
    df = load_raw_csv()
    df = cleanup_dataframe(df)

    print(df.head(25))
    print(df.columns)

if __name__ == '__main__':
    main()
