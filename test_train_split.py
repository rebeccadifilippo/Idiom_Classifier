import pandas as pd
from sklearn.model_selection import train_test_split

# Import your custom functions from load_data.py
from load_data import load_raw_csv, cleanup_dataframe

def main():
    print("1. Loading and cleaning raw data...")
    # Call your existing functions to get the cleanly formatted sentences
    raw_df = load_raw_csv()
    clean_df = cleanup_dataframe(raw_df)

    # Filter down to just the columns needed for the competition format
    df = clean_df[['sentence', 'label']]

    print(f"Total dataset size after cleaning: {len(df)} sentences")

    print("2. Performing stratified train/test split (80/20)...")
    # The 'stratify' parameter ensures the exact class ratios are maintained
    train_df, test_df = train_test_split(
        df, 
        test_size=0.20, 
        random_state=42, 
        stratify=df['label']
    )

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    print("3. Generating CodaBench files...")
    
    # Training data (goes in the Starting Kit)
    train_df.to_csv('train.csv', index=False)

    # Test Reference Data (The Secret Answer Key for your CodaBench Task)
    test_df.to_csv('test_reference.csv', index=False)

    # Test Input Data (Blanked out labels for the Starting Kit)
    test_input_df = test_df.copy()
    test_input_df['label'] = ''
    test_input_df.to_csv('test_input.csv', index=False)

    print("\nStratified train/test split complete! Files saved for CodaBench.")

if __name__ == "__main__":
    main()