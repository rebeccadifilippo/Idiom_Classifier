from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

def main():
	print("1. Loading training data from CodaBench scoringkit...")
	in_path = Path("CodaBench/starterkit/train.csv")

	if not in_path.exists():
		raise FileNotFoundError(f"Could not find expected input file: {in_path}")

	df = pd.read_csv(in_path)
	required_cols = {"sentence", "label"}
	missing_cols = required_cols - set(df.columns)
	if missing_cols:
		raise ValueError(f"Missing required columns in {in_path}: {sorted(missing_cols)}")

	# Keep only columns required for the competition format.
	df = df[["sentence", "label"]]
	print(f"Total input train size: {len(df)} rows")

	print("2. Performing stratified train/validation split (75/25)...")
	# Reserve 25% of the existing train set for validation.
	train_df, val_df = train_test_split(
		df,
		test_size=0.25,
		random_state=42,
		stratify=df["label"],
	)

	print(f"New train size: {len(train_df)}")
	print(f"Validation size: {len(val_df)}")

	print("3. Saving split files...")
	train_df.to_csv("final_train.csv", index=False)
	val_df.to_csv("final_validation.csv", index=False)

	print("\nStratified train/validation split complete! Files saved.\n")

if __name__ == "__main__":
	main()
