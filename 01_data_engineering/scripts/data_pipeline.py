"""
NovaTrust Credit Risk Intelligence Platform

Module:
Data Cleaning Pipeline

Author:
Sakshi Patil
"""

from pathlib import Path
import pandas as pd

def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.
    """
    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print("\nDuplicate Check")
    print("-" * 40)
    print(f"Rows Before : {before}")
    print(f"Rows After  : {after}")
    print(f"Duplicates Removed : {before - after}")

    return df

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    """

    print("\nMissing Value Check")
    print("-" * 40)

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
        return df

    for column in missing.index:

        percentage = (missing[column] / len(df)) * 100

        print(f"{column} : {missing[column]} ({percentage:.2f}%)")

        # Numeric columns
        if df[column].dtype in ["int64", "float64"]:

            median_value = df[column].median()

            df[column] = df[column].fillna(median_value)

            print(f"Filled with Median ({median_value})")

        # Text columns
        else:

            mode_value = df[column].mode()[0]

            df[column] = df[column].fillna(mode_value)

            print(f"Filled with Mode ({mode_value})")

    print("\nRemaining Missing Values")

    print(df.isnull().sum().sum())

    return df

def fix_data_types(df):
    """
    Display the data types of each column.
    """

    print("\nChecking Data Types")
    print("-" * 40)

    print(df.dtypes)

    return df

def save_clean_data(df, project_root):
    """
    Save cleaned dataset.
    """

    output_folder = (
        project_root
        / "01_data_engineering"
        / "processed_data"
    )

    output_folder.mkdir(exist_ok=True)

    csv_file = output_folder / "clean_loans.csv"

    df.to_csv(csv_file, index=False)

    print("\nClean dataset saved successfully!")

    print(csv_file)

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    dataset_path = (
        project_root
        / "01_data_engineering"
        / "raw_data"
        / "Loan_Default.csv"
    )

df = pd.read_csv(dataset_path)

df = remove_duplicates(df)

df = handle_missing_values(df)

df = fix_data_types(df)

save_clean_data(df, project_root)