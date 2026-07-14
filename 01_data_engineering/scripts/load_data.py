"""
Project:
NovaTrust Credit Risk Intelligence Platform

Module:
Data Loading

Purpose:
Load the raw loan dataset safely
and perform initial inspection.

Author:
Sakshi Patil

"""
from pathlib import Path
import pandas as pd

def load_data():
    """
    Load the raw loan dataset and return a Pandas DataFrame.
    """

    # Get the root project folder
    project_root = Path(__file__).resolve().parents[2]

    # Create the dataset path
    dataset_path = project_root / "01_data_engineering" / "raw_data" / "Loan_Default.csv"

    print("=" * 60)
    print("NovaTrust Credit Risk Intelligence Platform")
    print("=" * 60)

    print(f"\nDataset Location:\n{dataset_path}")

    # Check if dataset exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"\nDataset not found:\n{dataset_path}")

    print("\nLoading dataset...")

    # Read CSV
    df = pd.read_csv(dataset_path)

    print("Dataset loaded successfully!\n")

    return df, project_root

def generate_data_quality_report(df, project_root):
    """
    Generate a Markdown data quality report.
    """

    report_path = (
        project_root
        / "01_data_engineering"
        / "reports"
        / "data_quality_report.md"
    )

    total_missing = df.isnull().sum()
    missing_columns = total_missing[total_missing > 0]

    with open(report_path, "w", encoding="utf-8") as report:

        report.write("# Data Quality Report\n\n")

        report.write("## Dataset Summary\n\n")

        report.write(f"- Rows: {df.shape[0]}\n")
        report.write(f"- Columns: {df.shape[1]}\n")

        memory = df.memory_usage(deep=True).sum() / 1024**2

        report.write(f"- Memory Usage: {memory:.2f} MB\n\n")

        report.write("## Missing Values\n\n")

        if len(missing_columns) == 0:

            report.write("No missing values found.\n")

        else:

            for column, count in missing_columns.items():

                percentage = (count / len(df)) * 100

                report.write(
                    f"- {column}: {count} ({percentage:.2f}%)\n"
                )

    print("\nData Quality Report Generated Successfully!")

if __name__ == "__main__":

    df, project_root = load_data()

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"\nRows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nFirst 5 Records")
    print(df.head())

    print("\nColumn Names")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nMemory Usage")

    memory = df.memory_usage(deep=True).sum() / 1024**2

    print(f"{memory:.2f} MB")

    generate_data_quality_report(df, project_root)