import logging
import fire
import pandas as pd
import os
from datetime import datetime


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def clean_data(input_path: str, output_path: str):
    """
    Takes a csv input and cleans the data before it is used in the run_model_x phase. These changes are
    applied before doing a test-train split. This data should be put through EDA again before training stage.

    Steps performed:
    1. Remove invalid rows (all null)
    2. Add test_train label
    2. Drop invalid columns
    3. Extract numerical values from categorical values
    4. Define new columns: is_bad_loan and is_loan_complete

    Args:
        input_path: path to input csv file for feature engineering
        output_path: path to cleaned_file
    Output: None
    """
    # Pull data from csv file
    logger.debug(f"Pulling in csv data from {input_path}")

    if not os.path.isfile(input_path):
        logger.error(f"Input file {input_path} not found")

    df = pd.read_csv(input_path)

    # Step 1: Drop invalid rows
    logger.debug(f"Dropping invalid rows, starting count is {len(df)}")
    df = df.loc[
        df["loan_status"].notnull()
    ]
    logger.debug(f"Row count after removal is row count is {len(df)}")

    # Step 2: Drop invalid columns
    columns_to_drop = ["num_rate", "numrate", "interest_rate", "wtd_loans", "int_rate3", "int_rate2"]
    logger.debug(f"Dropping invalid columns {columns_to_drop}")
    df = df.drop(columns = columns_to_drop)

    # Step 3: Extract numerical values from categorical variables
    # Trim whitespace from all string columns
    # This is required as some object values have whitespace before and after the strings
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    # Map categorical values to numerical
    logger.debug(f"Extracting numerical values from categorical variables")
    df["term"] = df["term"].map({"60 months":60, "36 months": 36}).astype(float)
    df["emp_length"] = df["emp_length"].map(
        {
            "< 1 year": 0.5,
            "1 year": 1,
            "2 years": 2,
            "3 years": 3,
            "4 years": 4,
            "5 years": 5,
            "6 years": 6,
            "7 years": 7,
            "8 years": 8,
            "9 years": 9,
            "10+ years": 10,
        }
    ).astype(float)
    current_date = datetime.now()

    # Get days since date
    df["days_since_earliest_cr_line"] = (current_date - pd.to_datetime(df["earliest_cr_line"])).dt.days
    df = df.drop(columns = "earliest_cr_line")

    # Step 4: Define new columns based on loan status column
    df["is_bad_loan"] = df["loan_status"].map(
        {
            "Current": False,
            "Fully Paid": False,
            "In Grace Period": False,
            "Default": True,
            "Late (31-120 days)": True,
            "Late (16-30 days)": True,
            "Charged Off": True,
        }
    ).astype(bool)

    df["is_loan_complete"] = df["loan_status"].map(
        {
            "Current": False,
            "Fully Paid": True,
            "In Grace Period": False,
            "Default": True,
            "Late (31-120 days)": False,
            "Late (16-30 days)": False,
            "Charged Off": True,
        }
    ).astype(bool)

    df = df.drop(columns = "loan_status")

    # Output DF to excel file
    logger.debug(f"Outputting cleaned csv file to {output_path}")
    df.to_csv(output_path, index = False)


if __name__ == "__main__":
    fire.Fire(clean_data)