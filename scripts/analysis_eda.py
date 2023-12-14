import numpy as np
import logging
import fire
import pandas as pd
import os


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def run_eda(input_path: str, output_path: str):
    """
    Takes a csv input and performs exploratory data analysis then outputs an Excel file
    with 3 sheets corresponding to overall data, object data and numerical data.

    For all columns:
        1. Count of null values
        2. Data type breakdown - int, string, float, boolean

    If columns are object type:
        3. Breakdown of number of unique string values with varying thresholds

    If columns are numerical (or bool) type:
        4. mean
        5. skew
        6. quartile ranges (1%, 25%, 50%, 75%, 99%, MAX)
        7. Number above and below 0
        8. Correlation matrix
        9. Balance of the true/false target

    Args:
        input_path: path to input csv file for EDA analysis
        output_path: path to output EDA file
    Output: None
    """
    # Pull data from csv file
    logger.debug(f"Pulling in csv data from {input_path}")

    if not os.path.isfile(input_path):
        logger.error(f"Input file {input_path} not found")

    df = pd.read_csv(input_path)

    logger.debug(f"Performing EDA on data with {len(df.columns)} columns and {len(df)} rows")

    # Take all columns for initial check
    logger.debug(f"Performing initial EDA on all columns")
    null_counts = df.isnull().sum()
    # Create a DataFrame to store the results
    df_summary = pd.DataFrame({
        "data_type": df.dtypes,
        "num_null": null_counts.values,
        "percentage_null": (null_counts / len(df)) * 100
    })
    df_summary = df_summary.sort_values(by="num_null", ascending=False)

    # Take only non-numerical values for first half of analysis
    logger.debug(f"Performing EDA on non-numerical columns")

    df_categorical = df.select_dtypes(include=[object])

    logger.debug(f"Found {len(df_categorical.columns)} non numerical columns")

    df_categorical_summary = pd.DataFrame({
        # Get number of unique values with various thresholds
        "num_unique": df_categorical.nunique(),

        # Get list of unique values as a list of strings
        "unique_values_list": pd.Series({col: df_categorical[col].unique() for col in df_categorical})
    })

    # Loops over each threshold and returns values which have a count above that threshold
    # For example, if a unique value in a column is 7% of the values in a column, it will
    # be returned for a 5% threshold, but not a 10% threshold

    percentage_threshold_list = [0.001, 0.01, 0.05, 0.1]
    if len(df_categorical.columns) != 0:
        for percentage_threshold in percentage_threshold_list:
            df_categorical_column_list = []
            for column in df_categorical.columns:
                df_categorical_tmp = df_categorical[column].value_counts(normalize=True).loc[
                    df_categorical[column].value_counts(normalize=True) > percentage_threshold
                    ]
                df_categorical_column_list.append(
                    {
                        "column": column,
                        f"num_unique_{percentage_threshold}": df_categorical_tmp.nunique(),
                        f"unique_values_list_{percentage_threshold}": df_categorical_tmp.index.to_list()
                    }
                )

            # Add threshold, number of unique values and unique values list to the summary dataframe
            df_categorical_summary = pd.concat(
                [df_categorical_summary, pd.DataFrame(df_categorical_column_list).set_index("column")],
                axis=1
            )

    # take only numerical values for second half of analysis
    logger.debug(f"Performing EDA on boolean and numerical columns")

    df_numerical = df.select_dtypes(include=[np.number, bool])
    df_numerical = df_numerical.astype(float)  # Cast as float to allow for describe function

    logger.debug(f"{len(df_numerical.columns)} numerical and boolean columns found")

    df_numerical_summary = df_numerical.describe().transpose()
    df_numerical_summary["skew"] = df_numerical.skew()
    df_numerical_summary["1%"] = df_numerical.quantile(q=0.01)
    df_numerical_summary["99%"] = df_numerical.quantile(q=0.99)
    df_numerical_summary["nonzero"] = df_numerical.astype(bool).sum(axis=0) / df_numerical.shape[0]
    df_numerical_summary["above_zero"] = df_numerical[df_numerical > 0].count() / df_numerical.shape[0]
    df_numerical_summary["below_zero"] = df_numerical[df_numerical < 0].count() / df_numerical.shape[0]
    df_numerical_summary["num_unique_values"] = df_numerical.nunique()

    # Output correlation matrix
    logger.debug("Creating correlation matrix based on numerical columns")
    corr_matrix = df_numerical.corr()

    # Output DF to excel file
    logger.debug("Outputting excel document with EDA")
    with pd.ExcelWriter(output_path) as writer:
        df_summary.to_excel(writer, sheet_name="all_columns")
        df_categorical_summary.to_excel(writer, sheet_name="non-numerical_columns")
        df_numerical_summary.to_excel(writer, sheet_name="numerical_columns")
        corr_matrix.to_excel(writer, sheet_name="correlation_matrix")

if __name__ == "__main__":
    fire.Fire(run_eda)