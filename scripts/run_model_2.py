import logging
import fire
from xgboost import XGBRegressor
import pandas as pd
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def run_model(
        input_path: str,
        output_predictions_path: str,
        output_model_path: str,
        output_analysis_folder: str,
        tune_hyperparameters: bool = True,
    ):
    """
    Takes a cleaned csv, preprocesses the data, trains an xgboost classifier model and then predicts against the test
    set. Each of steps 1-4 are separated into sub functions.

    Steps performed:
    1. Preprocess data (preprocess)
    2. Train model (train_model)
    3. Make predictions (make_predictions)
    4. Analyze model and output files (analyze_results) - outputs are defined by command line arguments


    Args:
        input_path: path to input preprocessed csv file
        output_predictions_path: path to output analysis of out, file contains id, prediction and actual result
        output_model_path: path to output pickled model file
        output_analysis_folder: path to drop multiple analysis files in
        tune_hyperparameters: if true performs hyperpameter tuning


    Returns:
        None
    """

    # Pull data from csv file
    logger.debug(f"Pulling in csv data from {input_path}")

    if not os.path.isfile(input_path):
        logger.error(f"Input file {input_path} not found")

    df_cleaned = pd.read_csv(input_path).set_index("id")

    # Preprocess
    logger.debug("Performing preprocessing on cleaned data")
    X_train, X_test, y_train, y_test = preprocess(df_cleaned)

    # Hyperparameter tuning - Use mean absolute error to look at the absolute difference between target and actual
    if tune_hyperparameters:
        logger.debug("Performing hyperparameter tuning")
        hyperparameters = perform_hyperparameter_tuning(X_train, y_train, success_parameter="neg_mean_absolute_error")
    else:
        hyperparameters = None

    # Train model
    logger.debug(f"Performing model training on {len(X_train)} rows and {len(X_train.columns)} features")
    model = train_model(X_train, y_train, hyperparameters, evaluation_metric = "mae")

    # Predict on test data
    logger.debug("Making predictions on test dataset and running analyses")
    predictions = make_predictions(model, X_test, y_test)

    # Perform and output analyses
    logger.debug(f"outputting analyses to {output_analysis_folder} folder")
    analyze_model(predictions, X_test, model, output_analysis_folder, output_predictions_path, output_model_path)

def preprocess(df_cleaned: pd.DataFrame):
    """
    Takes a cleaned csv input and performs test-train split, target calculation and one hot encoding

    Steps performed:
    1. Filter for completed loans only
    2. Define a target variable
    3. Perform test-train split
    4. one-hot encode columns

    Args:
        df_cleaned: cleaned data file

    Returns:
        X_train: training features
        X_test: testing features
        y_train: training target
        y_test: testing target
    """

    # Take only completed loans
    df_cleaned = df_cleaned.loc[
        df_cleaned["is_loan_complete"] == True
    ]
    # Define features and target
    y = df_cleaned["total_pymnt"] - df_cleaned["funded_amnt"]

    # Remove columns which indicate a good or bad loan after the loan has complete - these cause data leakage
    X = df_cleaned.drop(columns = ["total_pymnt","total_rec_prncp", "total_rec_int", "out_prncp", "is_loan_complete",
                          "is_bad_loan"])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create a sklearn pipeline to perform one hot encoding
    one_hot_encoded_columns = ["addr_state", "purpose", "home_ownership"]
    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(
            transformers=[
                ("onehot", OneHotEncoder(), one_hot_encoded_columns)
            ],
            remainder="passthrough"
        )),
    ])

    # Fit and transform the training data
    X_train = pipeline.fit_transform(X_train)

    # Transform the testing data
    X_test = pipeline.transform(X_test)

    # Extract feature names from the last step of the pipeline (the regressor)
    # needed to preserve column names
    column_names = pipeline.get_feature_names_out()
    X_train = pd.DataFrame(X_train.toarray(), columns = column_names)
    X_test = pd.DataFrame(X_test.toarray(), columns=column_names)


    return X_train, X_test, y_train, y_test


def perform_hyperparameter_tuning(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        parameters_to_test: dict = None,
        num_folds: int = 5,
        success_parameter: str = None,
):
    """
    Perform basic hyperparameter tuning using RandomizedSearchCV (randomized search) for an XGBoost regressor.

    Args:
        X_train: features for the training subset of data
        y_train: target for the training subset of data
        parameters_to_test: dictionary of hyperparameters to test
        num_folds: number of cross validation folds
        success_parameter: what success parameter to use


    Returns:
        best_hyperparameters: Dictionary of the best hyperparameters for the model

    """

    logger.debug(f"defining regressor")
    xgb_regressor = XGBRegressor()

    # Define the hyperparameter distribution for randomized search
    # This is a relatively small search in the interest of efficiency/time
    if parameters_to_test is None:
        parameters_to_test = {
            "eta": [0.1, 0.3, 0.5],
            "gamma": [0, 0.1, 0.3],
            "lambda": [0, 1, 2],
            "alpha": [0, 0.1, 0.3],
            "max_depth": [4, 6, 8],
        }
    logger.debug(f"Performing hyperparameter tuning for parameters {parameters_to_test}")

    # Perform randomized search - use f1 score as the data is unbalanced
    randomized_search = RandomizedSearchCV(xgb_regressor, param_distributions=parameters_to_test, n_iter=10,
                                           cv=num_folds, scoring=success_parameter, verbose=2, n_jobs=-1,
                                           random_state=42)

    test1 =pd.DataFrame(X_train)
    test2 = pd.DataFrame(y_train)
    randomized_search.fit(X_train, y_train)

    # Print the best hyperparameters
    logger.debug("Best Hyperparameters:")
    logger.debug(f"{randomized_search.best_params_}")

    # Return the best parameters
    return randomized_search.best_params_

def train_model(X_train: pd.DataFrame,
                y_train: pd.DataFrame,
                hyperparameters=None,
                evaluation_metric = None
    ):
    """
    Takes (optional) tuned hyperparameter, x_train and y_train then outputs a pickle model

    Args:
        X_train: features for the training subset of data
        y_train: target for the training subset of data
        hyperparameters: tuned hyperparameters
        evaluation_metric: metric to optimize

    Output: pickled model file
    """

    # Define a regressor
    logger.debug(f"defining regressor")

    # Read in hyperparameters if they are passed to the function, otherwise use default
    if hyperparameters is None:
        logger.debug(f"Training model with default hyperparameters")
        xgb_regressor = XGBRegressor(eval_metric = evaluation_metric)
    else:
        logger.debug(f"Training model with tuned hyperparameters")
        xgb_regressor = XGBRegressor(eval_metric = evaluation_metric, **hyperparameters)

    # Trains model using logloss function
    xgb_regressor.fit(X_train, y_train)

    return xgb_regressor

def make_predictions(
        model: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test
    ):
    """
    Takes model file and test columns and creates a dataframe of predictions for the test values.


    Args:
        X_test: features for the test subset of data
        y_test: target for the test subset of data
        model: trained model

    Returns:
        predictions: predictions dataframe
    """

    # Create predictions dataframe using actual values
    predictions = pd.DataFrame(y_test, columns = ["actual"])
    #predictions = predictions.rename(columns={0:"actual"})

    # Make predictions on test
    logger.debug(f"Predicting on test dataset with {len(predictions)} rows")
    y_pred = model.predict(X_test)
    predictions["predicted"] = y_pred

    # Define a good loan by whether it is positive or not
    predictions["bad_loan_actual"] = predictions["actual"] < 0
    predictions["bad_loan_predicted"] = predictions["predicted"] < 0

    return predictions


def analyze_model(
    predictions: pd.DataFrame,
    X_test: pd.DataFrame,
    model,
    output_analysis_folder: str,
    output_predictions_path: str,
    output_model_path: str,
    cm_file_name = "confusion_matrix.png",
    shap_file_name="shap_summary_plot.png",
    metric_file_name="metrics_results.csv",
    ):
    """
    Takes model file, test data and outputs various analyses to assess model success. Files are output according
    to defined paths.

    Analyses output:
    1. Confusion Matrix
    2. Shap Values
    3. Success metrics (accuracy, F1 score, precision, recall, uplift)
    4. Raw prediction output
    5. Raw model output (as pickle file)


    Args:
        predictions: dataframe of test predictions
        X_test: test features
        model: model file
        output_analysis_folder: folder specified to drop analyses files
        output_predictions_path: path specified to drop raw predictions
        output_model_path: path specified to drop model file
        cm_file_name: confusion matrix image file name
        shap_file_name: shap image file name
        metric_file_name: success metrics csv file name

    Returns: None
    """

    # Output confusion matrix (png)
    logger.debug(f"Outputting confusion matrix as {cm_file_name}")
    conf_matrix = confusion_matrix(predictions["bad_loan_actual"], predictions["bad_loan_predicted"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.savefig(os.path.join(output_analysis_folder, cm_file_name))
    plt.clf()

    # Output shap values (png)
    logger.debug(f"Outputting shap summary plot as {shap_file_name}")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, feature_names=X_test.columns, features = X_test, show = False)
    plt.savefig(os.path.join(output_analysis_folder, shap_file_name))
    plt.clf()

    # Output model success metrics (csv)
    # Classification metrics using <0 as a bad loan, >0 as good loan
    logger.debug(f"Outputting success metrics as {metric_file_name}")
    accuracy = accuracy_score(predictions["bad_loan_actual"], predictions["bad_loan_predicted"])
    precision = precision_score(predictions["bad_loan_actual"], predictions["bad_loan_predicted"])
    recall = recall_score(predictions["bad_loan_actual"], predictions["bad_loan_predicted"])
    f1 = f1_score(predictions["bad_loan_actual"], predictions["bad_loan_predicted"])

    # Regressor metrics using predicted values
    mse = mean_squared_error(predictions["actual"], predictions["predicted"])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(predictions["actual"], predictions["predicted"])

    # Calculate uplift by comparing actual profit and profit when removing predicted bad loans
    # The idea behind this is to compare actual potential profitability with and without applying the model
    profit_actual = predictions["actual"].sum()
    profit_filtering_bad_loans = predictions.loc[predictions["predicted"] < 0]["actual"].sum()
    uplift = float(profit_filtering_bad_loans - profit_actual)/ float(profit_actual)


    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score","mean squared error", "root mean squared error",
                   "mean absolute error","uplift"],
        "Score": [accuracy, precision, recall, f1, mse, rmse, mae, uplift]
    })

    metrics_df.to_csv(os.path.join(output_analysis_folder, metric_file_name), index=False)

    # Output raw data (csv)
    logger.debug(f"Outputting predictions to {output_predictions_path}")
    predictions.to_csv(output_predictions_path)

    # Output Model
    logger.debug(f"Outputting model to {output_model_path}")
    with open(output_model_path, "wb") as model_file:
        pickle.dump(model, model_file)



if __name__ == "__main__":
    fire.Fire(run_model)