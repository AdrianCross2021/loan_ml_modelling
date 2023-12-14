# Loan Prediction Model README
This repository contains scripts for building and analyzing machine learning models for loan prediction. The primary goal is to predict whether a loan is likely to be bad based on historical loan data.

## Installation
To create the environment from the environment.yml file first run:

```console
conda env create -f environment.yml
```

Then to activate the environment run:

```console
conda activate wealthfront
```



## Scripts Overview
The scripts are designed so the clean_data.py file is run before either run_model_1.py or run_model_2.py are run. analysis_eda.py can be run at any time on a csv file.
### analysis_eda.py
This script performs an exploratory data analysis designed to provide insights into a given dataset. This produces an excel document with 4 sheets where each sheet contains summary data of 
* Sheet 1 - all columns with counts of null values
* Sheet 2 - categorical columns with number of unique values at various percentage thresholds
* Sheet 3 - numerical columns with a breakdown of the numbers which exist in each column (mean, median, max etc.)
* Sheet 4 - correlation matrix of all applicable columns

#### Usage:
```console
python eda_script.py --input_path <input_csv_path> --output_path <output_excel_path>
```
* input_path: Path to the raw CSV file for EDA analysis.
* output_path: Path to save the EDA results in an Excel file.

#### File outputs: 
* excel file with 3 tabs with the above summary


### clean_data.py
This script is designed to clean and process raw loan data, preparing it for machine learning model training.
#### Usage:
```console
python scripts/clean_data.py --input_path <input_csv_path> --output_path <output_csv_path>
```
* input_path: Path to the raw CSV file containing loan data.
* output_path: Path to save the cleaned and processed data.

#### File outputs: 
* cleaned csv data file


### run_model_1.py
This script is tailored for a binary classification task. It employs the XGBoost classifier to predict whether a loan is bad or not. Similar to the regression script, it involves data preprocessing, hyperparameter tuning, model training, prediction, and result analysis.

#### Usage:
```console
python scripts/run_model_1.py --input_path <input_csv_path> --output_predictions_path <output_predictions_path> --output_model_path <output_model_path> --output_analysis_folder <output_analysis_folder> [--tune_hyperparameters]
```

* input_path: Path to the preprocessed CSV file.
* output_predictions_path: Path to the output file containing predictions.
* output_model_path: Path to save the trained model.
* output_analysis_folder: Folder to store various analysis files.
* tune_hyperparameters: Optional flag to perform hyperparameter tuning.

#### File outputs: 
* pickle model file
* raw predictions data with predicted and actual results for the test data
* shap summary plot (png)
* confusion matrix (png)
* list of key metrics used to assess model performance (csv)

### run_model_2.py
This script is designed for a regression task. It uses the XGBoost regressor to predict the total payment of a loan. The main steps include data preprocessing, hyperparameter tuning, model training, prediction, and result analysis.

#### Usage:
```console
python scripts/run_model_2.py --input_path <input_csv_path> --output_predictions_path <output_predictions_path> --output_model_path <output_model_path> --output_analysis_folder <output_analysis_folder> [--tune_hyperparameters]
```

* input_path: Path to the preprocessed CSV file.
* output_predictions_path: Path to the output file containing predictions.
* output_model_path: Path to save the trained model.
* output_analysis_folder: Folder to store various analysis files.
* tune_hyperparameters: Optional flag to perform hyperparameter tuning.

#### File outputs: 
 * pickle model file
 * raw predictions data with predicted and actual results for the test data
 * shap summary plot (png)
 * confusion matrix (png)
 * list of key metrics used to assess model performance (csv)
  
