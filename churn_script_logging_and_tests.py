# library doc string
'''
Author: goldin2008
Date: December, 2021
This module implements the main script for the customer churn project with clean code
'''
import os
import logging
import yaml
import churn_library as cls

os.environ['QT_QPA_PLATFORM']='offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    perform_eda(df)

    # 1. Check if the list is empty or not
    # 2. Check if all files exist
    try:
        # Getting the list of directories
        path = config['eda']['save_path']
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        assert os.path.isfile('images/eda/churn_distribution.png')
        assert os.path.isfile('images/eda/customer_age_distribution.png')
        assert os.path.isfile('images/eda/marital_status_distribution.png')
        assert os.path.isfile('images/eda/total_transaction_distribution.png')
        assert os.path.isfile('images/eda/heatmap.png')
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing perform_eda: The eda file wasn't found")
        raise err


def test_encoder_helper(encoder_helper, df):
    '''
    test encoder helper
    '''
    cat_columns = config['data']['categorical_features']
    df = encoder_helper(df, cat_columns, 'Churn')

    try:
        for col in cat_columns:
            encoded_col = col + '_Churn'
            assert col in df.columns
            assert encoded_col in df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe miss some categorical column")
        return err

    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[1] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: "
                      "feature_engineering fail.")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(X_train, X_test, y_train, y_test)

    # Models
    # 1. Check if the list is empty or not
    # 2. Check if all files exist
    try:
        # Getting the list of directories
        path = config['models']['save_path']
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        assert os.path.isfile('models/rfc_model.pkl')
        assert os.path.isfile('models/logistic_model.pkl')
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: Model files not found")
        raise err

    # Results
    # 1. Check if the list is empty or not
    # 2. Check if all files exist
    try:
        # Getting the list of directories
        path = config['results']['save_path']
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        assert os.path.isfile('images/results/roc_curve_result.png')
        assert os.path.isfile('images/results/feature_importances.png')
        assert os.path.isfile('images/results/feature_importances_shap.png')
        assert os.path.isfile('images/results/rf_report.png')
        assert os.path.isfile('images/results/logistic_report.png')
    except FileNotFoundError as err:
        logging.error("Testing train_models: Results image files not found")
        raise err


def run_test():
    """
    Main function to run test cases
    """
    logging.info("TESTING import_data")
    df_data = test_import(cls.import_data)

    logging.info("TESTING perform_eda function")
    test_eda(cls.perform_eda, df_data)

    logging.info("TESTING encoder_helper function")
    df_data = test_encoder_helper(cls.encoder_helper, df_data)

    logging.info("TESTING perform_feature_engineering function")
    X_train, X_test, y_train, y_test = test_perform_feature_engineering(
        cls.perform_feature_engineering, df_data)

    logging.info("TESTING test_train_and_evaluate_model function")
    test_train_models(cls.train_models,
                      X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run_test()
