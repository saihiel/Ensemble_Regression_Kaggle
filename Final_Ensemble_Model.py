import numpy as np
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from pygam import LinearGAM, s, te, l
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import xgboost as xgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def process_and_split_training_data(seed, train_filename, test_filename):
    """
    Read training data and split into X and Y and further into train and validation sets using given random seed.
    :param seed: Value to set random seed at
    :param train_filename: Path to CSV file for training data
    :param test_filename: Path to CSV file for test data
    :return: Dictionary of train, validation and test data
    """
    ret = {}
    # Read training data
    train_df = pd.read_csv(train_filename, delimiter=',')
    # Split data in X & Y
    ret['y'] = train_df.iloc[:, 0]
    X = train_df.iloc[:, 1:]
    ret['X'] = X.drop(['X8'], axis=1)
    # Split training data into training and validation using given seed
    ret['Xtrain'], ret['XVal'], ret['Ytrain'], ret['YVal'] = train_test_split(X, ret['y'], random_state=seed)
    # Read test data
    Xtest = pd.read_csv(test_filename, delimiter=',')
    ret['Xtest'] = Xtest.drop(['id', 'X8'], axis=1)
    return ret

def get_GAM_predictions(Xtrain, Ytrain, Xtest):
    """
    Perform grid search and train Linear GAM model and return predictions for the test set.
    :param Xtrain: X values for training.
    :param Ytrain: Y values for training.
    :param Xtest:  X values for validation.
    :return: Predictions from Linear GAM model for test dataset
    """
    # Create an array of lambda values to search
    lams = np.logspace(-3, 20, 35)
    # GAM search requires numpy arrays
    Xtrain_np = np.array(Xtrain, dtype=np.float64)
    Ytrain_np = np.array(Ytrain, dtype=np.float64)

    # Linear Generalised Additive Model
    model = LinearGAM(
        s(99) + s(100)
        + l(3) + l(6) + l(8) + l(11) + l(7) + l(9) + l(12) + l(10)
        + l(14) + l(29) + l(15) + l(71) + l(17) + l(21) + l(107)
        + l(16) + l(68) + l(78) + l(61) + l(55) + l(31) + l(13)
        + l(37) + l(4) + l(5) + l(2) + te(4, 5) + te(68, 78)).gridsearch(Xtrain_np, Ytrain_np, lam=lams)
    return model.predict(Xtest)

def perform_feature_selection_for_XGB(model_parameters, Xtrain, Ytrain, XVal, YVal):
    """
    Create a subset of all features with the most predictive power.
    :param model_parameters: A Dictionary containing model parameters
    :param Xtrain: X values for training.
    :param Ytrain: Y values for training.
    :param XVal:  X values for validation.
    :param YVal:  Y values for validation.
    :return: The selection of features with the most predictive power.
    """
    selected_features = None
    # Create evaluation set for early stopping
    eval_set = [(Xtrain, Ytrain), (XVal, YVal)]
    # XGB model with given parameters
    xgb_model = xgb.XGBRegressor(**model_parameters)
    # Fit the model
    xgb_model.fit(Xtrain, Ytrain, early_stopping_rounds=25, eval_metric=['error', "rmse"], eval_set=eval_set,
                  verbose=False)
    # Perform Feature Selection
    for thresh in sort(xgb_model.feature_importances_):
        # Select features using threshold
        if (thresh < 0.008): continue
        selected_features = SelectFromModel(xgb_model, threshold=thresh, prefit=True, max_features=42)
        break
    return selected_features


def get_XGB_predictions(data, isTest, verbose):
    """
    Perform Feature Selection and train XGB model on the most informative features and return predictions on test set
    :param data: A Dictionary containing all the data as returned by 'process_and_split_training_data'
    :param isTest: Whether to make predictions on test data or validation data
    :param verbose: Level verbosity of printing for debugging.
    :return: Predictions made on the test set
    """
    params = {'max_depth': 2, 'eta': 0.05, 'n_estimators': 250, 'objective': 'reg:squarederror', 'eval_metric': 'rmse',
              'reg_lambda': 2, 'reg_alpha': 1.5, 'subsample': 0.7}
    selected_features = perform_feature_selection_for_XGB(params, data['Xtrain'], data['Ytrain'], data['XVal'], data['YVal'])
    # Define model with above parameters
    model = xgb.XGBRegressor(**params)

    # If in test mode then train on entire training and validation sets and make predictions on test set
    # Else train on only training set and make predictions on validation set
    X_train = data['Xtrain'] if isTest else data['X']
    Y_train = data['Ytrain'] if isTest else data['y']
    X_test = data['XVal'] if isTest else data['Xtest']

    # Transform training data to be the selected features only
    new_features_X_train = selected_features.transform(X_train)
    # Transform test data to be the selected features only
    new_features_Xtest = selected_features.transform(X_test)
    if verbose:
        print("Number of XGB Features Selected: ", new_features_X_train.shape[1])
    # Train model
    model.fit(new_features_X_train, Y_train)
    # Return Predictions
    return model.predict(new_features_Xtest)

def ensemble_model_predictions(isTest, seed, XGBp, GAMp, train_filename, test_filename, to_csv=False, verbose=False):
    """
    Train ensemble model and return predictions.
    :param isTest: Whether to make predictions on test data or validation data
    :param seed: Value to set random seed at
    :param XGBp: Weight given to the XGB models prediction
    :param GAMp: Weight given to the GAM models prediction
    :param train_filename: Path to CSV file for training data
    :param test_filename: Path to CSV file for test data
    :param to_csv: Whether to dump predictions to results.csv file
    :param verbose: Level verbosity of printing for debugging
    :return: Predictions made by the ensemble model
    """
    data = process_and_split_training_data(seed, train_filename, test_filename)
    # Linear GAM Predictions
    if isTest:
        # Train on entire train and validation set and make predictions on test set
        GAMPreds = get_GAM_predictions(data['X'], data['y'], data['Xtest'])
    else:
        # Train on only train and make predictions on validation set
        GAMPreds = get_GAM_predictions(data['Xtrain'], data['Ytrain'], data['XVal'])

    # Extreme Gradient Boosted Tree Regressor Predictions
    XGBpreds = get_XGB_predictions(data, isTest, verbose)

    # Calculate Ensemble Predictions
    Ensemble_Predictions = (XGBp * XGBpreds) + (GAMp * GAMPreds)

    if not isTest and verbose:
        print("Seed: ", seed)
        print("Ensemble RMSE: ", sqrt(mean_squared_error(data['YVal'], Ensemble_Predictions)))

    if to_csv:
        idCol = np.arange(1, 30001, dtype=np.int32)
        toPrint = np.array((idCol, Ensemble_Predictions))
        toPrint = toPrint.T
        np.savetxt("results.csv", toPrint, delimiter=",", header="id,y")
    return Ensemble_Predictions

predictions = ensemble_model_predictions(False, 1,0.5,0.5, 'trainingdata2.csv', 'test_predictors2.csv')