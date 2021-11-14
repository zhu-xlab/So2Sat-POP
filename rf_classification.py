# Random forest classification implementation, creates rf_model directory in the current directory to save all the logs
import glob
import os
import time
import _pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV

from constants import all_patches_mixed_train_part1, all_patches_mixed_test_part1, min_fimportance, kfold, n_jobs, param_grid, \
    covariate_list, current_dir_path
from utils import plot_feature_importance, validation_cls


def rf_classifier(file_name, ground_truth_col):
    """
    :param file_name: substring used for naming of the files
    :param ground_truth_col: column name to be used as ground truth
    :return:
    """
    print("Starting classification")
    # get all training cities
    all_train_cities = glob.glob(os.path.join(all_patches_mixed_train_part1, '*'))
    # prepare the training dataframe
    training_df = pd.DataFrame()
    for each_city in all_train_cities:
        city_csv = glob.glob(os.path.join(each_city, '*_features.csv'))[0]  # get the feature csv
        city_df = pd.read_csv(city_csv)
        training_df = training_df.append(city_df, ignore_index=True)  # append data from all the training cities

    # Get the dependent variables
    y = training_df[ground_truth_col]
    # Get the independent variables
    x = training_df[covariate_list]

    print("Starting training...\n")
    # Initialize the model
    rfmodel = RandomForestClassifier(n_estimators=500, oob_score=True, max_features='auto', n_jobs=-1, random_state=0)
    sel = SelectFromModel(rfmodel, threshold=min_fimportance)
    fited = sel.fit(x, y)
    feature_idx = fited.get_support()  # Get list of T/F for covariates for which OOB score is upper the threshold
    list_covar = list(x.columns[feature_idx])  # Get list of covariates with the selected features
    x = fited.transform(x)  # Update the dataframe with the selected features only

    # Instantiate the grid search model
    print("Starting Grid search with cross validation...\n")
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=param_grid, cv=kfold,
                               n_jobs=n_jobs, verbose=0)
    grid_search.fit(x, y)  # Fit the grid search to the data
    classifier = grid_search.best_estimator_  # Save the best classifier
    classifier.fit(x, y)  # Fit the best classifier
    # mean cross-validated score (OOB) and stddev of the best_estimator
    best_score = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
    best_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]

    rf_model_folder = os.path.join(current_dir_path, "rf_logs")  # path to the folder "rf_model"
    if not os.path.exists(rf_model_folder):
        os.mkdir(rf_model_folder)  # creates rf_logs folder inside the project folder

    model_folder = os.path.join(rf_model_folder, time.strftime("%Y%m%d-%H%M%S_") + file_name)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)  # creates folder inside the rf_logs folder, named as per time stamp and file_name

    model_name = time.strftime("%Y%m%d-%H%M%S_") + file_name  # model name
    rf_model_path = os.path.join(model_folder, model_name)  # path to saved model
    # save the best classifier
    with open(rf_model_path, 'wb') as f:
        _pickle.dump(classifier, f)
        f.close()

    # Get the log
    log = ""
    message = 'Parameter grid for Random Forest tuning :\n'
    for key in param_grid.keys():
        message += '    ' + key + ' : ' + ', '.join([str(i) for i in list(param_grid[key])]) + '\n'
    message += '    ' + 'min_fimportance' + ' : ' + str(min_fimportance) + '\n'
    log += message + '\n'

    # Print infos and save it in the log file - Tuned parameters
    message = 'Optimized parameters for Random Forest after grid search %s-fold cross-validation tuning :\n' % kfold
    for key in grid_search.best_params_.keys():
        message += '    %s : %s' % (key, grid_search.best_params_[key]) + '\n'
    log += message + '\n'

    message = "Mean cross-validated score (OOB) and stddev of the best_estimator : %0.3f (+/-%0.3f)" % (
        best_score, best_std) + '\n'
    log += message + '\n'

    # Print mean OOB and stddev for each set of parameters
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    message = "Mean cross-validated score (OOB) and stddev for every tested set of parameter :\n"

    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        message += "%0.3f (+/-%0.03f) for %r" % (mean, std, params) + '\n'
    log += message + '\n'

    # Print final model OOB
    message = 'Final Random Forest model run - internal Out-of-bag score (OOB) : %0.3f' % classifier.oob_score_
    oob_score = classifier.oob_score_
    log += message + '\n'

    # Save the log
    fout = open(os.path.join(model_folder, '%s_training_log.txt' % model_name), 'w')
    fout.write(log)
    fout.close()

    #################################################################################################

    # Start the predictions on completely unseen test data set
    print("Starting testing...\n")
    all_test_cities = glob.glob(os.path.join(all_patches_mixed_test_part1, '*'))   # get all test cities
    test_df = pd.DataFrame()
    for each_test_city in all_test_cities:
        test_city_csv = glob.glob(os.path.join(each_test_city, '*_features.csv'))[0]  # get the feature csv
        test_city_df = pd.read_csv(test_city_csv)
        test_df = test_df.append(test_city_df, ignore_index=True)  # append features from test cities

    # Get the population class
    y_test = test_df[ground_truth_col]
    # Get features
    x_test = test_df[list_covar]

    # load the trained model
    with open(rf_model_path, 'rb') as f:
        classifier = _pickle.load(f)

    # Get predictions
    prediction = classifier.predict(x_test)

    # Save the prediction
    df_pred = pd.DataFrame()
    df_pred["Grid_ID"] = test_df['GRD_ID']
    df_pred['Prediction'] = prediction
    df_pred['Reference'] = y_test
    pred_csv_path = os.path.join(model_folder, '%s_predictions.csv' % model_name)  # save the predictions to a csv
    df_pred.to_csv(pred_csv_path, index=False)

    #################################################################################################

    # Feature importances
    print("Creation of feature importance plot...\n")
    importances = classifier.feature_importances_  # get feature importance from the model
    path_plot = os.path.join(model_folder, "%s_RF_feature_importance" % model_name + 'new.png')  # path to saved plot
    plot_feature_importance(importances, x_test, path_plot)

    #################################################################################################

    # Calculating the metrics
    print('Calculating the metrics...')
    validation_log_path = os.path.join(model_folder, '%s_validation_log.txt' % model_name)
    classes = set(y_test)
    validation_cls(classes, pred_csv_path, validation_log_path)
    print('Finished classification \n \n')
