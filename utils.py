# contains reusable helper functions
import itertools

from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix, \
    balanced_accuracy_score, accuracy_score

import numpy as np
import pandas as pd
import math as ma
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

try:
    from osgeo import gdal
    from osgeo import osr, ogr
except:
    import gdal
    import osr


def raster2array(file_path, band):
    """
    :param file_path: path to the patch (raster)
    :param band: band number to read
    :return: array of raster values
    """
    raster = gdal.Open(file_path)
    band = raster.GetRasterBand(band)
    array = band.ReadAsArray()
    return array


def plot_confusion_matrix(cm, class_names, cm_path):
    """
    :param cm: a confusion matrix of integer classes, (array, shape = [n, n])
    :param class_names: String names of the integer classes, (array, shape = [n])
    :param cm_path: path to saved confusion matrix
    :return: matplotlib figure containing the plotted confusion matrix.
    """
    figure = plt.figure(figsize=(8, 8))
    plt.tight_layout()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(cm_path, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()
    return figure


def plot_normalized_confusion_matrix(cm, class_names, cm_path):
    """
    :param cm: a confusion matrix of integer classes, (array, shape = [n, n])
    :param class_names: String names of the integer classes, (array, shape = [n])
    :param cm_path: path to saved normalized confusion matrix
    :return: matplotlib figure containing the plotted confusion matrix.
    """
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    plt.tight_layout()
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(cm_path, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()
    return figure


def validation_cls(classes, pred_csv_path, validation_log_path):
    """
    :param pred_csv_path: path to saved csv file that contains predictions
    :param validation_log_path:  path to saved log file that contains validation metrics
    :return: Calculates the validation metrics and creates the validation log
    """
    df = pd.read_csv(pred_csv_path)
    val_targ = df['Reference']
    val_predict = df['Prediction']

    val_f1_micro = round(f1_score(val_targ, val_predict, average='micro'), 4)
    val_recall_micro = round(recall_score(val_targ, val_predict, average='micro'), 4)
    val_precis_micro = round(precision_score(val_targ, val_predict, average='micro'), 4)
    val_f1_macro = round(f1_score(val_targ, val_predict, average='macro'), 4)
    val_recall_macro = round(recall_score(val_targ, val_predict, average='macro'), 4)
    val_precis_macro = round(precision_score(val_targ, val_predict, average='macro'), 4)
    val_cm = confusion_matrix(val_targ, val_predict)
    accuracy = accuracy_score(val_targ, val_predict)
    balanced_accuracy = balanced_accuracy_score(val_targ, val_predict)
    classwise_accuracy = confusion_matrix(val_targ, val_predict, normalize="true").diagonal()
    print("Please check log at {}  \n".format(validation_log_path))
    with open(validation_log_path, 'a') as f:
        f.writelines(
            "Evaluation metrics on test data \n F1_macro: {} \n F1_micro: {} \n Precision_macro: {} \n "
            "Precision_micro: {} \n Recall_macro: {} \n Recall_micro: {} \n Accuracy: {}"
            "\n Balanced_accuracy: {} \n Class wise accuracy: {}".format(val_f1_macro, val_f1_micro,
                                                                                 val_precis_macro, val_precis_micro,
                                                                                 val_recall_macro, val_recall_micro,
                                                                                 accuracy, balanced_accuracy,
                                                                                 classwise_accuracy))
    cm_path = validation_log_path.replace('_validation_log.txt', '_cm.png')
    norm_cm_path = validation_log_path.replace('_validation_log.txt', '_cm_norm.png')
    # print(classification_report(val_targ, val_predict))

    labels = sorted(classes)
    labels_names = []
    for each in labels:
        labels_names.append('Class_' + str(each))

    _ = plot_confusion_matrix(val_cm, class_names=labels_names, cm_path=cm_path)
    _ = plot_normalized_confusion_matrix(val_cm, class_names=labels_names, cm_path=norm_cm_path)
    _ = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]


def validation_reg(pred_csv_path, oob_score, validation_csv_path):
    """
    :param pred_csv_path: Path to csv file, has saved predictions and expected values for test data
    :param oob_score: Final OOB score of the model
    :param validation_csv_path: path to new csv which will contain the metric for further evaluations
    """
    log_list = []
    df = pd.read_csv(pred_csv_path)
    output_log_validation = validation_csv_path.replace('.csv', '_log.txt')
    # Compute error
    df['error'] = df['Prediction'] - df['Reference']
    # Compute squared error raster
    df['sqerror'] = df['error'] ** 2
    # Compute absolute error raster
    df['abserror'] = abs(df['error'])

    # Compute overall validation statistics #
    rmse = ma.sqrt(round(df['sqerror'].mean(), 2))  # Compute RMSE (Root mean squared error)
    mean_ref = df['Reference'].mean()  # Compute mean reference population per admin unit
    prct_rmse = (rmse / mean_ref) * 100  # Compute %RMSE
    MAE = df['abserror'].mean()  # Compute MAE (Mean absolute error)
    TAE = df['abserror'].sum()  # Compute TAE (Total absolute error)
    POPTOT = df['Reference'].sum()  # Compute Total reference population
    PREDTOT = df['Prediction'].sum()  # Compute Total predicted population
    corr = np.corrcoef(df['Reference'], df['Prediction'])[0, 1]  # Get correlation value
    r_squared = (corr**2)   # Get r-squared value

    # Outputs print and log
    log = ""
    log += "Total reference population = %s \n" % round(POPTOT, 1)
    log += "Total predicted population = %s \n" % round(PREDTOT, 1)
    log += '\n'
    log += "Final Random Forest model run - internal Out-of-bag score (OOB) = %s \n" % round(oob_score, 3)
    log += "Mean absolute error of prediction (MAE) = %s \n" % round(MAE, 3)
    log += "Root mean squared error of prediction (RMSE) = %s \n" % round(rmse, 3)
    log += "Root mean squared error of prediction in percentage (Percent_RMSE) = %s \n" % round(prct_rmse, 3)
    log += "Total absolute error (TAE) = %s \n" % round(TAE, 3)
    log += "R squared = %s \n" % round(r_squared, 3)
    log_list.extend([[round(oob_score, 3), round(MAE, 3), round(prct_rmse, 3), round(r_squared, 3)]])
    print("Please check log at {} \n ".format(output_log_validation))
    fout = open(output_log_validation, 'w')
    fout.write(log)
    fout.close()


def plot_feature_importance(importances, x_test, path_plot):
    """
    :param importances: array of feature importance from the model
    :param x_test: data frame for test cities
    :param path_plot: path to feature importance plot
    :return: Create and save the feature importance plot
    """
    indices = np.argsort(importances)[::-1]
    indices = indices[:12]  # get indices of only top 12 features
    x_axis = importances[indices][::-1]
    idx = indices[::-1]
    y_axis = range(len(x_axis))
    Labels = []
    for i in range(len(x_axis)):
        Labels.append(x_test.columns[idx[i]])  # get corresponding labels of the features
    y_ticks = np.arange(0, len(x_axis))
    fig, ax = plt.subplots()
    ax.barh(y_axis, x_axis)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(Labels)
    ax.set_title("Random Forest Feature Importances")
    fig.tight_layout()
    plt.savefig(path_plot, bbox_inches='tight', dpi=400)  # Export in .png file (image)
    plt.close()  # Avoid the plot to be displayed
