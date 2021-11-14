"""
Generic data reader.
Returns all the patches with attributes of a given data and
their corresponding population count and population class labels

Note: Reads from both So2Sat POP Part1 and So2Sat POP Part2 data folders.
"""

import glob
import os
import rasterio

import numpy as np
import pandas as pd

from rasterio.enums import Resampling

from constants import all_patches_mixed_train_part1, all_patches_mixed_test_part1, all_patches_mixed_train_part2,\
    all_patches_mixed_test_part2, img_rows, img_cols, osm_features


def load_data(f_names, channels):
    """
    :param f_names: path to all the files of a data folder
    :param channels: number of channels corresponding to the data
    :return: all the instances of a data with its attributes
    """
    X = np.empty((len(f_names), img_rows, img_cols, channels))
    for i, ID in enumerate(f_names):

        # load tif file
        with rasterio.open(ID, 'r') as ds:
            image = ds.read(out_shape=(ds.count, img_rows, img_cols), resampling=Resampling.average)

        new_arr = np.empty([channels, img_rows, img_cols])

        # looping over all the channels
        for k, layer in enumerate(image):
            arr = layer
            new_arr[k] = arr
            X[i, ] = np.transpose(new_arr, (1, 2, 0))
    return X


def load_osm_data(f_names, channels):
    """
    :param f_names: path to all the files of osm_features data folder
    :param channels: number of channels corresponding to the osm_features data
    :return: all the instances of osm_features data with its attributes
    """
    X = np.empty((len(f_names), osm_features, channels))
    for i, ID in enumerate(f_names):
        # load csv
        df = pd.read_csv(ID, header=None)[1]
        # remove inf and Nan values
        df = df[df.notna()]
        df_array = np.array(df)
        df_array[df_array == np.inf] = 0

        new_arr = np.empty([channels, osm_features])

        new_arr[0] = df_array

        X[i, ] = np.transpose(new_arr, (1, 0))
    return X


def get_fnames_labels(folder_path, data):
    """
    :param folder_path: path to so2sat sub folder test/train
    :param data: name of the data folder, ex: 'lcz', 'lu', ...
    :return: all the instances of a data with its attributes and labels (population count & class) of each instance
    """
    city_folders = glob.glob(os.path.join(folder_path, "*"))  # list all the cities in folder_path
    f_names_all = np.array([])  # file names
    c_labels_all = np.array([])  # class labels
    p_count_all = np.array([])  # population counts
    for each_city in city_folders:
        data_path = os.path.join(each_city, data)  # path to the specifies data folder
        if data == 'dem':  # for dem data also, load the csv from So2Sat POP Part 1
            csv_path = os.path.join(each_city.replace('Part2', 'Part1'), each_city.split(os.sep)[-1:][0] + '.csv')
        else:
            csv_path = os.path.join(each_city, each_city.split(os.sep)[-1:][0] + '.csv')  # path to the cvs file of
            # the city

        city_df = pd.read_csv(csv_path)  # read csv as dataframe
        ids = city_df['GRD_ID']  # get the id of each patch
        pop = city_df['POP']  # corresponding pop count
        classes = city_df['Class']  # corresponding Class
        classes_str = [str(x) for x in classes]
        classes_paths = [data_path + '/Class_' + x + '/' for x in classes_str]
        for index in range(0, len(classes_paths)):
            if data == 'osm_features':  # osm features ends with '.csv'
                f_names = [classes_paths[index] + ids[index] + '_' + data + '.csv']  # creating full path for each id
            else:
                f_names = [classes_paths[index] + ids[index] + '_' + data + '.tif']  # creating full path for each id
            f_names_all = np.append(f_names_all, f_names, axis=0)  # append file names together
            pop_count = [pop[index]]
            p_count_all = np.append(p_count_all, pop_count, axis=0)   # append pop count together
            class_label = [classes[index]]
            c_labels_all = np.append(c_labels_all, class_label, axis=0)  # append class labels together

    if data.__contains__('sen2'):
        X = load_data(f_names_all, channels=3)  # load the data for sentinel-2 files

    if data == 'viirs' or data == 'lcz' or data == "dem":
        X = load_data(f_names_all, channels=1)  # load the data for viirs, lcz, dem files

    if data == 'lu':
        X = load_data(f_names_all, channels=4)  # load the data for lu files

    if data == 'osm_features':  # load the data for osm features files
        X = load_osm_data(f_names_all, channels=1)

    return X, p_count_all, c_labels_all


if __name__ == "__main__":

    # load all the files and their corresponding population count and class for "sen2_rgb_autumn" data in "train" folder
    X_train_sen2_rgb_autumn,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1,
                                                                               data='sen2_rgb_autumn')
    # load all the files and their corresponding population count and class for "sen2_rgb_autumn" data in "test" folder
    X_test_sen2_rgb_autumn,  y_test_count, y_test_class = get_fnames_labels(all_patches_mixed_test_part1,
                                                                            data='sen2_rgb_autumn')

    # load all the files and their corresponding population count and class for "sen2_rgb_summer" data in "train" folder
    X_train_sen2_rgb_summer,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1,
                                                                               data='sen2_rgb_summer')
    # load all the files and their corresponding population count and class for "sen2_rgb_summer" data in "test" folder
    X_test_sen2_rgb_summer,  y_test_count, y_test_class = get_fnames_labels(all_patches_mixed_test_part1,
                                                                            data='sen2_rgb_summer')

    # load all the files and their corresponding population count and class for "sen2_rgb_spring" data in "train" folder
    X_train_sen2_rgb_spring,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1,
                                                                               data='sen2_rgb_spring')
    # load all the files and their corresponding population count and class for "sen2_rgb_spring" data in "test" folder
    X_test_sen2_rgb_spring,  y_test_count, y_test_class = get_fnames_labels(all_patches_mixed_test_part1,
                                                                            data='sen2_rgb_spring')

    # load all the files and their corresponding population count and class for "sen2_rgb_winter" data in "train" folder
    X_train_sen2_rgb_winter,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1,
                                                                               data='sen2_rgb_winter')
    # load all the files and their corresponding population count and class for "sen2_rgb_winter" data in "test" folder
    X_test_sen2_rgb_winter,  y_test_count, y_test_class = get_fnames_labels(all_patches_mixed_test_part1,
                                                                            data='sen2_rgb_winter')

    # load all the files and their corresponding population count and class for "viirs" data in "train" folder
    X_train_viirs,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1, data='viirs')
    # load all the files and their corresponding population count and class for "viirs" data in "test" folder
    X_test_viirs,  y_test_count, y_test_class = get_fnames_labels(all_patches_mixed_test_part1, data='viirs')

    # load all the files and their corresponding population count and class for "lcz" data in "train" folder
    X_train_lcz,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1, data='lcz')
    # load all the files and their corresponding population count and class for "lcz" data in "test" folder
    X_test_lcz,  y_test_count, y_test_class = get_fnames_labels(all_patches_mixed_test_part1, data='lcz')

    # load all the files and their corresponding population count and class for "lu" data in "train" folder
    X_train_lu,  y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1, data='lu')
    # load all the files and their corresponding population count and class for "lu" data in "test" folder
    X_test_lu,  y_test_count, y_test_class = get_fnames_labels(all_patches_mixed_test_part1, data='lu')

    # load all the files and their corresponding population count and class for "dem" data in "train" folder
    X_train_dem, y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part2, data='dem')
    # load all the files and their corresponding population count and class for "dem" data in "test" folder
    X_test_dem, y_test_count, y_test_class = get_fnames_labels(all_patches_mixed_test_part2, data='dem')

    # load all the files and their corresponding population count and class for "osm" data in "train" folder
    X_train_osm, y_train_count, y_train_class = get_fnames_labels(all_patches_mixed_train_part1, data='osm_features')
    # load all the files and their corresponding population count and class for "osm" data in "test" folder
    X_test_osm, y_test_count, y_test_class = get_fnames_labels(all_patches_mixed_test_part1, data='osm_features')

    print('All instances are loaded')