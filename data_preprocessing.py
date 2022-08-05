# Process the data and creates the features for the training of random forest model.
# In each city folder creates a city_name_features.csv file with 125 features
import glob
import os

import numpy as np
import pandas as pd
import math as ma

from utils import raster2array


def get_id_response_var(all_patches, city_df):
    """
    :param all_patches: list of all patches
    :param city_df: data frame for the city
    :return: grid id, population class, population count, population density, log (population density)
    """
    id_list = []
    class_list = []
    pop_count = []
    pop_dens = []
    log_pop_dens = []
    for each_patch in all_patches:
        id = os.path.split(each_patch)[1].rsplit('_')[0]  # ID of the grid cell
        id_list.append(id)
        id_index = city_df.index[city_df['GRD_ID'] == id].tolist()[0]
        pop = city_df['POP'][id_index]  # get the absolute population count
        pop_den = pop / 1000000   # population density per 1000,000 m-sq (1km-sq)
        if pop_den == 0:
            log_pop_den = 0
        else:
            log_pop_den = ma.log(pop_den)   # calculating log of population density
        pop_count.append(pop)
        pop_dens.append(pop_den)
        log_pop_dens.append(log_pop_den)
        class_patch = each_patch.split(os.sep)[-2].rsplit('_')[1]   # corresponding class for the patch
        class_list.append(class_patch)
    return id_list, class_list, pop_count, pop_dens, log_pop_dens


def mean_med_std_max_min(band):
    """
    :param band: r, g, b band od sen2 patch
    :return: mean, median, std, max, min of sen2 patch
    """
    sen2_mean_band = np.mean(band)
    sen2_med_band = np.median(band)
    sen2_std_band = np.std(band)
    sen2_max_band = np.max(band)
    sen2_min_band = np.min(band)
    return sen2_mean_band, sen2_med_band, sen2_std_band, sen2_max_band, sen2_min_band


def sen2_features(all_patches):
    """
    :param all_patches: list of all patches
    :return: mean, median, std, max, min features for each r, g, b bands (5 X 3 = 15 features)
    """
    # sen2 feature lists
    sen2_mean_r_feat = []
    sen2_mean_g_feat = []
    sen2_mean_b_feat = []
    sen2_med_r_feat = []
    sen2_med_g_feat = []
    sen2_med_b_feat = []
    sen2_std_r_feat = []
    sen2_std_g_feat = []
    sen2_std_b_feat = []
    sen2_max_r_feat = []
    sen2_max_g_feat = []
    sen2_max_b_feat = []
    sen2_min_r_feat = []
    sen2_min_g_feat = []
    sen2_min_b_feat = []

    # iterate over each sen2 patch
    for each_patch in all_patches:
        # get the r, g, b bands
        r = raster2array(each_patch, 4)
        g = raster2array(each_patch, 3)
        b = raster2array(each_patch, 2)
        # get features for r band
        sen2_mean_r, sen2_med_r, sen2_std_r, sen2_max_r, sen2_min_r = mean_med_std_max_min(band=r)
        # get features for g band
        sen2_mean_g, sen2_med_g, sen2_std_g, sen2_max_g, sen2_min_g = mean_med_std_max_min(band=g)
        # get features for b band
        sen2_mean_b, sen2_med_b, sen2_std_b, sen2_max_b, sen2_min_b = mean_med_std_max_min(band=b)
        # list of sen2 mean feature for r,g,b bands
        sen2_mean_r_feat.append(sen2_mean_r)
        sen2_mean_g_feat.append(sen2_mean_g)
        sen2_mean_b_feat.append(sen2_mean_b)
        # list of sen2 median feature for r,g,b bands
        sen2_med_r_feat.append(sen2_med_r)
        sen2_med_g_feat.append(sen2_med_g)
        sen2_med_b_feat.append(sen2_med_b)
        # list of sen2 std feature for r,g,b bands
        sen2_std_r_feat.append(sen2_std_r)
        sen2_std_g_feat.append(sen2_std_g)
        sen2_std_b_feat.append(sen2_std_b)
        # list of sen2 max feature for r,g,b bands
        sen2_max_r_feat.append(sen2_max_r)
        sen2_max_g_feat.append(sen2_max_g)
        sen2_max_b_feat.append(sen2_max_b)
        # list of sen2 min feature for r,g,b bands
        sen2_min_r_feat.append(sen2_min_r)
        sen2_min_g_feat.append(sen2_min_g)
        sen2_min_b_feat.append(sen2_min_b)

    return sen2_mean_r_feat, sen2_mean_g_feat, sen2_mean_b_feat, sen2_med_r_feat, sen2_med_g_feat, sen2_med_b_feat, \
        sen2_std_r_feat, sen2_std_g_feat, sen2_std_b_feat, sen2_max_r_feat, sen2_max_g_feat, sen2_max_b_feat, \
        sen2_min_r_feat, sen2_min_g_feat, sen2_min_b_feat


def average_mean_features(file_path, band):
    """
    :param file_path: path to patch file
    :param band: band number to read, required for multi band data
    :return: mean and max of the patch
    """
    raster_array = raster2array(file_path, band)
    raster_mean = np.mean(raster_array)
    raster_max = np.max(raster_array)
    return raster_mean, raster_max


def lu_features(file_path, band):
    """
    :param file_path: path to patch file
    :param band: band number to read, required for multi band data
    :return: total area of a patch that belongs to a particular band
    """
    lu_array = raster2array(file_path, band)
    lu_total_area = np.sum(lu_array)
    return lu_total_area


def feature_engineering(all_patches_mixed_path):
    """
    Creates features csv for each city, named city_name_features.csv
    :param all_patches_mixed_path: path to the folder that contains the cities to process
    :return: None
    """
    # preparing features for part 1 of dataset
    if all_patches_mixed_path.endswith("Part1"):
        print('Preparing features for So2sat Part1 \n')
        all_folders = glob.glob(os.path.join(all_patches_mixed_path, '*'))
        for each_folder in all_folders:
            all_cities = glob.glob(os.path.join(each_folder, '*'))
            for each_city in all_cities:
                # declare lists for input data source
                lu_1_feat = []
                lu_2_feat = []
                lu_3_feat = []
                lu_4_feat = []
                lcz_feat = []
                viirs_mean_feat = []
                viirs_max_feat = []
                osm_feat = []

                city_name = os.path.split(each_city)[1]  # get the name of the city from the city path
                feature_csv_file = os.path.join(each_city, city_name + '_features.csv')  # create feature csv for city
                all_data = glob.glob(os.path.join(each_city, '*'))  # get all the data folders
                city_csv_file = os.path.join(each_city, city_name + '.csv')  # get the city's csv
                city_df = pd.read_csv(city_csv_file)  # data frame for the city

                for each_data in all_data:   # for each data folder in a city
                    all_patches = []    # list to all patches
                    if each_data.endswith('.csv'):   # skip the csv file, get only data folders
                        # skip the file
                        continue
                    all_classes = glob.glob(os.path.join(each_data, '*'))  # get all class folders in data folder
                    for each_class in all_classes:
                        class_patches = glob.glob(os.path.join(each_class, '*'))
                        for x in class_patches:
                            all_patches.append(x)  # get list of all the city patches

                    if each_data.endswith('lu'):  # process lu data
                        for each_patch in all_patches:
                            area_lu_1 = lu_features(each_patch, band=1)  # area that belongs to band 1 (commercial) of lu patch
                            area_lu_2 = lu_features(each_patch, band=2)  # area that belongs to band 2 (industrial) of lu patch
                            area_lu_3 = lu_features(each_patch, band=3)  # area that belongs to band 3 (residential) of lu patch
                            area_lu_4 = lu_features(each_patch, band=4)  # area that belongs to band 4 (other) of lu patch
                            lu_1_feat.append(area_lu_1)  # lu band 1 area feature list
                            lu_2_feat.append(area_lu_2)  # lu band 2 area feature list
                            lu_3_feat.append(area_lu_3)  # lu band 3 area feature list
                            lu_4_feat.append(area_lu_4)  # lu band 4 area feature list

                    if each_data.endswith('lcz'):   # process lcz data
                        for each_patch in all_patches:
                            lcz_array = raster2array(each_patch, 1)
                            lcz_class = np.argmax(np.bincount(lcz_array.flatten()))  # get the majority lcz class of the patch
                            lcz_feat.append(lcz_class)  # majority lcz class feature list

                    if each_data.endswith('viirs'):  # process nightlights data
                        for each_patch in all_patches:
                            viirs_mean, viirs_max = average_mean_features(each_patch, band=1)  # get mean and max of viirs patch
                            viirs_mean_feat.append(viirs_mean)   # viirs mean feature list
                            viirs_max_feat.append(viirs_max)  # viirs max feature list

                    if each_data.endswith('autumn'):  # process sen2 autumn data
                        sen2_aut_mean_r_feat, sen2_aut_mean_g_feat, sen2_aut_mean_b_feat, sen2_aut_med_r_feat, \
                         sen2_aut_med_g_feat, sen2_aut_med_b_feat, sen2_aut_std_r_feat, sen2_aut_std_g_feat, \
                         sen2_aut_std_b_feat, sen2_aut_max_r_feat, sen2_aut_max_g_feat, sen2_aut_max_b_feat, \
                         sen2_aut_min_r_feat, sen2_aut_min_g_feat, sen2_aut_min_b_feat = sen2_features(all_patches)

                    if each_data.endswith('spring'):  # process sen2 spring data
                        sen2_spr_mean_r_feat, sen2_spr_mean_g_feat, sen2_spr_mean_b_feat, sen2_spr_med_r_feat, \
                         sen2_spr_med_g_feat, sen2_spr_med_b_feat, sen2_spr_std_r_feat, sen2_spr_std_g_feat, \
                         sen2_spr_std_b_feat, sen2_spr_max_r_feat, sen2_spr_max_g_feat, sen2_spr_max_b_feat, \
                         sen2_spr_min_r_feat, sen2_spr_min_g_feat, sen2_spr_min_b_feat = sen2_features(all_patches)

                    if each_data.endswith('summer'):  # process sen2 summer data
                        sen2_sum_mean_r_feat, sen2_sum_mean_g_feat, sen2_sum_mean_b_feat, sen2_sum_med_r_feat, \
                         sen2_sum_med_g_feat, sen2_sum_med_b_feat, sen2_sum_std_r_feat, sen2_sum_std_g_feat, \
                         sen2_sum_std_b_feat, sen2_sum_max_r_feat, sen2_sum_max_g_feat, sen2_sum_max_b_feat, \
                         sen2_sum_min_r_feat, sen2_sum_min_g_feat, sen2_sum_min_b_feat = sen2_features(all_patches)

                    if each_data.endswith('winter'):  # process sen2 winter data
                        sen2_win_mean_r_feat, sen2_win_mean_g_feat, sen2_win_mean_b_feat, sen2_win_med_r_feat, \
                         sen2_win_med_g_feat, sen2_win_med_b_feat, sen2_win_std_r_feat, sen2_win_std_g_feat, \
                         sen2_win_std_b_feat, sen2_win_max_r_feat, sen2_win_max_g_feat, sen2_win_max_b_feat, \
                         sen2_win_min_r_feat, sen2_win_min_g_feat, sen2_win_min_b_feat = sen2_features(all_patches)

                    if each_data.endswith('osm_features'):  # process the osm data
                        for each_patch in all_patches:
                            osm_features = pd.read_csv(each_patch, header=None)  # read the osm feature csv file
                            osm_features = osm_features.dropna()  # drop the NA fields
                            osm_features = osm_features.T    # take the transpose to (2,56)
                            all_keys = osm_features.iloc[0].tolist()   # get all the keys
                            values = osm_features.iloc[1].tolist()   # get all the corresponding values for the keys
                            values = [0 if x == np.inf else x for x in values]  # remove inf values
                            values = [0 if x == np.nan else x for x in values]  # remove nan values
                            osm_feat.append(values)  # append osm features for each patch

                        df_osm = pd.DataFrame(osm_feat, columns=all_keys)  # data frame for osm features

                id_list, class_list, pop_count, pop_dens, log_pop_dens = get_id_response_var(all_patches, city_df)

                df_rest = pd.DataFrame()  # initialize data frame for a city
                # add all the features to data frame
                df_rest['GRD_ID'] = id_list
                df_rest['CLASS'] = class_list
                df_rest['POP'] = pop_count
                df_rest['POP_DENS'] = pop_dens
                df_rest['LOG_POP_DENS'] = log_pop_dens

                # Features
                df_rest['LCZ_CL'] = lcz_feat
                df_rest['LU_1_A'] = lu_1_feat
                df_rest['LU_2_A'] = lu_2_feat
                df_rest['LU_3_A'] = lu_3_feat
                df_rest['LU_4_A'] = lu_4_feat
                df_rest['VIIRS_MEAN'] = viirs_mean_feat
                df_rest['VIIRS_MAX'] = viirs_max_feat

                df_rest['SEN2_AUT_MEAN_R'] = sen2_aut_mean_r_feat
                df_rest['SEN2_AUT_MEAN_G'] = sen2_aut_mean_g_feat
                df_rest['SEN2_AUT_MEAN_B'] = sen2_aut_mean_b_feat
                df_rest['SEN2_AUT_MED_R'] = sen2_aut_med_r_feat
                df_rest['SEN2_AUT_MED_G'] = sen2_aut_med_g_feat
                df_rest['SEN2_AUT_MED_B'] = sen2_aut_med_b_feat
                df_rest['SEN2_AUT_STD_R'] = sen2_aut_std_r_feat
                df_rest['SEN2_AUT_STD_G'] = sen2_aut_std_g_feat
                df_rest['SEN2_AUT_STD_B'] = sen2_aut_std_b_feat
                df_rest['SEN2_AUT_MAX_R'] = sen2_aut_max_r_feat
                df_rest['SEN2_AUT_MAX_G'] = sen2_aut_max_g_feat
                df_rest['SEN2_AUT_MAX_B'] = sen2_aut_max_b_feat
                df_rest['SEN2_AUT_MIN_R'] = sen2_aut_min_r_feat
                df_rest['SEN2_AUT_MIN_G'] = sen2_aut_min_g_feat
                df_rest['SEN2_AUT_MIN_B'] = sen2_aut_min_b_feat

                df_rest['SEN2_SPR_MEAN_R'] = sen2_spr_mean_r_feat
                df_rest['SEN2_SPR_MEAN_G'] = sen2_spr_mean_g_feat
                df_rest['SEN2_SPR_MEAN_B'] = sen2_spr_mean_b_feat
                df_rest['SEN2_SPR_MED_R'] = sen2_spr_med_r_feat
                df_rest['SEN2_SPR_MED_G'] = sen2_spr_med_g_feat
                df_rest['SEN2_SPR_MED_B'] = sen2_spr_med_b_feat
                df_rest['SEN2_SPR_STD_R'] = sen2_spr_std_r_feat
                df_rest['SEN2_SPR_STD_G'] = sen2_spr_std_g_feat
                df_rest['SEN2_SPR_STD_B'] = sen2_spr_std_b_feat
                df_rest['SEN2_SPR_MAX_R'] = sen2_spr_max_r_feat
                df_rest['SEN2_SPR_MAX_G'] = sen2_spr_max_g_feat
                df_rest['SEN2_SPR_MAX_B'] = sen2_spr_max_b_feat
                df_rest['SEN2_SPR_MIN_R'] = sen2_spr_min_r_feat
                df_rest['SEN2_SPR_MIN_G'] = sen2_spr_min_g_feat
                df_rest['SEN2_SPR_MIN_B'] = sen2_spr_min_b_feat

                df_rest['SEN2_SUM_MEAN_R'] = sen2_sum_mean_r_feat
                df_rest['SEN2_SUM_MEAN_G'] = sen2_sum_mean_g_feat
                df_rest['SEN2_SUM_MEAN_B'] = sen2_sum_mean_b_feat
                df_rest['SEN2_SUM_MED_R'] = sen2_sum_med_r_feat
                df_rest['SEN2_SUM_MED_G'] = sen2_sum_med_g_feat
                df_rest['SEN2_SUM_MED_B'] = sen2_sum_med_b_feat
                df_rest['SEN2_SUM_STD_R'] = sen2_sum_std_r_feat
                df_rest['SEN2_SUM_STD_G'] = sen2_sum_std_g_feat
                df_rest['SEN2_SUM_STD_B'] = sen2_sum_std_b_feat
                df_rest['SEN2_SUM_MAX_R'] = sen2_sum_max_r_feat
                df_rest['SEN2_SUM_MAX_G'] = sen2_sum_max_g_feat
                df_rest['SEN2_SUM_MAX_B'] = sen2_sum_max_b_feat
                df_rest['SEN2_SUM_MIN_R'] = sen2_sum_min_r_feat
                df_rest['SEN2_SUM_MIN_G'] = sen2_sum_min_g_feat
                df_rest['SEN2_SUM_MIN_B'] = sen2_sum_min_b_feat

                df_rest['SEN2_WIN_MEAN_R'] = sen2_win_mean_r_feat
                df_rest['SEN2_WIN_MEAN_G'] = sen2_win_mean_g_feat
                df_rest['SEN2_WIN_MEAN_B'] = sen2_win_mean_b_feat
                df_rest['SEN2_WIN_MED_R'] = sen2_win_med_r_feat
                df_rest['SEN2_WIN_MED_G'] = sen2_win_med_g_feat
                df_rest['SEN2_WIN_MED_B'] = sen2_win_med_b_feat
                df_rest['SEN2_WIN_STD_R'] = sen2_win_std_r_feat
                df_rest['SEN2_WIN_STD_G'] = sen2_win_std_g_feat
                df_rest['SEN2_WIN_STD_B'] = sen2_win_std_b_feat
                df_rest['SEN2_WIN_MAX_R'] = sen2_win_max_r_feat
                df_rest['SEN2_WIN_MAX_G'] = sen2_win_max_g_feat
                df_rest['SEN2_WIN_MAX_B'] = sen2_win_max_b_feat
                df_rest['SEN2_WIN_MIN_R'] = sen2_win_min_r_feat
                df_rest['SEN2_WIN_MIN_G'] = sen2_win_min_g_feat
                df_rest['SEN2_WIN_MIN_B'] = sen2_win_min_b_feat

                df = pd.concat([df_rest, df_osm], axis=1)  # appending the rest of features and osm features
                df.to_csv(feature_csv_file, index=False)   # save the features to csv files
                print("City {} finished".format(city_name))
        print('All cities processed for So2Sat POP Part 1 \n')

    else:
        print('Preparing features for So2sat Part2 \n')
        all_folders = glob.glob(os.path.join(all_patches_mixed_path, '*'))
        for each_folder in all_folders:
            all_patches_mixed_path_part1 = each_folder.replace('Part2', 'Part1')
            all_cities = glob.glob(os.path.join(each_folder, '*'))
            for each_city in all_cities:
                # declare lists for input data source
                dem_mean_feat = []
                dem_max_feat = []

                city_name = os.path.split(each_city)[1]  # get the name of the city from the city path
                feature_csv_file = os.path.join(all_patches_mixed_path_part1, city_name + '/' + city_name + '_features.csv')  # create feature csv for city
                all_data = glob.glob(os.path.join(each_city, '*'))  # get all the data folders
                city_csv_file = os.path.join(all_patches_mixed_path_part1, city_name + '/' + city_name + '.csv')  # get the city's csv
                city_df = pd.read_csv(city_csv_file)  # data frame for the city

                for each_data in all_data:  # for each data folder in a city
                    all_patches = []  # list to all patches
                    if each_data.endswith('.csv'):  # skip the csv file, get only data folders
                        # skip the file
                        continue
                    all_classes = glob.glob(os.path.join(each_data, '*'))  # get all class folders in data folder
                    for each_class in all_classes:
                        class_patches = glob.glob(os.path.join(each_class, '*'))
                        for x in class_patches:
                            all_patches.append(x)  # get list of all the city patches

                    if each_data.endswith('dem'):  # process dem data
                        for each_patch in all_patches:
                            dem_mean, dem_max = average_mean_features(each_patch,
                                                                      band=1)  # get mean and max of dem patch
                            dem_mean_feat.append(dem_mean)  # dem mean feature list
                            dem_max_feat.append(dem_max)  # dem max feature list

                id_list, class_list, pop_count, pop_dens, log_pop_dens = get_id_response_var(all_patches, city_df)
                df_rest = pd.DataFrame()  # initialize data frame for a city

                if os.path.isfile(feature_csv_file):
                    df_rest['DEM_MEAN'] = dem_mean_feat
                    df_rest['DEM_MAX'] = dem_max_feat
                    df_part1 = pd.read_csv(feature_csv_file)
                    df = pd.concat([df_part1, df_rest], axis=1)  # appending the rest of features and osm features
                    df.to_csv(feature_csv_file, index=False)  # save the features to csv files
                    print("City {} finished".format(city_name))
                else:
                    # add all the features to data frame
                    df_rest['GRD_ID'] = id_list
                    df_rest['CLASS'] = class_list
                    df_rest['POP'] = pop_count
                    df_rest['POP_DENS'] = pop_dens
                    df_rest['LOG_POP_DENS'] = log_pop_dens
                    df_rest = pd.DataFrame()  # initialize data frame for a city
                    df_rest.to_csv(feature_csv_file, index=False)  # save the features to csv files
                    print("City {} finished".format(city_name))
        print('All cities processed for So2Sat POP Part 2 \n')


