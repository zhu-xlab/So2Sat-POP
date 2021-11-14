# Contains all the constants defined in the project
import os

# paths to the current folder
if os.name == "nt":  # locally
    current_dir_path = os.getcwd()

img_rows = 100  # patch height
img_cols = 100  # patch width
osm_features = 56  # number of osm based features

# paths to So2Sat POP Part1 folder
all_patches_mixed_part1 = os.path.join(current_dir_path, 'So2Sat POP Part1')  # path to So2Sat POP Part 1 data folder
all_patches_mixed_train_part1 = os.path.join(all_patches_mixed_part1, 'train')   # path to train folder
all_patches_mixed_test_part1 = os.path.join(all_patches_mixed_part1, 'test')   # path to test folder

# paths to So2Sat POP Part2 folder
all_patches_mixed_part2 = os.path.join(current_dir_path, 'So2Sat POP Part2')  # path to So2Sat POP Part 2 data folder
all_patches_mixed_train_part2 = os.path.join(all_patches_mixed_part2, 'train')   # path to train folder
all_patches_mixed_test_part2 = os.path.join(all_patches_mixed_part2, 'test')   # path to test folder

# covariates used to train the model
covariate_list = ['DEM_MEAN', 'DEM_MAX', 'LCZ_CL', 'LU_1_A', 'LU_2_A', 'LU_3_A', 'LU_4_A', 'VIIRS_MEAN', 'VIIRS_MAX',
                  'SEN2_AUT_MEAN_R', 'SEN2_AUT_MEAN_G', 'SEN2_AUT_MEAN_B', 'SEN2_AUT_MED_R', 'SEN2_AUT_MED_G',
                  'SEN2_AUT_MED_B', 'SEN2_AUT_STD_R', 'SEN2_AUT_STD_G', 'SEN2_AUT_STD_B', 'SEN2_AUT_MAX_R',
                  'SEN2_AUT_MAX_G', 'SEN2_AUT_MAX_B', 'SEN2_AUT_MIN_R', 'SEN2_AUT_MIN_G', 'SEN2_AUT_MIN_B',
                  'SEN2_SPR_MEAN_R', 'SEN2_SPR_MEAN_G', 'SEN2_SPR_MEAN_B', 'SEN2_SPR_MED_R', 'SEN2_SPR_MED_G',
                  'SEN2_SPR_MED_B', 'SEN2_SPR_STD_R', 'SEN2_SPR_STD_G', 'SEN2_SPR_STD_B', 'SEN2_SPR_MAX_R',
                  'SEN2_SPR_MAX_G', 'SEN2_SPR_MAX_B', 'SEN2_SPR_MIN_R', 'SEN2_SPR_MIN_G', 'SEN2_SPR_MIN_B',
                  'SEN2_SUM_MEAN_R', 'SEN2_SUM_MEAN_G', 'SEN2_SUM_MEAN_B', 'SEN2_SUM_MED_R', 'SEN2_SUM_MED_G',
                  'SEN2_SUM_MED_B', 'SEN2_SUM_STD_R', 'SEN2_SUM_STD_G', 'SEN2_SUM_STD_B', 'SEN2_SUM_MAX_R',
                  'SEN2_SUM_MAX_G', 'SEN2_SUM_MAX_B', 'SEN2_SUM_MIN_R', 'SEN2_SUM_MIN_G', 'SEN2_SUM_MIN_B',
                  'SEN2_WIN_MEAN_R', 'SEN2_WIN_MEAN_G', 'SEN2_WIN_MEAN_B', 'SEN2_WIN_MED_R', 'SEN2_WIN_MED_G',
                  'SEN2_WIN_MED_B', 'SEN2_WIN_STD_R', 'SEN2_WIN_STD_G', 'SEN2_WIN_STD_B', 'SEN2_WIN_MAX_R',
                  'SEN2_WIN_MAX_G', 'SEN2_WIN_MAX_B', 'SEN2_WIN_MIN_R', 'SEN2_WIN_MIN_G', 'SEN2_WIN_MIN_B', 'aerialway',
                  'aeroway', 'amenity', 'barrier', 'boundary', 'building', 'craft', 'emergency', 'geological',
                  'healthcare', 'highway', 'historic', 'landuse', 'leisure', 'man_made', 'military', 'natural',
                  'office', 'place', 'power', 'public Transport', 'railway', 'route', 'shop', 'sport', 'telecom',
                  'tourism', 'water', 'waterway', 'addr:housenumber', 'restrictions', 'other', 'n', 'm', 'k_avg',
                  'intersection_count', 'streets_per_node_avg', 'streets_per_node_counts_argmin',
                  'streets_per_node_counts_min', 'streets_per_node_counts_argmax', 'streets_per_node_counts_max',
                  'streets_per_node_proportion_argmin', 'streets_per_node_proportion_min',
                  'streets_per_node_proportion_argmax', 'streets_per_node_proportion_max', 'edge_length_total',
                  'edge_length_avg', 'street_length_total', 'street_length_avg', 'street_segments_count',
                  'node_density_km', 'intersection_density_km', 'edge_density_km', 'street_density_km', 'circuity_avg',
                  'self_loop_proportion']


min_fimportance = 0.002

# Parameter for the Grid Search for hyperparameter optimization
param_grid = {'oob_score': [True], 'bootstrap': [True],
              'max_features': ['sqrt', 0.05, 0.1, 0.2, 0.3, 0.4],
              'n_estimators': [250, 350, 500, 750, 1000]}

# Kfold parameter
kfold = 10
n_jobs = -1
