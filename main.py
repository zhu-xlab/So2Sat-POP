from rf_classification import rf_classifier
from rf_regression import rf_regressor
from data_preprocessing import feature_engineering
from constants import all_patches_mixed_part1, all_patches_mixed_part2

if __name__ == '__main__':

    # create features for training and testing data from So2Sat POP Part1 and So2Sat POP Part2
    feature_engineering(all_patches_mixed_part1)
    feature_engineering(all_patches_mixed_part2)

    # Perform classification, ground truth is population class
    rf_classifier(file_name='rf_cls', ground_truth_col='CLASS')

    # Perform regression, ground truth could be population count (POP) or population density (POP_DENS) or log of
    # population density (LOG_POP_DENS)
    rf_regressor(file_name='rf_reg', ground_truth_col='POP')



