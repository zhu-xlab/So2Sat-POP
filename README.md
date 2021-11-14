# So2Sat-POP

## Visualization of the sample patches
![Upto_class13_patch_example](https://user-images.githubusercontent.com/61827990/140515202-172ea74d-fea0-42bf-833d-2ba6300b9f11.PNG)

Sample patches from the odd numbered classes of our dataset. Lower classes depicts sparsely populated areas while higher classes depicts densely populated areas.

## Data Download
### Technical University of Munich:
First version:
```
This is the first version of the So2Sat POP dataset covering 106 EU cities. 
The data set has two parts. Each part can be downloaded using the following links:
So2Sat POP Part1 DOI:
So2Sat POP Part2 DOI:
Data set provides the predefined train/test split.
Randomly selected: 75% as train (80 cities) / 25% as test (26 cities)
```

## Institute
[Signal Processing in Earth Observation](https://www.asg.ed.tum.de/sipeo/home/) , Technical University of Munich, and Remote Sensing Technology Institute, German Aerospace Center.

## Funding
This research was funded by the European Research Council (ERC) under the European Unions Horizon 2020 research and innovation program with the grant number ERC-2016-StG-714087 (Acronym:  So2Sat, project website:  www.so2sat.eu), Helmholtz Association under the framework of the Helmholtz Artificial Intelligence Cooperation Unit–Local Unit “Munich Unit @Aeronautics, Space and Transport (MASTr),” and Helmholtz Excellent Professorship “Data Science in Earth Observation – Big Data Fusion for Urban Research and by the German Federal Ministry of Education and Research (BMBF) in the framework of the international future AI lab "AI4EO – Artificial Intelligence for Earth Observation: Reasoning, Uncertainties, Ethics and Beyond" (Grant number: 01DD20001). Additionally, Sugandha Doda is supported by the Helmholtz Association under the joint research school “Munich School for Data Science - MUDS”

## Description of the folders and files
### Folder Structure
![folder_structure_new](https://user-images.githubusercontent.com/61827990/138909117-511e66b9-76bb-4851-a11c-e46b0ff68630.PNG)


### So2Sat POP Part1

train folder: 
```
training data folder contains 80 cities and each city folder has patches from sentinel-2 (sen2_rgb_autumn, sen2_rgb_spring, 
sen2_rgb_summer, sen2_rgb_winter), local climate zone (lcz), viirs nightlights (viirs), land use classifications (lu), 
open source maps (osm), osm based features in Comma Separated Value (CSV) files (osm_features) and corresponding labels 
(population class and population count) in a separate CSV file for each city.
```

test folder: 
```
test data folder contains 26 cities and each city folder has patches from sentinel-2 (sen2_rgb_autumn, sen2_rgb_spring, 
sen2_rgb_summer, sen2_rgb_winter), local climate zone (lcz), viirs nightlights (viirs), land use classifications (lu), 
open source maps (osm), osm based features in Comma Separated Value (CSV) files (osm_features) and corresponding labels 
(population class and population count) in separate CSV file for each city.
```
### So2Sat POP Part2
train folder:
```
training data folder contains 80 cities and each city folder has patches from only digital elevation model (dem). All the dem
patches have been standarized by removing the mean and scaling to unit variance. Please get in touch with us to know the more
details.
```

train folder:
```
test data folder contains 26 cities and each city folder has patches from only digital elevation model (dem). All the dem 
patches have been standarized by removing the mean and scaling to unit variance. Please get in touch with us to know the more 
details.
```

Pixel size for tif file is: 10m by 10m

### Dependencies

Create a conda environment with python 3.8

Packages:
```
joblib
matplotlib
opencv-python
pandas
scikit-learn
gdal
rasterio
```
In case of package conflicts, a requirements.txt with the specific package versions of the working environment has been added to the repo.


Please note that to install GDAL and rasterio, you may need to download the binary wheels for your system ([GDAL](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal) and [rasterio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio)). Run from the downloads folder.
```
pip install GDAL-3.3.3-cp38-cp38-win_amd64.whl
pip install rasterio-1.2.10-cp38-cp38-win_amd64.whl
```

Download the data and run the following scripts:
```
demo_data_loader.py: Sample code to load all the patches for a data folder and their corresponding labels. 
data_preprocessing.py: For each city folder creates a city_name_features.csv file in So2Sat POP Part1 folder, used for RF training.
rf_regression.py: Random forest regression implementation to predict the population count and and evaluate on test cities.
rf_classification.py: Random forest classification implementation to predict the population class and evaluate on test cities.

```
