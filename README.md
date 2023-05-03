# NYC Taxi Fare Prediction

* **One Sentence Summary** This repository holds an attempt to apply Machine Learning (ML) models in an attempt to predict the taxi fare price in New York City when given the pickup and dropoff locations.

## Overview

* Challenge Link: https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction
  * **Definition of the tasks / challenge**  The task at hand is to predict the fare amount for a taxi ride in New York City, with machine learning techniques from the Keras models.
  * **Your approach** The approach in this repository formulates the problem as a deep learning training model. Using the Sequential/Dense models from the Keras model, we ran 25 epochs on the data set.
  * **Summary of the performance achieved** Our best model was able to predict the taxi fare price within 5%, 90% of the time. At the current time of writing, the best performance on Kaggle of this metric is 4.08%. 

## Summary of Workdone

### Data

* Data:
  * Type: float64
    * Input: CSV file consisting of 8 columns which contained a unique time key, the fare price amount and 6 features to go along with it
  * Size: 5.7 GB which was then reduced to 7% -> 400 MB
  * Instances (Train, Test, Validation Split):
    *   Train: 55,000,000 rows but reduced to 4,000,000
    *   Test: 10,000 rows
    *   No Validation Split

#### Preprocessing / Clean up

* Due to data consisting of 55,000,000 rows and a size of 5.7 GBs, it was cut down to only 4,000,000 rows which had a final size of 400 MB.
* A small amount of null entries existed so they were dropped from the set
* There exist fare amounts that are in the negatives and high up into the $1000s, so the range of the fare amount is limited to $2.00 and $600.
* Passenger count went up to 208, so it was reduced to a reasonal maximum capacity of a typical taxi which is 5 passengers.
* pickup_datetime was transformed into a numerical type and split into multiple attributes consisting of Year, Month, Day, Weekday and Hour.
* Longitude and Latitude values that were outside of the bounds of (-75,-72) and (40,42) were removed as it does not lie in the NYC area.

#### Data Visualization

![Untitled](https://user-images.githubusercontent.com/94269160/235988450-d1e8186f-2d3b-4fd3-abc0-8108bdd3ac9c.png)
![Untitled1](https://user-images.githubusercontent.com/94269160/235988448-469338ca-a7f9-40d4-9850-630a4d815098.png)
![Untitled2](https://user-images.githubusercontent.com/94269160/235988446-60ad987e-b316-4d56-b639-86d2bfda0cc0.png)
![Untitled3](https://user-images.githubusercontent.com/94269160/235988442-98595c3f-53b1-404c-a19e-cbd515e11528.png)
![Untitled4](https://user-images.githubusercontent.com/94269160/235988434-77ad5ad9-6070-4b59-9fed-0dad2533613a.png)
![Hist](https://user-images.githubusercontent.com/94269160/235988458-2e681445-20ae-4c5a-8a99-383dfbf2cdf0.png)
![LatLongDiff](https://user-images.githubusercontent.com/94269160/235988456-870ad70b-35c2-498c-97f9-ef714ee4c988.png)
![FarePerPass](https://user-images.githubusercontent.com/94269160/235988453-efc9e963-9f41-488a-a12e-4966133c14b9.png)

### Problem Formulation

* Define:
  * Input / Output
  * Models:
    * Used Linear Regression
    

### Training

* Describe the training:
  * How you trained: software and hardware.
  * How did training take.
  * Training curves (loss vs epoch for test/train).
  * How did you decide to stop training.
  * Any difficulties? How did you resolve them?

### Performance Comparison

* Clearly define the key performance metric(s).
* Show/compare results in one table.
* Show one (or few) visualization(s) of results, for example ROC curves.

### Conclusions

* State any conclusions you can infer from your work. Example: LSTM work better than GRU.

### Future Work

* What would be the next thing that you would try.
* What are some other studies that can be done starting from here.

## How to reproduce results

* In this section, provide instructions at least one of the following:
   * Reproduce your results fully, including training.
   * Apply this package to other data. For example, how to use the model you trained.
   * Use this package to perform their own study.
* Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.

### Overview of files in repository

* Describe the directory structure, if any.
* List all relavent files and describe their role in the package.
* An example:
  * utils.py: various functions that are used in cleaning and visualizing data.
  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * models.py: Contains functions that build the various models.
  * training-model-1.ipynb: Trains the first model and saves model during training.
  * training-model-2.ipynb: Trains the second model and saves model during training.
  * training-model-3.ipynb: Trains the third model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.

* Note that all of these notebooks should contain enough text for someone to understand what is happening.

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

### Data

* Point to where they can download the data.
* Lead them through preprocessing steps, if necessary.

### Training

* Describe how to train the model

#### Performance Evaluation

* Describe how to run the performance evaluation.


## Citations

* https://www.kaggle.com/code/dster/nyc-taxi-fare-starter-kernel-simple-linear-model
