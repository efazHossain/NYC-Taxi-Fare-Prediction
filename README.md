# NYC Taxi Fare Prediction
![taxi-fare-1561717](https://user-images.githubusercontent.com/94269160/235989642-d8654982-b3ea-4df2-ba8f-786d8b1e1415.jpg)

* **One Sentence Summary** This repository holds an attempt to apply Machine Learning (ML) models in an attempt to predict the taxi fare price in New York City when given the pickup and dropoff locations.

### Overview
* Challenge Link: https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction
  * **Definition of the tasks / challenge**:  The task at hand is to predict the fare amount for a taxi ride in New York City, with machine learning techniques using a Linear Regression problem. The challenge evaluates the RMSE (Root-Mean-Squared-Error) between the prediction model and what is the ground truth. A good RMSE score that is close to ~0 but not higher than 5 can help tell you how incorrect the model might be on unseen data.
  * **Summary of the performance achieved**: Our best model was able to predict the taxi fare price within 8%, 90% of the time. At the current time of writing, the best performance on Kaggle of this metric is 4.08%. 

### Data
* Data:
  * Type: float64
    * Input: CSV file consisting of 8 columns which contained a unique time key, the fare price amount and 6 features to go along with it
  * Size: 5.7 GB which was then reduced to 7% -> 400 MB
  * Instances (Train, Test, Validation Split):
    *  Train: 55,000,000 passengers initially but reduced to 4,000,000 passengers
    *  Test: 10,000 passengers
    *  Validation Split of 100,000 passengers
  * Features:
    * pickup_datetime - timestamp value indicating when the taxi ride started.
    * pickup_longitude - float for longitude coordinate of where the taxi ride started.
    * pickup_latitude - float for latitude coordinate of where the taxi ride started.
    * dropoff_longitude - float for longitude coordinate of where the taxi ride ended.
    * dropoff_latitude - float for latitude coordinate of where the taxi ride ended.
    * passenger_count - integer indicating the number of passengers in the taxi ride.

#### Preprocessing / Clean up

* Due to data consisting of 55,000,000 rows but it was cut down to only 4,000,000 rows being read in.
* A small amount of null entries existed so they were dropped from the set.
* There exist fare amounts that are in the negatives and high up into the $1000s, so the range of the fare amount is limited to $2.00 and $600.
* Passenger count went up to 208, so it was reduced to a reasonal maximum capacity of a typical taxi which is 5 passengers.
* Longitude and Latitude values that were outside of the bounds of (-75,-72) and (40,42) were removed as it does not lie in the NYC area.

#### Data Visualization
* pickup_longitude
![Untitled](https://user-images.githubusercontent.com/94269160/235988450-d1e8186f-2d3b-4fd3-abc0-8108bdd3ac9c.png)
* pickup_latitude
![Untitled1](https://user-images.githubusercontent.com/94269160/235988448-469338ca-a7f9-40d4-9850-630a4d815098.png)
* dropoff_longitude
![Untitled2](https://user-images.githubusercontent.com/94269160/235988446-60ad987e-b316-4d56-b639-86d2bfda0cc0.png)
* dropoff_latitude
![Untitled3](https://user-images.githubusercontent.com/94269160/235988442-98595c3f-53b1-404c-a19e-cbd515e11528.png)
* passenger_count
![Untitled4](https://user-images.githubusercontent.com/94269160/235988434-77ad5ad9-6070-4b59-9fed-0dad2533613a.png)
* Histogram of Fare Prices
![Hist](https://user-images.githubusercontent.com/94269160/235988458-2e681445-20ae-4c5a-8a99-383dfbf2cdf0.png)
* Average Difference of Latitude vs Longitude
![LatLongDiff](https://user-images.githubusercontent.com/94269160/235988456-870ad70b-35c2-498c-97f9-ef714ee4c988.png)
* Fare Price Per Passenger (up to 5)
![FarePerPass](https://user-images.githubusercontent.com/94269160/235988453-efc9e963-9f41-488a-a12e-4966133c14b9.png)

### Problem Formulation
* Define: Ideally, the model should reflect the data that is contained in the sample_submission.csv file or at least close to it. And to do so, we need to train a model or models in this case.
  * Input / Output: 
      * Input: train.csv, test.csv
      * Output: submission.csv - a 2x2 .csv file that contains the key (date-time) and fare_amount in USD.
  * Models:
    * Used Linear Regression -> predict the value of the taxi fare based upon past data
    * Used GradientBoostingRegressor -> calculate the difference between the trained model and past models in order to train a weak model to fit the difference, or in better words, a residual.

### Training
* Describe the training:
  * Using LinearRegression
        * Model was trained under two sets, a training and a validation set. Both sets were used in an attempt to replicate the fare price of the sample_submission.csv file by using the RMSE (Root-Mean-Squared-Error) score to see if there is any skew between the data and the model.
  * Using GradientBoostingRegressor
        * Model was trained with a regression fit to predict the fare price by using the data and comparing it with a weaker regression model to boost itself into predicting the "true" value by using the difference between each prediction.

### Conclusions
* Overall, the fare cost for a taxi in New York City should be around $11.35 dollars as in the sample-submission.csv file.
* In my model, the fare cost would average at around $11.47 with a RMSE score of 4.08, which ideally is not a bad score but, a value closer to ~0 would be a much better showcase of the model training.

### Future Work
* In the future, I plan to use more better models to get the RMSE score to at least around 1-3 as a lower RMSE score shows how well a model is accurately predicting the trend from a given dataset.
* Models from Keras would especially help accomplish more as of currently, I do not have much experience with it.

## How to reproduce results
* First start with downloading the files from the link
* Then, import the libraries shown below
* Next, create a cell and extract the .zip file into your repository, it will slowly extract all 5.7 GBs of data into a "train.csv" file
* Be sure to also make a list of the column names that can be used to plot the features and split data sets.
* Look at the head of the dataframe and general statistics of the df by using df.head() and df.describe()
* Afterwards, plot each features via for-loop
* Histogram the fare_amount column to see the spread of the fare prices ranging from $2 to $80.
* Drop missing values and start splitting data into training and validation sets.
* Using the LinearRegression model from sci-kit-learn, we can fit a model using the training sets and find a RMSE (Root-Mean-Squared-Error) score.
* Another model that was used is the GradientBoostingRegressor which uses a loss function to start predicting the trend for the data.
* Finally, make an array of the predicted values and turn it into a .csv file after categorizing it to only contain the key (date and time) and fare_amount from the original dataset.

### Overview of files in repository
* Files in Repository:
  *  EfazH_NYC-Taxi-Fare.ipynb: notebook using LinearRegression
  *  EfazH_NYC_Taxi_V2.ipynb: notebook using GradientBoostingRegressor  
  *  efazh_kaggle.csv: my submission.csv file showing the key (date-time) and fare_amount  
  *  efazh_kaggle_v2.csv: second submission .csv file showing only the fare_amount, averaging at around $11.38 which is close to the sample_submission file
  * sample_submission.csv: a sample submission file in the correct format (columns key and fare_amount). This file 'predicts' fare_amount to be $11.35 for all rows, which is the mean fare_amount from the training set.
  
### Software Setup
* Python 3.7 or higher
* Required Packages:
  * Pandas: view .csv as a dataframe and analyze it
  * Numpy: set up an array for predicted values
  * Matplotlib.pyplot: plot each feature and the main feature itself
  * Sklearn: train_test_split, mean_squared_error and LinearRegression
  * ZipFile: extract the data from the .zip file in Kaggle

## Citations
* https://www.kaggle.com/code/dster/nyc-taxi-fare-starter-kernel-simple-linear-model
* https://kaggle.com/competitions/new-york-city-taxi-fare-prediction
