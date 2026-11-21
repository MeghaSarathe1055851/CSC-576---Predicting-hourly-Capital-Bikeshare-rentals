# Predicting hourly Capital Bikeshare rentals

Source https://doi.org/10.24432/C5W894. Raw hour.csv is included.

## Run from this folder with Python 3.11

Close generated CSV files in Excel before running.

    python -m pip install -r requirements.txt
    python src/01_prepare_data.py
    python src/02_train_models.py

## Setup

Target is cnt. Inputs are hr, workingday, season, weathersit, mnth,
weekday, holiday, temp, hum, windspeed, yr, and atemp.
Dates are used for sorting and splitting. instant, casual, and registered
are excluded. Training is before August 1, 2012. Testing is August-December
2012. All models use the same 3,632 test hours. 

## Results

| Model | MAE | RMSE | R-squared |
| --- | ---: | ---: | ---: |
| Training mean | 175.73 | 233.55 | -0.121 |
| Linear regression | 99.26 | 134.07 | 0.631 |
| Random forest | 65.86 | 95.26 | 0.814 |

MAE and RMSE use rentals per hour. Results and figures are saved under
results/ and figures/. test_predictions.csv pairs actual and predicted cnt.
Weather inputs are observed conditions.