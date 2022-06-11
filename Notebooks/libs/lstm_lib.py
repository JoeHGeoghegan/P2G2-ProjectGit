# Relataly Library

# LSTM Model modified from https://www.relataly.com/stock-market-prediction-using-multivariate-time-series-in-python/1815/

import math # Mathematical functions 
import numpy as np # Fundamental package for scientific computing with Python
import pandas as pd # Additional functions for analysing and manipulating data
from datetime import date, timedelta, datetime # Date Functions
from pandas.plotting import register_matplotlib_converters # This function adds plotting functions for calender dates
import matplotlib.pyplot as plt # Important package for visualization - we use this to plot the market data
import matplotlib.dates as mdates # Formatting dates
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error # Packages for measuring model performance / errors
from tensorflow.keras import Sequential # Deep learning library, used for neural networks
from tensorflow.keras.layers import LSTM, Dense, Dropout # Deep learning classes for recurrent and regular densely-connected layers
from tensorflow.keras.callbacks import EarlyStopping # EarlyStopping during model training
from sklearn.preprocessing import RobustScaler, MinMaxScaler # This Scaler removes the median and scales the data according to the quantile range to normalize the price data 
import seaborn as sns # Visualization
sns.set_style('white', { 'axes.spines.right': False, 'axes.spines.top': False})

# Sort Dataframe (LK code)
def load_sort_df(filepath):
    df = pd.read_csv(f'{filepath}', parse_dates = True, infer_datetime_format = True)
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], infer_datetime_format = True, errors = 'coerce', format = '%Y/%m/%d')
    df = df.set_index('Unnamed: 0')
    df.index.name = None
    df = df.reset_index().rename({'index': 'date'}, axis=1)
    train_df = df.sort_values(by=['date']).copy()
    train_df = train_df.set_index('date')
    return train_df

# List of considered Features
features_with_sec = ['revenue',
            'cost_without_depletion_and_amortization',
            'operations_and_support_expense',
            'selling_and_market_expense',
            'research_and_development_expense',
            'general_and_administrative_expense',
            'depreciation_depletion_and_amortization',
            'costs_and_expenses',
            'operating_income_loss',
            'interest_expense',
            'nonoperating_income_expense',
            'incomeloss_from_continuining_operations',
            'income_tax_expense_benefit',
            'income_loss_from_equity_method_investments',
            'profit_loss',
            #'unnamed',
            'net_income_loss',
            'earnings_per_share_basic',
            'earnings_per_share_diluted',
            'weighted_average_number_of_shares_outstanding_basic',
            'weight_average_number_of_diluted_shares_outstanding',
            'net_income_loss_attributable_to_redeemable_noncontrolling_interest',
            #'sec_compound_sentiment',
            'sec_positive_sentiment',
            'sec_neutral sentiment',
            'sec_negative_sentiment',
            'stockmarket_compound_sentiment',
            'stockmarket_positive_sentiment',
            'stockmarket_neutral_sentiment',
            'stockmarket_negative_sentiment',
            'volume',
            'close']

def prepare_lstm(train_df, features):
    # Create the dataset with features and filter the data to the list of FEATURES
    data = pd.DataFrame(train_df)
    data_filtered = data[features]

    # We add a prediction column and set dummy values to prepare the data for scaling
    data_filtered_ext = data_filtered.copy()
    data_filtered_ext['Prediction'] = data_filtered_ext['close']

    # Print the tail of the dataframe
    data_filtered_ext.tail()

    # Scaling
    # Convert the data to numpy values
    np_data_unscaled = np.array(data_filtered)

    # Transform the data by scaling each feature to a range between 0 and 1
    scaler = MinMaxScaler()
    np_data_scaled = scaler.fit_transform(np_data_unscaled)

    # Set the sequence length - this is the timeframe used to make a single prediction
    sequence_length = "50"

    # Prediction Index
    index_Close = data_filtered.columns.get_loc("close")

    # Split the training data into train and train data sets
    # As a first step, we get the number of rows to train the model on 80% of the data 
    train_data_len = math.ceil(np_data_scaled.shape[0] * 0.8)

    # Create the training and test data
    train_data = np_data_scaled[0:train_data_len, :]
    test_data = np_data_scaled[train_data_len - sequence_length:, :]

    # The RNN needs data with the format of [samples, time steps, features]
    # Here, we create N samples, sequence_length time steps per sample, and 6 features
    def partition_dataset(sequence_length, data):
        x, y = [], []
        data_len = data.shape[0]
        for i in range(sequence_length, data_len):
            x.append(data[i-sequence_length:i,:]) #contains sequence_length values 0-sequence_length * columns
            y.append(data[i, index_Close]) #contains the prediction values for validation,  for single-step prediction
    
        # Convert the x and y to numpy arrays
        x = np.array(x)
        y = np.array(y)
        return x, y

    # Generate training data and test data
    x_train, y_train = partition_dataset(sequence_length, train_data)
    x_test, y_test = partition_dataset(sequence_length, test_data)

    return x_train, y_train, x_test, y_test

