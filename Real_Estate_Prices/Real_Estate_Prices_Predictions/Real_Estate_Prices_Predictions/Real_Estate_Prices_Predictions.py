'''
Author: Esther Edith Spurlock

Project Name: Predicting Real Estate Proces

Project Description: This project comes from Kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
    The data describes residential homes in Ames, Iowa
    The goal is to predict the price of the homes in the area
    The code here is inspired by these solutions: 
        https://www.kaggle.com/akashsdas/predict-house-prices-in-depth-eda
        https://www.kaggle.com/skirmer/fun-with-real-estate-data
'''

#Import statements
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, learning_curve
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

#import other python files I have written
import explore_data
import transform_data

#the name of the column we want to predict
TO_PREDICT = 'SalePrice'


def main():
    '''
    Goes through the machine learning pipeline from data import to final result

    Inputs: none

    Outputs: Model Outcomes
    '''
    #first, we need to load in the training and testing data
    training_df, testing_df = load_data() 
    print("Back in main")

    #now we will explore our data
    explore_data.main(training_df)
    print("Back in main")

    #now we will transform our data
    transform_data.main(training_df)
    print("Back in main")

def load_data():
    '''
    Loads in the training and testing dataset

    Kaggle has already split up the training and testing datasets. In a perfect world,
    I would prefer to have one dataset and randomly split up training and testing myself,
    but the final sales price for the testing data is not included. For a Kaggle project,
    this makes sense, but this also means I won't be able to calculate the accuracy, precision,
    or other performance metrics for this data.

    Inputs: none

    Outputs: the training and testing data sets as pandas dataframes
    '''
    training_df = pd.read_csv('data/train.csv')
    print("Training data loaded")
    testing_df = pd.read_csv('data/test.csv')
    print("Testing data loaded")
    return training_df, testing_df

main()
