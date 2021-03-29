'''
Author: Esther Edith Spurlock

Project Name: Predicting Real Estate Proces

Project Description: This project comes from Kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
    The data describes residential homes in Ames, Iowa
    The goal is to predict the price of the homes in the area
    The code here is inspired by these solutions: 
        https://www.kaggle.com/akashsdas/predict-house-prices-in-depth-eda
        https://www.kaggle.com/skirmer/fun-with-real-estate-data
    Also, I used a machine learning pipeline I previous created as reference:
        https://github.com/eespurlock/CAPP_30254/tree/master/assignment3
        https://github.com/eespurlock/CAPP_30254/tree/master/assignment5
'''

#Import statements
import numpy as np
import pandas as pd

#import the garbage collector
import gc

#import other python files I have written
import explore_data
import transform_data
import model_data

#the name of the column we want to predict
TO_PREDICT = 'SalePrice'


def main():
    '''
    Goes through the machine learning pipeline from data import to final result

    Inputs: none

    Outputs: Model Outcomes
    '''
    #first, we need to load in the training and testing data
    df = load_data() 
    print("Back in main")

    #now we will explore our data
    #explore_data.main(df)
    #print("Back in main")

    #now we will transform our data
    variable_df, feature_df = transform_data.main(df)
    print("Back in main")

    model_data.main(variable_df, feature_df)
    gc.collect()

def load_data():
    '''
    Loads in the training and testing dataset

    Kaggle has already split up the training and testing datasets. However,
    because I would like to calculate the precision and accuracy of my models,
    and I am not going to turn this project back into Kaggle for evaluation,
    I am only going to use the training data and ignore the testing data.

    Inputs: none

    Outputs: the training and testing data sets as pandas dataframes
    '''
    df = pd.read_csv('data/train.csv')
    return df

main()
