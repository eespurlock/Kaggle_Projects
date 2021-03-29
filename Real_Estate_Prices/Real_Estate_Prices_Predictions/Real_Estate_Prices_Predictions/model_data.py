'''
This file puts the data we have created into various machine learning models
and analyzes their precision and accuracy to determine which models would
be best for determining real estate prices
'''

#imports for pandas, numpy, and mysql
import pandas as pd
import numpy as np
import mysql.connector

#imports for sklearn models
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,\
    GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split

#imports for sklearn metrics
from sklearn.metrics import mean_squared_error, r2_score

#import the garbage collector
import gc

#Defined constants for the models we will use
LOG_REGRESSION = "Logistic Regression"
LIN_REGRESSION = "Linear Regression"
RIDGE = "Ridge"
KNN = "K Nearest Neighbors"
TREE = "Decision Trees"
FOREST = "Random Forests"
EXTRA = "Extra Trees"
ADA_BOOSTING = "Ada Boosting"
GRAD_BOOSTING = "Gradient Boosting"
BAGGING = "Bagging"

#Defined constants for the dictionaries
MODEL = "Model"
IN_NAME_ONE = "Input Name One"
IN_LST_ONE = "Input List One"
IN_NAME_TWO = "Input Name Two"
IN_LST_TWO = "Input List Two"
INPUTS_NAME = "Inputs Name"
DUMMY_NAME = "None"

#Lists of the various inputs I will use in my models
C_VALS = [1.0, 1.2, 1.5, 2.0, 2.5]
NEIGHBORS = [1, 5, 10, 20, 50, 100]
MAX_DEPTH = [1, 5, 20, 50, 100, 200]
THOUSANDS = [1000, 3000, 5000]
N_ESTIMATORS = [5, 10, 30, 50, 100, 200]
LEARNING_RATE = [0.5, 1.0, 2.0]
MAX_SAMPLES = [10, 50, 100, 500]
N_JOBS = [1, -1]
ALPHA = [0.5, 1.0, 1.5, 2.0]
DUMMY = ["None"]

#This is for easier readability
#Describes the individual models and their inputs
LOG_REG_DICT = {MODEL: LogisticRegression,
                IN_NAME_ONE: 'C',
                IN_LST_ONE: C_VALS,
                IN_NAME_TWO: DUMMY_NAME,
                IN_LST_TWO: DUMMY}
LIN_REG_DICT = {MODEL: LinearRegression,
                IN_NAME_ONE: 'n_jobs',
                IN_LST_ONE: N_JOBS,
                IN_NAME_TWO: DUMMY_NAME,
                IN_LST_TWO: DUMMY}
KNN_DICT = {MODEL: KNeighborsClassifier,
            IN_NAME_ONE: 'n_neighbors',
            IN_LST_ONE: NEIGHBORS,
            IN_NAME_TWO: DUMMY_NAME,
            IN_LST_TWO: DUMMY}
TREE_DICT = {MODEL: DecisionTreeClassifier,
             IN_NAME_ONE: 'max_depth',
             IN_LST_ONE: MAX_DEPTH,
             IN_NAME_TWO: DUMMY_NAME,
             IN_LST_TWO: DUMMY}
FOREST_DICT = {MODEL: RandomForestClassifier,
               IN_NAME_ONE: 'n_estimators',
               IN_LST_ONE: THOUSANDS,
               IN_NAME_TWO: 'max_depth',
               IN_LST_TWO: MAX_DEPTH}
EXTRA_DICT = {MODEL: ExtraTreesClassifier,
               IN_NAME_ONE: 'n_estimators',
               IN_LST_ONE: THOUSANDS,
               IN_NAME_TWO: 'max_depth',
               IN_LST_TWO: MAX_DEPTH}
ADA_DICT = {MODEL: AdaBoostClassifier,
            IN_NAME_ONE: 'n_estimators',
            IN_LST_ONE: N_ESTIMATORS,
            IN_NAME_TWO: 'learning_rate',
            IN_LST_TWO: LEARNING_RATE}
BAG_DICT = {MODEL: BaggingClassifier,
            IN_NAME_ONE: 'n_estimators',
            IN_LST_ONE: N_ESTIMATORS,
            IN_NAME_TWO: 'max_samples',
            IN_LST_TWO: MAX_SAMPLES}
RIDGE_DICT = {MODEL: Ridge,
              IN_NAME_ONE: 'alpha',
              IN_LST_ONE: ALPHA,
              IN_NAME_TWO: 'max_iter',
              IN_LST_TWO: THOUSANDS}
GRAD_DICT = {MODEL: GradientBoostingClassifier,
             IN_NAME_ONE: 'learning_rate',
             IN_LST_ONE: LEARNING_RATE,
             IN_NAME_TWO: 'n_estimators',
             IN_LST_TWO: N_ESTIMATORS}

#dictionaries describing the different models we will run and their inputs
#models where we specify only one input
#MODELS_DICT = {LOG_REGRESSION: LOG_REG_DICT,
#                         KNN: KNN_DICT,
#                         TREE: TREE_DICT,
#                         LIN_REGRESSION: LIN_REG_DICT,
#                         FOREST: FOREST_DICT,
#                         EXTRA: EXTRA_DICT,
#                         ADA_BOOSTING: ADA_DICT,
#                         BAGGING: BAG_DICT,
#                         RIDGE: RIDGE_DICT}

#our modeling timed out at a certain point and I am just
#going to run the models it did not get through
MODELS_DICT = {ADA_BOOSTING: ADA_DICT,
                BAGGING: BAG_DICT,
                RIDGE: RIDGE_DICT}

def main(variable_df, feature_df):
    '''
    The main of our file. Goes through the process of:
        1: splitting the data into training and testing sets
        2: running the training data through machine learning models
        3: testing the models

    Inputs: variable_df: a dataframe with all of our data variables
        feature_df: a dataframe with our feature data

    Does not output anything, instead, inputs the data into our SQL database
    '''
    #first, we split our dataframes into testing and training data
    feature_train, feature_test, variable_train, variable_test = \
        train_test_split(variable_df, feature_df)
    print("Now we run our models")
    run_models(MODELS_DICT, variable_train, variable_test, feature_train, \
        feature_test)

def run_models(models_dict, variable_train, variable_test, feature_train, \
    feature_test):
    '''
    Goes through the different models we are running

    Inputs:
        models_dict: the dictionary describing the models we will run
        variable_train: variable for the training set
        variable_test: variable for the testing set
        feature_train: features for the training set
        feature_test: features for the testing set

    Does not output anything, instead, inputs the data into our SQL database
    '''
    #we loop through the dictionary
    for model_name, this_model_dict in models_dict.items():
        print(model_name)
        #we pull our model and inputs out from the dictionary
        model = this_model_dict[MODEL]
        in_name_one = this_model_dict[IN_NAME_ONE]
        in_lst_one = this_model_dict[IN_LST_ONE]
        in_name_two = this_model_dict[IN_NAME_TWO]
        in_lst_two = this_model_dict[IN_LST_TWO]
        #we loop through our inputs
        for input_one in in_lst_one:
            for input_two in in_lst_two:
                #we create the input for a single model
                if in_name_two == DUMMY_NAME:
                    in_dict = {in_name_one: input_one}
                    in_label = in_name_one + ": " + str(input_one)
                else:
                    in_dict = {in_name_one: input_one,
                               in_name_two: input_two}
                    in_label = in_name_one + ": " + str(input_one) + " & " + \
                        in_name_two + ": " + str(input_two)
                #a dictionary that holds the details of our different models
                details_dict = {MODEL: model_name,
                                INPUTS_NAME: in_label}
                #we create the unfitted model
                model_unfit = model(**in_dict)
                #now we test our model
                test_models(model_unfit, details_dict, variable_train, \
                    variable_test, feature_train, feature_test)

def test_models(model_unfit, details_dict, variable_train, variable_test, \
    feature_train, feature_test):
    '''
    Fits the data to our model and tests to see how effective our models are

    Inputs:
        model_unfit: the model we have run, that is not fitted to our data
        details_dict: the details of the model we have run
        variable_train: variable for the training set
        variable_test: variable for the testing set
        feature_train: features for the training set
        feature_test: features for the testing set

    Does not output anything, instead, inputs the data into our SQL database
    '''
    #get the information out of the details dictionary
    model_name = details_dict[MODEL]
    inputs_name = details_dict[INPUTS_NAME]

    #First we fit the model to the training data
    model = model_unfit.fit(feature_train, variable_train)

    #Next, we predict the housing prices
    predicted = model.predict(feature_test)
    
    #now we evaluate the model
    r2 = r2_score(variable_test, predicted)
    add_to_all_models(model_name, inputs_name, "R2 Score", r2)
    mean_error = mean_squared_error(variable_test, predicted, squared=False)
    add_to_all_models(model_name, inputs_name, "Mean Squared Error", \
        mean_error)

    #collect the garbage
    gc.collect()

def add_to_all_models(model_name, inputs, eval_name, result):
    '''
    Inputs data into MySQL

    Inputs:
        model_name: the name of our model
        inputs: the inputs we used to create the model
        eval_name: the name of our evaluation metric
        result: the result of our evaluation
    
    Does not output anything, instead, inputs the data into our SQL database
    '''
    #put the data into a mysql database
    mydb = mysql.connector.connect(
        host = 'localhost',
        user='root',
        password='mysqlrootpwd',
        database='real_estate_model_analysis')
    mycursor = mydb.cursor()
    sql = """INSERT INTO real_estate_model_analysis.model_evaluations 
            (model_name, model_inputs, evaluation_name, result) VALUES (
            %s, %s, %s, %s)"""
    val = (model_name, inputs, eval_name, float(result))
    mycursor.execute(sql, val)
    mydb.commit()
