This project creates a machine learning pipeline to predict real estate proces based on features of
a given property.

This is a project originally from the Kaggle site: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Included here are:

data folder:

	test.csv: raw data meant for Kaggle users to submit their predictions
		I did not use this file

	train.csv: raw data describing the features of a property
		this is the file I uploaded and used to run and test my models

	continuous_v_categorical_analysis.txt: my description and analysis of the
		raw data along with how I intended to use the data

	data_description.txt: descriptions of the raw data provided by Kaggle

	model_evaluations.csv: a file describing the different models I ran and their
		corresponding R squared and mean squared error values
		downloaded from MySQL


Real_Estate_Prices_Predictions.py: the main Python script of the project that goes through
	evaluation of the data, transformation of the data, and modeling of the data

explore_data.py: a Python script that looks at the raw data and allows the user to see the content
	of the raw data

transform_data.py: a Python script that takes the raw data and transforms it so it can go into 
	models easily with features that will be useful

model_data.py: a Python script that takes the transformed data, runs it through models, and 
	evaluated the models puts the evaluations of the models into a MySQL database

model_evaluation_report.txt: a text file describing what I found from running the models
