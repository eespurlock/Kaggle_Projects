'''
This file goes through the data exploration process

I am only going to explore the training dataset and then use those insights to
transform both the training and the testing datasets
'''

def main(df):
    '''
    Goes through the whole data exploration process

    Inputs: df: a pandas dataframe we want to explore

    Outputs: None
    '''
    #we get the column names of the dataframe
    colnames = df.columns
    continuous_v_categorical(df, colnames)

def continuous_v_categorical(df, colnames):
    '''
    Looks through each of the columns in a dataframe so we can determine if the column
    is continuous or categorical, and also gives us a sense of how many n/a values are
    in a given column

    My analysis of this output can be found in the continuout_v_categorical_analyses text
    file in the data folder. This analysis also draws from information given in the
    data_description text file also in the data folder. As part of this analysis, I also used iPython
    to get a deeper look at the composition of each column. The code is not included here.

    Inputs:
        df: a pandas dataframe we are exploring
        colnames: a python index object with the names of the columns

    Outputs: None
    '''
    #loops through all the column names
    for col in colnames:
        #creates an object for the column we are using
        curr_col = df[col]
        print(col)
        #get the number of unique values
        unique_num = curr_col.nunique()
        if unique_num < 100:
            print(curr_col.unique())
        else:
            print(unique_num)
        #print number of n/a values
        print("The number of n/a values: " + str(curr_col.isna().sum()))


        
