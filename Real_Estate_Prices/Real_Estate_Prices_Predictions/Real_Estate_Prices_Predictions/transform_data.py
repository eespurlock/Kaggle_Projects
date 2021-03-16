'''
This uses the insights gleaned from the explore_data file to transform our data
into a dataset that we can put into our machine learning pipeline
'''

#import statements
import pandas as pd

#a list of columns that we are not going to use
#!!!I have not written the code dealing with this yet!!!
DEL_COLS = ['ID', 'Utilities', 'BldgType', 'BsmtCond', 'Heating', 'GarageArea',
            'PoolQC', 'MiscFeature', 'MiscVal', 'SalePrice']

#a list of columns we will use as-is
AS_IS_COLS = ['LotArea', 'YearBuilt', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
              'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
              'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
              'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
              'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
              'ScreenPorch', 'PoolArea', 'LotFrontage', 'MasVnrArea']

#a list of categorical columns where we will create new columns for each value
#that indicates if a property falls into that category or not
CAT_COLS_LST = ["MasVnrType", "YrSold", "Neighborhood"]

#The following is to increase readability
#These are dictionaries of the various categorical columns describing the values
#in the original columns and the values we want to map them on to
SUBCLASS = {"1_Story": [20, 30, 40, 120],
            "1_1/2_Story": [45, 50, 150],
            "2_Story": [60, 70, 160],
            "2_1/2_Story": [75],
            "Split_Multi_Level": [80],
            "Split_Foyer": [85],
            "Duplex": [90],
            "PUD": [120, 160, 180],
            "2_Fam_Conver": [190]}
ZONING = {"Zoning_Res_Low_Density": ["RL"], 
          "Zoning_Res_Med_to_High_Density": ["RM", "RH"],
          "Zoning_Other": ["FV", "C (all)"]}
FOUNDATION = {"Concrete_Foundation": ["PConc"],
              "CinderBlock_Foundation": ["CBlock"],
              "BrickTile_Foundation": ["BrkTil"],
              "Foundation_Other": ["Slab", "Stone", "Wood"]}
GARAGE_TYPE = {"Attached_Garage": ["Attchd"],
               "Detached_Garage": ["Detchd"],
               "Other_Garage": ["2Types", "Basment", "BuiltIn", "CarPort"]}
SALETYPE = {"Conv_Warranty_Deed_Sale": ["WD"],
            "New_Sale": ["New"],
            "Other_Sale": ["CWD", "VWD", "COD", "Con", "ConLw", "ConLI", "ConLD", "Oth"]}
SALECOND = {"Normal_Condition": ["Normal"],
            "Abnormal_Condition": ["Abnorml"],
            "Partial_Condition": ["Partial"],
            "Other_Condition": ["AdjLand", "Alloca", "Family"]}
SALEMONTH = {"Jan_Sale": [1],
             "Feb_Sale": [2],
             "Mar_Sale": [3],
             "Apr_Sale": [4],
             "May_Sale": [5],
             "June_Sale": [6],
             "July_Sale": [7],
             "Aug_Sale": [8],
             "Sept_Sale": [9],
             "Oct_Sale": [10],
             "Nov_Sale": [11],
             "Dec_Sale": [12]}
LOTCONFIG = {"Inside_Lot": ["Inside"],
             "Corner_Lot": ["Corner"],
             "CulDuSac_Lot": ["CulDSac"],
             "Frontage_Lot": ["FR2","FR3"]}
ROOFSTYLE = {"Gable_Roof": ["Gable"],
             "Hip_Roof": ["Hip"],
             "Other_Roof": ["Flat", "Gambrel", "Mansard", "Shed"]}

#a dictionary of dictionaries describing the categorical columns in the dataset 
#and what the different values will map onto
CAT_COLS_DICT = {"MSSubClass": SUBCLASS, "MSZoning": ZONING, "Foundation": FOUNDATION,
                 "SaleType": SALETYPE, "SaleCondition": SALECOND, "MoSold": SALEMONTH,
                 "LotConfig": LOTCONFIG, "RoofStyle": ROOFSTYLE}

#a dictionary of tuples describing categorical columns that we will change to
#be 0 or 1. The tuple will indicate the new name of the column and the value(s)
#that will be mapped to 1
ZERO_ONE_DICT = {"Street": ("Is_paved", ["Pave"]),
                 "Alley": ("Alley", ["Grvl", "Pave"]),
                 "LandContour": ("Land_is_Level", ["Lvl"]),
                 "RoofMatl": ("Shingle_Roof", ["CompShg"]),
                 "CentralAir": ("Central_Air", ["Y"]),
                 "Electrical": ("Electrical_Standard", ["SBrkr"]),
                 "Functional": ("Typical_Functionality", ["Typ"]),
                 "PavedDrive": ("Fully_Paved_Drive", ["Y"]),
                 "Fence": ("Has_Fence", ["GdPrv", "MnPrv", "GdWo", "MnWw"]),
                 "HouseStyle": ("2nd_Level_Finished", ["1.5Fin", "2.5Fin"])}

#The following is to increase readability
#These are dictionaries of various continuous columns that need remapping
#describing the values in the original columns and the values we want to map them on to
LOTSHAPE = {"Reg": 0, "IR1": 1, "IR2": 2, "IR3": 3}
LANDSLOPE = {"Gtl": 0, "Mod": 1, "Sev": 2}

#a dictionary of dictionaries describing continuous variables that need to be
#remapped. The 2nd dictionary will describe what the original values need to
#be mapped to
#!!!I have not written the code dealing with this yet!!!
#!!!use this website for guidance on remapping
#https://www.geeksforgeeks.org/using-dictionary-to-remap-values-in-pandas-dataframe-columns/
CONT_REMAP_DICT = {"LotShape": LOTSHAPE, "LandSlope": LANDSLOPE}

#a list of tuples with a column name and the value that the column's n/a value
#should be mapped onto
NA_HANDLING_COLS = [("LotFrontage", 0), ("MasVnrArea", "MEAN")]

# !!! columns that I have not dealt with yet !!!
#Codition 1 / Condition 2
#OverallQual / OverallCond
#YearRemodAdd
#Exterior1st / Exterior2nd
#ExterQual / ExterCond
#BsmtQual
#BsmtExposure
#BsmtFinType1 / BsmtFinType2 / BsmtFinSF1 / BsmtFinSF2 / BsmtUnfSF
#HeatingQC
#KitchenQual
#FireplaceQu
#GarageFinish
#GarageQual / GarageCond

def main(df):
    '''
    Goes through the transformation of the data

    Inputs: df: a Pandas dataframe we want to transform

    Outputs: new_df: the pandas dataframe we have created by transforming df
    '''
    #create a new dataframe that is empty
    new_df = pd.DataFrame()
    #first, we deal with n/a values so the columns can be transformed with no issue
    df = fill_na(df, NA_HANDLING_COLS)
    #!!!We are going to re-write these lines to be more functional programming
    #but we are saving that until after all of the other code is written!!!
    new_df = as_is_transform(df, new_df, AS_IS_COLS)
    new_df = categorical_transform(df, new_df, CAT_COLS_LST)
    new_df = categorical_transform_2(df, new_df, CAT_COLS_DICT)
    new_df = zero_one_transform(df, new_df, ZERO_ONE_DICT)
    return(new_df)

def fill_na(df, cols_lst):
    '''
    Goes through the columns that need n/a values to be filled and fills
        them with the appropriate value

    Inputs:
        df: the pandas dataframe we are using
        cols_lst: a list of tuples that indicates the column names and the values
            the n/a values should be mapped onto

    Outputs: df: the pandas dataframe with the n/a values filled in
    '''
    for tup in cols_lst:
        colname, new_value = tup
        if new_value == "MEAN":
            new_value = df[colname].mean()
        df[colname] = df[colname].fillna(new_value)

def as_is_transform(df, new_df, cols_lst):
    '''
    Takes columns that will stay the same from df and add them to new_df

    Inputs:
        df: the original pandas dataframe
        new_df: the pandas dataframe we are in the process of creating
        cols_lst: the list of column names that we are adding to new_df from df

    Outputs: new_df: the pandas dataframe we are creating by transforming df
    '''
    #loop through the columns in the list
    for col in cols_lst:
        new_df[col] = df[col]
    return(new_df)

def categorical_transform(df, new_df, cols_lst):
    '''
    Takes categorical columns and creates new columns for each of the unique
        values. The columns created will indicate if a property has that value
        or not.
    
     Inputs:
        df: the original pandas dataframe
        new_df: the pandas dataframe we are in the process of creating
        cols_lst: the list of categorical column names that we need to transform

    Outputs: new_df: the pandas dataframe we are creating by transforming df
    '''
    #loop through the columns in the list
    for col in cols_lst:
        #find the unique values in the column
        unique_vals = df[col].unique()
        #loop through the unique values and create a new column
        for val in unique_vals:
            #create a new column name
            new_colname = col + "_" + str(val)
            #creat a new column based on the old one
            new_df[new_colname] = df[col].apply(lambda x: 1 if x == val else 0)
    return(new_df)

def categorical_transform_2(df, new_df, cols_dict):
    '''
    Takes categorical columns and creates new columns for a subset of the unique
        values. The columns created will indicate if a property is in that subset
        or not.
    
     Inputs:
        df: the original pandas dataframe
        new_df: the pandas dataframe we are in the process of creating
        cols_dict: a dictionary of dictionaries describing what the
            values in the column are and the value they should be mapped onto

    Outputs: new_df: the pandas dataframe we are creating by transforming df
    '''
    #loop through the dictionary
    for colname, values_dict in cols_dict.items():
        for new_colname, values_lst in values_dict.items():
            #create a new column based on the old one
            new_df[new_colname] = df[colname].apply(lambda x: 1 if x in values_lst else 0)
    return(new_df)

def zero_one_transform(df, new_df, cols_dict):
    '''
    Takes categorical columns and indicates if they are a certain value or not

    Inputs:
        df: the original pandas dataframe
        new_df: the pandas dataframe we are in the process of creating
        cols_dict: a dictionary of tuples describing what values in the column 
            should be mapped on to 1

    Outputs: new_df: the pandas dataframe we are creating by transforming df
    '''
    #loop through the dictionary
    for colname, tup in cols_dict.items():
        new_colname, one_lst = tup
        #create a new column based on the old one
        new_df[new_colname] = df[colname].apply(lambda x: 1 if x in one_lst else 0)
    return(new_df)
