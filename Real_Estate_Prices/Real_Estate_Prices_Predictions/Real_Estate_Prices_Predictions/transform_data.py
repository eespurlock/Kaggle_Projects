'''
This file uses the insights gleaned from the explore_data file to transform our data
into a dataset that we can put into our machine learning pipeline
'''

#import statements
import pandas as pd

#a list of columns that we are not going to use
#I am not writing code to deal with these columns because they will
#simply not be added to the new dataframe I create
#DEL_COLS = ['ID', 'Utilities', 'BldgType', 'BsmtCond', 'Heating', 'GarageArea',
#            'PoolQC', 'MiscFeature', 'MiscVal']

#Constant that is our feature column
FEATURE = 'SalePrice'

#a list of columns we will use as-is
AS_IS_COLS = ['LotArea', 'YearBuilt', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
              'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
              'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
              'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
              'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
              'ScreenPorch', 'PoolArea', 'LotFrontage', 'MasVnrArea',
              'OverallQual', 'OverallCond', 'BsmtUnfSF']

#a list of categorical columns where we will create new columns for each value
#that indicates if a property falls into that category or not
CAT_COLS_LST = ["MasVnrType", "YrSold", "Neighborhood", "YearRemodAdd"]

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
BASEEXPOSURE = {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NA": 0}
BASEQUAL = {"Ex": 100, "Gd": 90, "TA": 80, "Fa": 70, "NA": 0}
EX_TO_PO = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
GARAGEFINISH = {"Fin": 3, "RFn": 2, "Unf": 1, "NA": 0}

#a dictionary of dictionaries describing continuous variables that need to be
#remapped. The 2nd dictionary will describe what the original values need to
#be mapped to
CONT_REMAP_DICT = {"LotShape": LOTSHAPE, "LandSlope": LANDSLOPE, "BsmtExposure": BASEEXPOSURE,
                   "BsmtQual": BASEQUAL, "HeatingQC": EX_TO_PO, "KitchenQual": EX_TO_PO,
                   "FireplaceQu": EX_TO_PO, "GarageFinish": GARAGEFINISH,
                   "ExterQual": EX_TO_PO, "ExterCond": EX_TO_PO,
                   "GarageQual": EX_TO_PO, "GarageCond": EX_TO_PO}

#a list of tuples with a column name and the value that the column's n/a value
#should be mapped onto
NA_HANDLING_COLS = [("LotFrontage", 0), ("MasVnrArea", "MEAN"), ("BsmtExposure", "NA"),
                    ("BsmtQual", "NA"), ("FireplaceQu", "NA"), ("GarageFinish", "NA"),
                    ("GarageQual", "NA"), ("GarageCond", "NA"),
                    ("BsmtFinType1", "NA"), ("BsmtFinType2", "NA"),
                    ("GarageYrBlt", 0)]

#a list of tuples with a column name beginning and the names of 2 columns that
#track the same metric. The values in both of these columns need to become
#their own column that indicates if either of these columns holds a specific
#value
SAME_METRIC_COLS = [("Condition", ("Condition1", "Condition2")),
                    ("Exterior", ("Exterior1st", "Exterior2nd"))]

#a list of the basement types and square footage that we need to combine
BASEMENT_SQ_FT_COLS = [('BsmtFinType1', 'BsmtFinSF1'), ('BsmtFinType2', 'BsmtFinSF2')]

def main(df):
    '''
    Goes through the transformation of the data

    Inputs: df: a Pandas dataframe we want to transform

    Outputs: new_df: the pandas dataframe we have created by transforming df
        feature_df: a pandas dataframe with just the feature we want to predict for
    '''
    #a list of tuples of the functions we need to go through and their input
    FUNCTIONS_LST = [(as_is_transform, AS_IS_COLS),
        (categorical_transform, CAT_COLS_LST),
        (categorical_transform_2, CAT_COLS_DICT),
        (zero_one_transform, ZERO_ONE_DICT),
        (remap_transform, CONT_REMAP_DICT),
        (combine_same_metric_transform, SAME_METRIC_COLS),
        (sq_ft_combination, BASEMENT_SQ_FT_COLS)]
    #create a new dataframe that is empty for both features and variables
    new_df = pd.DataFrame()
    feature_df = pd.DataFrame()

    #creates the feature dataframe
    feature_df[FEATURE] = df[FEATURE]

    #first, we deal with n/a values so the columns can be transformed with no issue
    df = fill_na(df, NA_HANDLING_COLS)
    
    #loop through the functions we need to transform the data
    for tup in FUNCTIONS_LST:
        funct, input = tup
        new_df = funct(df, new_df, input)

    return(new_df, feature_df)

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
    return(df)

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

def remap_transform(df, new_df, cols_dict):
    '''
    Takes columns that are currently strings and remaps them to ints based
    on the values in the dictionary

    Inputs:
        df: the original pandas dataframe
        new_df: the pandas dataframe we are in the process of creating
        cols_dict: a dictionary of dictionaries describing what the values 
            in the column should be mapped on to

    Outputs: new_df: the pandas dataframe we are creating by transforming df
    '''
    #get the column names for new_df
    all_new_colnames = new_df.columns
    #loop through the dictionary
    for colname, dict in cols_dict.items():
            new_df[colname] = df[colname].map(dict)
    return(new_df)

def combine_same_metric_transform(df, new_df, cols_lst):
    '''
    This function takes two columns that have the same values in them and creates
        new columns that will say if the unique values are in either of the two columns
    
    Inputs:
        df: the original pandas dataframe
        new_df: the pandas dataframe we are in the process of creating
        cols_lst: a list of tuples describing the columns that have the same
            values in them

    Outputs: new_df: the pandas dataframe we are creating by transforming df

    '''
    #loop through the list
    for tup in cols_lst:
        #create a list that will hold the unique values from both column 1 and column 2
        col_beginning, col_tup = tup
        col1, col2 = col_tup
        two_col_unique = get_unique(df, col1, col2)
        #check to see if a vlue is in either column
        for val in two_col_unique:
            new_colname = col_beginning + "_" + val
            new_df[new_colname] = (df[[col1, col2]].eq(val)).any(axis=1)
            new_df[new_colname] = new_df[new_colname].apply(lambda x: 1 if x else 0)
    return(new_df)

def get_unique(df, col1, col2):
    '''
    Takes 2 columns from a dataframe and gets the unique values that are
        are across both of them

    Inputs:
        df: the dataframe with the 2 columns
        col1: (string) the name of the 1st columns
        col2: (string) the name of the 2nd column

    Outputs: two_col_unique: a list of the unique values in the 2 columns
    '''
    two_col_unique = [] 
    for val in df[col1].unique():
        two_col_unique.append(val)
    for val in df[col2].unique():
        if val not in two_col_unique:
            two_col_unique.append(val)
    return(two_col_unique)

def sq_ft_combination(df, new_df, cols_lst):
    '''
    This function is specifically written to combine the types of basement
        finishes with the square footage of that finish type

    Inputs:
        df: the original pandas dataframe
        new_df: the pandas dataframe we are in the process of creating
        cols_lst: a list of the basement types and square feet that we
            want to use to create new columns

    Outputs: new_df: the pandas dataframe we are creating by transforming df
    '''
    #get the unique values of the two columns
    two_col_unique_dirty = get_unique(df, cols_lst[0][0], cols_lst[1][0])
    #we now need to take the values "NA", "Unf" out of our list of unique values
    #unfinished square footage has already been recorded in its own column
    #and na values merely mean there is no basement
    two_col_unique=[]
    for val in two_col_unique_dirty:
        if val not in ["NA", "Unf"]:
            two_col_unique.append(val)
    #create a counter
    count = 1
    
    #now we will combine the type of the basement type with its square footage
    for type_sqft_tup in cols_lst:
        bsmt_type, sqft = type_sqft_tup
        df[str(count)] = list(zip(df[bsmt_type], df[sqft]))
        count += 1
    
    #now we need to loop through the 2 new columns
    for num in ["1", "2"]:
        #and we need to loop through the unique values
        for val in two_col_unique:
            new_colname = val + num
            df[new_colname] = df[num].apply(lambda x: x[1] if x[0] == val else 0)

    #now we combine the columns that track the same type
    for val in two_col_unique:
        col1 = val + "1"
        col2 = val + "2"
        new_colname = "Bsmt_" + val + "_Sqft"
        new_df[new_colname] = df[col1] + df[col2]

    return(new_df)
