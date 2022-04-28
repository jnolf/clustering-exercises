import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

import os
import env
from env import username, password, host
import acquire

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
import sklearn.preprocessing

################################## Acquire #####################################

def acquire_zillow(use_cache=True):
    ''' 
    This function acquires all necessary housing data from zillow 
    needed to better understand future pricing
    '''    
    if os.path.exists('zillow.csv') and use_cache:
        print('Using cached csv')
        return pd.read_csv('zillow.csv')
    print('Acquiring data from SQL database')

    database_url_base = f'mysql+pymysql://{username}:{password}@{host}/zillow'
    query = '''
            SELECT prop.*, 
               pred.logerror, 
               pred.transactiondate, 
               air.airconditioningdesc, 
               arch.architecturalstyledesc, 
               build.buildingclassdesc, 
               heat.heatingorsystemdesc, 
               landuse.propertylandusedesc, 
               story.storydesc, 
               construct.typeconstructiondesc 
               FROM   properties_2017 prop 
               
               INNER JOIN (SELECT parcelid,
                                  logerror,
                                  Max(transactiondate) transactiondate 
                           FROM   predictions_2017 
                           GROUP  BY parcelid, logerror) pred
                       USING (parcelid)
                       
               LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
               LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
               LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
               LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
               LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
               LEFT JOIN storytype story USING (storytypeid) 
               LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
               WHERE  prop.latitude IS NOT NULL 
               AND prop.longitude IS NOT NULL AND transactiondate <= '2017-12-31' 
'''
    df = pd.read_sql(query, database_url_base)
    df.to_csv('zillow.csv', index=False)
   
    return df


############################### Deal With Missing Values #############################
########################### Missing Values Counts By Column ##########################

def missing_values_by_column(df):
    missing_cols = pd.concat([
                   df.isna().sum().rename('count'),
                   df.isna().mean().rename('percent')
                   ], axis=1)
    return missing_values

############################ Missing Values Counts By Row ############################

def missing_values_by_row(df):
    missing_rows = pd.concat([
                   df.isna().sum(axis=1).rename('num_cols_missing'),
                   df.isna().mean(axis=1).rename('pct_cols_missing'),
                   ], axis=1).value_counts().sort_index()
    return pd.DataFrame(missing_counts_and_percents).reset_index()

######################### Drop Missing Values Based on Pct ###########################

def handle_missing_values(df, prop_required_column = .6, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

####################### Remove Additional Unneeded Columns ##########################

# MUST PROVIDE A SPECIFIED "cols_to_remove = ['']" prior
def remove_cols(df, cols_to_remove):
    df = df.drop(columns = cols_to_remove)
    return df

############################### Deal With Outliers #################################

def remove_outliers(df, k, cols):
    ''' this function take in a dataframe, k value, and specified columns 
    within a dataframe and then return the dataframe with outliers removed
    '''
    for col in cols:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#################### Final Wrangle To Fill in Any Useful Nulls #######################

def wrangle_zillow():
    df = pd.read_csv('zillow.csv')    
    # Identify the use codes that are single family from SequelAce
    single_fam_use = [261, 262, 263, 264, 265, 266, 268, 273, 275, 276, 279]
    # Return a df with only single family use properties
    df = df[df.propertylandusetypeid.isin(single_fam_use)]    
    # Restrict df to only those properties with at least 1 bath & bed and >500 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull()) 
            & (df.calculatedfinishedsquarefeet>500)]        
    # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)    
    # Make a column for building age
    df['age'] = 2017 - df.yearbuilt
    # Make a column for county based on FIPS
    df['county'] = df.fips.replace({6037:'Los Angeles',
                           6059:'Orange',          
                           6111:'Ventura'})    
    # Remove unnecessary columns
    df = remove_cols(df, ['id',
            'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 
            'heatingorsystemtypeid','propertycountylandusecode', 
            'propertylandusetypeid','propertyzoningdesc', 
            'propertylandusedesc', 'unitcnt', 'censustractandblock'])   
    # Fill in nuls and NaNs with desired values then drop anythign else not needed
    df['lotsizesquarefeet'].fillna(value = df.lotsizesquarefeet.median()
                                   , inplace = True)
    df['buildingqualitytypeid'].fillna(value = df.buildingqualitytypeid.median()
                                       , inplace = True)
    df['yearbuilt'].fillna(value = df.yearbuilt.mean()
                           , inplace = True)
    df['heatingorsystemdesc'].fillna(value = df.heatingorsystemdesc.mode()[0]
                                     , inplace = True)
    df['structuretaxvaluedollarcnt'].fillna(value = df.taxvaluedollarcnt-df.landtaxvaluedollarcnt
                                            , inplace = True)
    cols = ['regionidzip', 'regionidcity']
    for col in cols:
        mode = int(df[col].mode())
        df[col].fillna(value = mode, inplace = True)
    df.dropna(inplace = True)
    return df

############################# Split #################################

def split_data(df):
    ''' 
    This function will take in the data and split it into train, 
    validate, and test datasets for modeling, evaluating, and testing
    '''
    train_val, test = train_test_split(df, train_size = .8, random_state = 123)
    train, validate = train_test_split(train_val, train_size = .7, random_state = 123)
    return train, validate, test

################################### Scale Our Data #################################

def min_max_df(df):
    '''
    Scales the df. using the MinMaxScaler()
    takes in the df and returns the df in a scaled fashion.
    '''
    # Create the scaler
    scaler = sklearn.preprocessing.MinMaxScaler()
    # Fit the scaler 
    scaler.fit(df)
    # Transform and rename columns for the df
    df_scaled = pd.DataFrame(scaler.transform(train), columns = train.columns.tolist())
    return df_scaled

def min_max_split(train, validate, test):
    '''
    Scales the 3 data splits. using the MinMaxScaler()
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    If return_scaler is true, the scaler object will be returned as well.
    '''
    # Create the scaler
    scaler = sklearn.preprocessing.MinMaxScaler()
    # Fit scaler on train dataset
    scaler.fit(train)
    # Transform and rename columns for all three datasets
    train_scaled = pd.DataFrame(scaler.transform(train), columns = train.columns.tolist())
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns = train.columns.tolist())
    test_scaled = pd.DataFrame(scaler.transform(test), columns = train.columns.tolist())
    return train_scaled, validate_scaled, test_scaled

####################################################################################
################################### Mall Stuff #####################################

# def get_db_url(database):
#     from env import host, user, password
#     url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
#     return 

def get_mall_data(use_cache=True):
    filename = "mall_customers.csv"
    if os.path.isfile(filename) and use_cache:
        print("Using cache file...")
        return pd.read_csv(filename)
    print("Acquiring data from SQL database...")
    data = 'mall_customers'
    url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{data}'
    mall = pd.read_sql('SELECT * FROM customers', url)
    mall.to_csv(filename)
    return mall

def min_max_mall(train, validate, test):
    columns_to_scale = ['age', 'spending_score', 'annual_income']
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    scaler = MinMaxScaler()
    scaler.fit(train[columns_to_scale])

    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])

    return train_scaled, validate_scaled, test_scaled

def outlier_function(df, cols, k):
	#function to detect and handle oulier using IQR rule
    for col in df[cols]:
        q1 = df.annual_income.quantile(0.25)
        q3 = df.annual_income.quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return 

def wrangle_mall():
    
    # acquire data
    sql = 'select * from customers'


    # acquire data from SQL server
    mall = get_mall_data(sql)
    
    # handle outliers
    mall = outlier_function(mall, ['age', 'spending_score', 'annual_income'], 1.5)
    
    # get dummy for gender column
#     dummy_df = pd.get_dummies(mall.gender, drop_first=True)
#     mall = pd.concat([mall, dummy_df], axis=1).drop(columns = ['gender'])
#     mall.rename(columns= {'Male': 'is_male'}, inplace = True)
    # return mall_df
    return mall