import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize']=(10,18)
from datetime import datetime
from datetime import date
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns 
import warnings
from datetime import datetime, date
sns.set()



# `parse_dates` will recognize the column is date time
data = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/NYCtaxi/train.csv', parse_dates=['pickup_datetime'])
test = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/NYCtaxi/test.csv', parse_dates=['pickup_datetime'])
data.head(3)


# add holiday information
holiday = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/holiday/NYC_2016Holidays.csv',sep=';')


time_data.to_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/time_data.csv',index=False)
time_test.to_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/time_test.csv',index=False)

# add ORSM information
fastrout1 = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/NYCTaxi_OSRM/fastest_routes_train_part_1.csv',
                        usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps','step_direction'])
fastrout2 = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/NYCTaxi_OSRM/fastest_routes_train_part_2.csv',
                        usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps','step_direction'])


osrm_data = data[['total_distance','total_travel_time','number_of_steps','right_steps','left_steps']]
osrm_test = test[['total_distance','total_travel_time','number_of_steps','right_steps','left_steps']]

data.to_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/osrm_data.csv',index=False,
            columns = ['total_distance','total_travel_time','number_of_steps','right_steps','left_steps'])
test.to_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/osrm_test.csv',index=False,
            columns = ['total_distance','total_travel_time','number_of_steps','right_steps','left_steps'])


Other_dist_data.to_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/Other_dist_data.csv',index=False)
Other_dist_test.to_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/Other_dist_test.csv',index=False)



data.to_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/kmean10_data.csv',index=False,columns = ['pickup_dropoff_loc'])
test.to_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/kmean10_test.csv',index=False,columns = ['pickup_dropoff_loc'])



# add weather information
weather = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/weather/KNYC_Metars.csv', parse_dates=['Time'])


weather_data.to_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/weather_data.csv',index=False)
weather_test.to_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/weather_test.csv',index=False)

outliers = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/outliers.csv')
time_data       = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/time_data.csv')
weather_data    = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/weather_data.csv')
osrm_data       = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/osrm_data.csv')
Other_dist_data = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/Other_dist_data.csv')
kmean10_data    = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/kmean10_data.csv')

time_test       = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/time_test.csv')
weather_test    = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/weather_test.csv')
osrm_test       = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/osrm_test.csv')
Other_dist_test = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/Other_dist_test.csv')
kmean10_test    = pd.read_csv('/Users/yuki/Desktop/big_data_analytics/project_bigdata/data/features/kmean10_test.csv')




