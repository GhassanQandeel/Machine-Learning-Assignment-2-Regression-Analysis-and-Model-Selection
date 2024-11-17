
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
# from geopy.geocoders import Nominatim
# from geopy.exc import GeocoderTimedOut
from mpl_toolkits.basemap import Basemap
import seaborn as sns

df =pd.read_csv("cars.csv")
# Display data statistics
print("Data Statistics:")
print("*****************************************************")
print(f"Number of Examples: {len(df)}")
print(f"Number of Features: {df.shape[1]}")
print("*******************************************************\n")
###############################
# To choose suitable missing value for our data set, We should see what missing values is look like
missing_summary = df.isnull().sum()
print("Missing Values Summary:\n", missing_summary)
# As we see the cylinder feature have missing value and
print("for cylinder summary :",df['cylinder'].describe())
# we can see the most value repeated is 4 for more than 2856 so we  wil replace the missing value with mode
###############################
