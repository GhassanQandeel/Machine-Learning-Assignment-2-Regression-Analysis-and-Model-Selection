from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re
import requests
from sklearn.linear_model import LinearRegression

from scipy.stats import zscore
from sklearn.impute import SimpleImputer
# from geopy.geocoders import Nominatim
# from geopy.exc import GeocoderTimedOut
from mpl_toolkits.basemap import Basemap
import seaborn as sns
from sklearn.model_selection import train_test_split


# hello Ghassan how are u

def to_USD(rates, amount, target_currency):
    return float(amount) * rates.get(target_currency, 0)


url = f"https://v6.exchangerate-api.com/v6/4a5736bde4a84c84e2e208c3/latest/USD"
response = requests.get(url)

df = pd.read_csv("cars.csv")
# Display data statistics


print("Data Statistics:")
print("*****************************************************")
print(f"Number of Examples: {len(df)}")
print(f"Number of Features: {df.shape[1]}")
print("*******************************************************\n")

missing_summary = df.isnull().sum()
print("Missing Values Summary:\n", missing_summary)

# ###############################
# # To choose suitable missing value for our data set, We should see what missing values is look like
# missing_summary = df.isnull().sum()
# print("Missing Values Summary:\n", missing_summary)
# # As we see the cylinder feature have missing value and
# print("for cylinder summary :",df['cylinder'].describe())
# ###############################

# i perform to clean each column because the data is very dirty
# then replace it with strategy

#NOTE :::: There fucking outliers in engine_capacity


#Here we clean the seats from not nan value and not relly value  column
df['seats'] = df['seats'].apply(lambda row: np.nan if "Seater" not in str(row) else row)

#Here we clean the Top speed from not nan value and not relly value  column
df["top_speed"] = df["top_speed"].apply(lambda row: row if re.findall(r'\d\d\d', str(row)) != [] else np.nan)
#Here we clean the Horse power  from not nan value and not relly value  column

df["horse_power"] = df["horse_power"].apply(lambda row: row if re.findall(r'\d+', str(row)) != [] else np.nan)

df["cylinder"] = df["cylinder"].apply(lambda row: row if re.findall(r'\d+', str(row)) != [] else np.nan)
df["engine_capacity"] = df["engine_capacity"].apply(lambda row: row if re.findall(r'\d+', str(row)) != [] else np.nan)

#That i will replace it with api if i have time
currency_codes = [
    "USD", "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG",
    "AZN", "BAM", "BBD", "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB",
    "BRL", "BSD", "BTN", "BWP", "BYN", "BZD", "CAD", "CDF", "CHF", "CLP",
    "CNY", "COP", "CRC", "CUP", "CVE", "CZK", "DJF", "DKK", "DOP", "DZD",
    "EGP", "ERN", "ETB", "EUR", "FJD", "FKP", "FOK", "GBP", "GEL", "GGP",
    "GHS", "GIP", "GMD", "GNF", "GTQ", "GYD", "HKD", "HNL", "HRK", "HTG",
    "HUF", "IDR", "ILS", "IMP", "INR", "IQD", "IRR", "ISK", "JEP", "JMD",
    "JOD", "JPY", "KES", "KGS", "KHR", "KID", "KMF", "KRW", "KWD", "KYD",
    "KZT", "LAK", "LBP", "LKR", "LRD", "LSL", "LYD", "MAD", "MDL", "MGA",
    "MKD", "MMK", "MNT", "MOP", "MRU", "MUR", "MVR", "MWK", "MXN", "MYR",
    "MZN", "NAD", "NGN", "NIO", "NOK", "NPR", "NZD", "OMR", "PAB", "PEN",
    "PGK", "PHP", "PKR", "PLN", "PYG", "QAR", "RON", "RSD", "RUB", "RWF",
    "SAR", "SBD", "SCR", "SDG", "SEK", "SGD", "SHP", "SLE", "SLL", "SOS",
    "SRD", "SSP", "STN", "SYP", "SZL", "THB", "TJS", "TMT", "TND", "TOP",
    "TRY", "TTD", "TVD", "TWD", "TZS", "UAH", "UGX", "UYU", "UZS", "VES",
    "VND", "VUV", "WST", "XAF", "XCD", "XDR", "XOF", "XPF", "YER", "ZAR",
    "ZMW", "ZWL"
]

#Here we clean the Price column from not nan value and not relly value

df["price"] = df["price"].apply(lambda row: row if (str(row)[:3] in currency_codes) else np.nan)
if response.status_code == 200:
    data = response.json()
    rates = data["conversion_rates"]
    #Here we standard all values to USD USing API
    df["price"] = df["price"].apply(
        lambda row: to_USD(rates, str(row)[4:].replace(',', ''), str(row)[:3]) if str(row) != "nan" else np.nan)

else:
    print("Error fetching data:", response.status_code, response.text)

print("********************************")
#Here we will split numirical featuers from catrgorical
df["top_speed"] = df["top_speed"].apply(lambda row: float(row) if str(row) != "nan" else np.nan)
df["horse_power"] = df["horse_power"].apply(lambda row: float(row) if str(row) != "nan" else np.nan)
df["cylinder"] = df["cylinder"].apply(lambda row: float(row) if (row != np.nan) else np.nan)
df["engine_capacity"] = df["engine_capacity"].apply(lambda row: float(row) if str(row) != "nan" else np.nan)

numeric_columns = df.select_dtypes(include=['number']).columns
categorical_columns = df.select_dtypes(exclude=['number']).columns

print(numeric_columns)
print(categorical_columns)

print("we print the following pattern of data to what suitable missing value replacing ")
for col in numeric_columns:
    print(df[col].value_counts(), "\n/////////")

missing_summary = df.isnull().sum()
print("Missing Values Summary:\n", missing_summary)

#Engine MEdian
#Cylinder mean
#Horse power Mean
#Top Speed mean
df["top_speed"] = df["top_speed"].fillna(df["top_speed"].mean())
df["horse_power"] = df["horse_power"].fillna(df["horse_power"].mean())
df["cylinder"] = df["cylinder"].fillna(df["cylinder"].mean())
df["engine_capacity"] = df["engine_capacity"].fillna(df["engine_capacity"].median())
df["seats"] = df["seats"].fillna(df["seats"].mode()[0])

missing_summary = df.isnull().sum()
print("Missing Values Summary:\n", missing_summary)

#For the price column i will take the null values as test data set

print("we print the following pattern of data to what suitable missing value replacing For categorical columns ")
for col in categorical_columns:
    print(df[col].value_counts(), "\n/#/#/#/#/#/#/#/#/#")

categorical_columns = ['seats', 'brand', 'country']
for col in categorical_columns:
    df_encoded = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, df_encoded], axis=1)
    df = df.drop(columns=[col])

#Here we will split our data set to three data sets (Traning , validation, test):
#for test data set we will take the empty price value as Test data set
#so
test_DataSet = df[df['price'].isna()]
validation_DataSet_temp = df[df['price'].notna()]

validation_DataSet, training_DataSet = train_test_split(validation_DataSet_temp, test_size=0.75,
                                                        random_state=42)  # 0.75 of 80% is 60%
# validation_DataSet = pd.DataFrame(validation_DataSet)
# training_DataSet = pd.DataFrame(training_DataSet)

print(validation_DataSet)
print(test_DataSet)
print(training_DataSet)
#Know we will begin with linear regresion moel
#
# training_labels_x = training_DataSet.drop(['price'], axis=1)
# training_label_y = training_DataSet['price']
#
# model = LinearRegression()
# model.fit(training_labels_x, training_label_y)
