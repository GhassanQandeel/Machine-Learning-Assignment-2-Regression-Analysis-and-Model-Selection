
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re
import requests

from scipy.stats import zscore
from sklearn.impute import SimpleImputer
# from geopy.geocoders import Nominatim
# from geopy.exc import GeocoderTimedOut
from mpl_toolkits.basemap import Basemap
import seaborn as sns

# hello Ghassan how are u 
def to_USD(rates , amount , target_currency):
    return float(amount) * rates.get(target_currency, 0)


url = f"https://v6.exchangerate-api.com/v6/4a5736bde4a84c84e2e208c3/latest/USD"
response = requests.get(url)


df =pd.read_csv("cars.csv")
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
df["top_speed"] = df["top_speed"].apply(lambda row: row if  re.findall(r'\d\d\d', str(row)) != [] else np.nan)
#Here we clean the Horse power  from not nan value and not relly value  column

df["horse_power"]=df["horse_power"].apply(lambda row: row if  re.findall(r'\d+', str(row)) != [] else np.nan)

df["cylinder"]=df["cylinder"].apply(lambda row: row if re.findall(r'\d+', str(row)) != [] else np.nan)
df["engine_capacity"]=df["engine_capacity"].apply(lambda row: row if re.findall(r'\d+', str(row)) != [] else np.nan)



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

df["price"]=df["price"].apply(lambda row: row if (str(row)[:3] in currency_codes) else np.nan)
if response.status_code == 200:
    data = response.json()
    rates = data["conversion_rates"]
    #Here we standard all values to USD USing API
    df["price"] = df["price"].apply(lambda row: to_USD(rates,str(row)[4:].replace(',',''),str(row)[:3]) if str(row) !="nan" else np.nan)

else:
    print("Error fetching data:", response.status_code, response.text)


print("********************************")
#Here we will split numirical featuers from catrgorical
df["top_speed"] = df["top_speed"].apply(lambda row: float(row) if str(row) != "nan" else np.nan)
df["horse_power"]=df["horse_power"].apply(lambda row: float(row) if str(row) != "nan" else np.nan)
df["cylinder"]=df["cylinder"].apply(lambda row: float(row) if (row != np.nan) else np.nan)
df["engine_capacity"]=df["engine_capacity"].apply(lambda row: float(row) if str(row) != "nan" else np.nan)

numeric_columns = df.select_dtypes(include=['number']).columns
categorical_columns = df.select_dtypes(exclude=['number']).columns




print(numeric_columns)
print(categorical_columns)




print("we print the following pattern of data to what suitable missing value replacing ")
for col in numeric_columns:
    print(df[col].value_counts(),"\n/////////")

#Engine MEdian
#Cylinder mean


missing_summary = df.isnull().sum()
print("Missing Values Summary:\n", missing_summary)
