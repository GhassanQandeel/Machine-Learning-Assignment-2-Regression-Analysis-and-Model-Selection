from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re
import requests
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from IPython.display import display



import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split


# hello Ghassan how are u

def to_USD(rates, amount, target_currency):
    return float(amount) * rates.get(target_currency, 0)


url = f"https://mocki.io/v1/6b55e2fc-4bfd-4e13-9589-30636717e6ce"
response = requests.get(url)

df = pd.read_csv("cars.csv")
# Display data statistics


print("Data Statistics:")
print("*****************************************************")
print(f"Number of Examples: {len(df)}")
print(f"Number of Features: {df.shape[1]}")
print("*******************************************************\n")
# i perform to clean each column because the data is very dirty
# then replace it with strategy

# Here we clean the seats from not nan value and not relly value  column
df['seats'] = df['seats'].apply(lambda row: np.nan if "Seater" not in str(row) else row)
# Here we clean the Top speed from not nan value and not relly value  column
df["top_speed"] = df["top_speed"].apply(lambda row: row if re.findall(r'\d\d\d', str(row)) != [] else np.nan)
# Here we clean the Horse power  from not nan value and not relly value  column
df["horse_power"] = df["horse_power"].apply(lambda row: row if re.findall(r'\d+', str(row)) != [] else np.nan)
df["cylinder"] = df["cylinder"].apply(lambda row: row if re.findall(r'\d+', str(row)) != [] else np.nan)
df["engine_capacity"] = df["engine_capacity"].apply(lambda row: row if re.findall(r'\d+', str(row)) != [] else np.nan)

# json file url
json_url=f"https://mocki.io/v1/24326926-978b-4c04-a7f8-d79022e96d6f"
response_codes=requests.get(json_url)
if response_codes.status_code==200:
    data_codes=response_codes.json()
    codes=data_codes["currency_codes"]
    # Here we clean the Price column from not nan value and not relly value
    df["price"] = df["price"].apply(lambda row: row if (str(row)[:3] in codes) else np.nan)
if response.status_code == 200:
    data = response.json()
    rates = data["conversion_rates"]
    # Here we standard all values to USD USing API
    df["price"] = df["price"].apply(
        lambda row: to_USD(rates, str(row)[4:].replace(',', ''), str(row)[:3]) if str(row) != "nan" else np.nan)
else:
    print("Error fetching data:", response.status_code, response.text)

scaler = StandardScaler()
df['price'] = scaler.fit_transform(df[['price']])

print("********************************")

# Here we will split numerical features from categorical
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

# Engine MEdian
# Cylinder mean
# Horse power Mean
# Top Speed mean
df["top_speed"] = df["top_speed"].fillna(df["top_speed"].mean())
df["horse_power"] = df["horse_power"].fillna(df["horse_power"].mean())
df["cylinder"] = df["cylinder"].fillna(df["cylinder"].mean())
df["engine_capacity"] = df["engine_capacity"].fillna(df["engine_capacity"].median())
df["seats"] = df["seats"].fillna(df["seats"].mode()[0])

missing_summary = df.isnull().sum()
print("Missing Values Summary:\n", missing_summary)

# For the price column i will take the null values as test data set

categorical_columns = ['seats', 'brand', 'country']
for col in categorical_columns:
    df_encoded = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, df_encoded], axis=1)
    df = df.drop(columns=[col])

# Here we will split our data set to three data sets (Traning , validation, test):
# for test data set we will take the empty price value as Test data set
# so
test_DataSet = df[df['price'].isna()]
validation_DataSet_temp = df[df['price'].notna()]
validation_DataSet, training_DataSet = train_test_split(validation_DataSet_temp, test_size=0.75,random_state=42)  # 0.75 of 80% is 60%
#############################################################
# end of preprocessing
#############################################################
# Know we will begin with linear regression model

training_labels_x = training_DataSet.drop(['price'], axis=1)
training_labels_x = training_labels_x.drop(['car name'], axis=1)
training_label_y = training_DataSet['price']

validation_labels_x = validation_DataSet.drop(['price'], axis=1)
validation_labels_x = validation_labels_x.drop(['car name'], axis=1)
validation_label_y = validation_DataSet['price']

test_labels_x = test_DataSet.drop(['price'], axis=1)
test_labels_x = test_labels_x.drop(['car name'], axis=1)
test_label_y = test_DataSet['price']

model = LinearRegression()
model.fit(training_labels_x, training_label_y)

y_val_pred = model.predict(validation_labels_x)
lin_mse = mean_squared_error(validation_label_y, y_val_pred)
lin_r2 = r2_score(validation_label_y, y_val_pred)
lin_mae = mean_absolute_error(validation_label_y, y_val_pred)




#For LASSO Regression i will search for best alpha (Hyper parameter )
print("Here we print lass and ridge reg for 0.01, 0.1, 1, 10, 100 alpha values ")
print("*****************************************************")
param_grid = [0.01, 0.1, 1, 10, 100]
for alpha in param_grid:
    lasso = Lasso(alpha=alpha)
    ridge = Ridge(alpha=alpha)
    lasso.fit(training_labels_x, training_label_y)
    ridge.fit(training_labels_x, training_label_y)
    print(f"Lasso Regression Model Performance for alpha: {alpha}:")
    mse_lasso = mean_squared_error(validation_label_y, lasso.predict(validation_labels_x))
    r2_lasso = r2_score(validation_label_y, lasso.predict(validation_labels_x))
    mae_lasso = mean_absolute_error(validation_label_y, lasso.predict(validation_labels_x))
    print(f"Mean Squared Error: {mse_lasso:.2f}")
    print(f"R² Score: {r2_lasso:.2f}")
    print(f"Mean Absolute Error: {mae_lasso:.2f}")
    print("//////////////////////////////////////////////////")
    print(f"Ridge Regression Model Performance for alpha: {alpha}:")
    mse_ridge = mean_squared_error(validation_label_y, ridge.predict(validation_labels_x))
    r2_ridge = r2_score(validation_label_y, ridge.predict(validation_labels_x))
    mae_ridge = mean_absolute_error(validation_label_y, ridge.predict(validation_labels_x))
    print(f"Mean Squared Error: {mse_ridge:.2f}")
    print(f"R² Score: {r2_ridge:.2f}")
    print(f"Mean Absolute Error: {mae_ridge:.2f}")

print("*****************************************************")
param = {'alpha': [0.01, 0.1,1,10, 100]}

grid_search_lasso = GridSearchCV(estimator=lasso, param_grid=param, scoring='r2', cv=5)
grid_search_lasso.fit(training_labels_x, training_label_y)
print("optimal alpha for lasso", grid_search_lasso.best_params_)
mse_lasso = mean_squared_error(validation_label_y, grid_search_lasso.predict(validation_labels_x))
r2_lasso = r2_score(validation_label_y, grid_search_lasso.predict(validation_labels_x))
mae_lasso = mean_absolute_error(validation_label_y, grid_search_lasso.predict(validation_labels_x))
grid_search_ridge = GridSearchCV(estimator=ridge, param_grid=param, scoring='r2', cv=5)
grid_search_ridge.fit(training_labels_x, training_label_y)
print("optimal alpha for ridge", grid_search_ridge.best_params_)
mse_ridge = mean_squared_error(validation_label_y, grid_search_ridge.predict(validation_labels_x))
r2_ridge = r2_score(validation_label_y, grid_search_ridge.predict(validation_labels_x))
mae_ridge = mean_absolute_error(validation_label_y, grid_search_ridge.predict(validation_labels_x))




# Okay , As i understand we need to construct closed form solution for our data set then compare it with linear reg model
x = np.array(training_labels_x)
y = np.array(training_label_y)
m, n = x.shape

# Here for column multiply it with last parameter without feature
x = np.hstack((np.ones((m, 1)), x))


# w = ( ( X.T * X )^ −1) * ( X.T ) * y
def closed_form_solution(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return (np.linalg.pinv(X.T @ X)) @ X.T @ y


# Here the parameter of the closed form solution
closed_form = closed_form_solution(x, y)

x_validation_array = np.array(validation_labels_x)
i, j = x_validation_array.shape
x_validation_array = np.hstack((np.ones((i, 1)), x_validation_array))

y_val_pred_from_closed_form = x_validation_array @ closed_form
closed_form_mse = mean_squared_error(validation_label_y, y_val_pred_from_closed_form)
closed_form_r2 = r2_score(validation_label_y, y_val_pred_from_closed_form)
closed_form_mae = mean_absolute_error(validation_label_y, y_val_pred_from_closed_form)



# Gradiant descent let see it in another time
# For Polynomial we will transform features from linear degree == 1 to non linear using transform the features as below

#
# # When i put the degree 3 the program take about five minutes to run
# #when i put it 10 give me error is cant place this number of feature
# #so i will keep it 2
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(training_labels_x)
x_validation_poly = poly.fit_transform(validation_labels_x)
poly_regression = LinearRegression()

poly_regression.fit(x_train_poly, training_label_y)

poly_val_pred = poly_regression.predict(x_validation_poly)

poly_mse = mean_squared_error(validation_label_y, poly_val_pred)
poly_r2 = r2_score(validation_label_y, poly_val_pred)
poly_mae = mean_absolute_error(validation_label_y, poly_val_pred)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(training_labels_x)
x_validation_scaled = scaler.transform(validation_labels_x)
# RBF Kernel with Support Vector Regression (SVR)
rbf_svr = SVR(kernel='rbf', C=1.0, gamma='scale')  # Adjust 'C' and 'gamma' for best results
rbf_svr.fit(x_train_scaled, training_label_y)

# Predictions
validation_pred_rbf = rbf_svr.predict(x_validation_scaled)

# Evaluate Performance
mse_rbf = mean_squared_error(validation_label_y, validation_pred_rbf)
r2_rbf = r2_score(validation_label_y, validation_pred_rbf)
rbf_mae = mean_absolute_error(validation_label_y, validation_pred_rbf)


#For feature selection we will we will do for loop and check each time we add feature
selected_features = []  # Start with this
remaining_features = list(training_labels_x.columns)
model_performance = []  # MSE for each iteration
max_features = len(remaining_features)

last_mse = float("inf")  # Initialize last MSE to infinity

while remaining_features and (len(selected_features) < max_features):
    best_feature = None
    best_mse = float("inf")

    # Test each remaining feature
    for feature in remaining_features:
        current_features = selected_features + [feature]
        model = LinearRegression()
        model.fit(training_labels_x[current_features], training_label_y)
        y_pred = model.predict(validation_labels_x[current_features])
        mse = mean_squared_error(validation_label_y, y_pred)

        if mse < best_mse:
            best_mse = mse
            best_feature = feature

    if best_mse >= last_mse:
        print("No improvement in MSE")
        break

    selected_features.append(best_feature)
    remaining_features.remove(best_feature)
    model_performance.append(best_mse)
    last_mse = best_mse

    print(f"Step {len(selected_features)}: Added feature '{best_feature}' with MSE = {best_mse:.4f}")


model_metrics = [
    {"Model": "Linear Regression", "MSE": lin_mse, "MAE": lin_mae, "R-squared": lin_r2},
    {"Model": "Lasso Regression", "MSE": mse_lasso, "MAE": mae_lasso, "R-squared": r2_lasso},
    {"Model": "Ridge Regression", "MSE": mse_ridge, "MAE": mae_ridge, "R-squared": r2_ridge},
    {"Model": "Closed Form Solution", "MSE": closed_form_mse, "MAE": closed_form_mae, "R-squared": closed_form_r2},
    {"Model": "Polynomial Regression", "MSE": poly_mse, "MAE": poly_mae, "R-squared": poly_r2},
    {"Model": "Radial Basis Function (RBF)", "MSE": mse_rbf, "MAE": rbf_mae, "R-squared": r2_rbf}
]

for metric in model_metrics:
    print("*******************************************************")
    print(f"{metric['Model']} Model Performance")
    print(f"Mean Squared Error: {metric['MSE']:.2f}")
    print(f"R² Score: {metric['MAE']:.2f}")
    print(f"Mean Absolute Error: {metric['R-squared']:.2f}")


print("Here we see the best model performance ")
#So we will test our model on test data set so
x_test_scaled = scaler.fit_transform(test_labels_x)

prediction_y=rbf_svr.predict(x_test_scaled)

