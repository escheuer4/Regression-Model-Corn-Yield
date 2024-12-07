import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder

# NASS API key
api_key = 'AA42421A-F6FE-3754-A80A-9A6CD8A9EB23'

# URL for the API
base_url = 'https://quickstats.nass.usda.gov/api/api_GET/'

# Query parameters
params = {
    'key': api_key,
    'commodity_desc': 'CORN', 
    'year__GE': '2019',        
    'state_name': 'OHIO',     
    'statisticcat_desc' : 'YIELD',
    'format': 'CSV'    
}

# Make the API request
response = requests.get(base_url, params=params)

# Check if the request was successful
if response.status_code == 200:
    with open('nass_data.csv', 'w') as file:
        file.write(response.text)

    # Load the data
    data = pd.read_csv('nass_data.csv')
    print("Raw data shape:", data.shape)
    print("Unique statistic categories:", data["statisticcat_desc"].unique())

    # Filter for yield data
    yield_data = data[data["statisticcat_desc"].str.contains("YIELD", case=False, na=False)]
    print("Filtered data shape:", yield_data.shape)
else:
    print(f"Error: {response.status_code}")


yield_data = yield_data.dropna(subset=["Value"])

target = yield_data["Value"]

categorical_columns = [
    "commodity_desc",
    "class_desc",
    "prodn_practice_desc",
    "util_practice_desc",
    "state_name",
    "county_name",
    "unit_desc",
]

# Fill missing values in categorical columns with a placeholder
yield_data[categorical_columns] = yield_data[categorical_columns].fillna("Unknown")

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_features = encoder.fit_transform(yield_data[categorical_columns])

# Convert to DataFrame
encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

numeric_features = yield_data[["year"]]
features = pd.concat([numeric_features.reset_index(drop=True), encoded_features_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"R^2 Score: {score:.2f}")

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
print(results.head())
results.to_csv("predictions_vs_actual.csv", index=False)