import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv('Dataset.csv', sep=';')

# Pollutants to predict
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# Convert 'date' to datetime and extract year
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
df['year'] = df['date'].dt.year

# Drop rows with missing pollutant values
df = df.dropna(subset=pollutants)

# Features and target
X = df[['id', 'year']]
y = df[pollutants]

# One-hot encode 'id'
X_encoded = pd.get_dummies(X, columns=['id'], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Train the model
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Save the model and columns
joblib.dump(model, 'pollution_model.pkl')
joblib.dump(X_encoded.columns.tolist(), 'model_columns.pkl')

print("Model training complete and files saved: pollution_model.pkl, model_columns.pkl")
