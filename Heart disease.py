import pandas as pd

# Define the column names based on the dataset documentation
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", 
    "ca", "thal", "target"
]

# Load the dataset into a DataFrame
df = pd.read_csv("heart_disease.csv", header=None, names=column_names)

# Preview the data
print(df.head())

# Fill missing values in 'ca' and 'thal' with their respective medians
df['ca'].fillna(df['ca'].median(), inplace=True)
df['thal'].fillna(df['thal'].median(), inplace=True)

# Check for any remaining NaNs
missing_values = df[['ca', 'thal']].isnull().sum()
print(missing_values)