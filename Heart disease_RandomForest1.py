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

# Check for missing values
print(df.isnull().sum())


# Replace any non-numeric values with NaN
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')

#print(df.isnull().sum())

# Now fill missing values in 'ca' and 'thal' with their respective medians
df['ca'].fillna(df['ca'].median(), inplace=True)
df['thal'].fillna(df['thal'].median(), inplace=True)

# Check for any remaining NaNs
missing_values = df[['ca', 'thal']].isnull().sum()
print(missing_values)


# View summary statistics
print(df.describe())

# Check target distribution
print(df['target'].value_counts())

import matplotlib.pyplot as plt

# Plot distributions for each feature
df.hist(bins=20, figsize=(15, 10))
#plt.show()

import seaborn as sns

# Display correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
#plt.show()

# Set a correlation threshold
correlation_threshold = 0.2

# Identify features that have a strong correlation with the target variable
correlated_features = df.corr()['target'][df.corr()['target'].abs() > correlation_threshold].index.tolist()

# Print the selected features
print("Selected features based on correlation with target:")
print(correlated_features)

from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = df[correlated_features[:-1]]  # All features except 'target'
y = df['target']  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the resulting datasets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_rf_pred = rf_model.predict(X_test)

# Evaluate the Random Forest model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_rf_pred))

"""from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Create a logistic regression model
model = LogisticRegression(solver='liblinear', max_iter=2000)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))"""


