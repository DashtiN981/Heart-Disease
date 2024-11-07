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

from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Fit SMOTE to the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check the distribution of the target variable after resampling
print("Original training set size:", y_train.value_counts())
print("Resampled training set size:", y_train_resampled.value_counts())

"""from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'random_state': [42]
}

# Initialize Random Forest model
rf_model = RandomForestClassifier()

# Initialize Grid Search
grid_search = GridSearchCV(rf_model, param_grid, cv=5)

# Fit Grid Search to the resampled training data
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters found:", best_params)

# Train the best model on the resampled data
best_rf_model = grid_search.best_estimator_

# Make predictions on the original test set
y_pred = best_rf_model.predict(X_test)

# Evaluate the best Random Forest model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
