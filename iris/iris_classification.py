import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download dataset
path = kagglehub.dataset_download("saurabh00007/iriscsv")

print("Dataset path:", path)

# Load CSV file
df = pd.read_csv(path + "/Iris.csv")

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Remove Id column
df = df.drop("Id", axis=1)

# Data Visualization
sns.pairplot(df, hue="Species")
plt.show()

# Features and Target
X = df.drop("Species", axis=1)
y = df["Species"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Custom Prediction
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(sample)

print("\nPredicted Flower:", prediction[0])