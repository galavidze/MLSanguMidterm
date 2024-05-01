import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load the data
data = pd.read_csv('spam-data.csv')

# Split data into features (X) and target labels (y)
X = data.drop(columns=['Class'])
y = data['Class']

# Divide data into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model and print confusion matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
