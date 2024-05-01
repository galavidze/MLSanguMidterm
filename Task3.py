#Lashagiorgi Alavidze
#Overall - the output is 0, meaning that the model predicted it not to be spam.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
data = pd.read_csv('spam-data.csv')

# Split data into features (X) and target labels (y)
X = data.drop(columns=['Class'])
y = data['Class']

# Build and train logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Load email content from emails.txt
with open('emails.txt', 'r') as file:
    email_content = file.readline()

# Extract email features using CountVectorizer
vectorizer = CountVectorizer(vocabulary=X.columns)
email_features = vectorizer.fit_transform([email_content])

# Check email for spam
prediction = model.predict(email_features)
if prediction[0] == 1:
    print("The first email is predicted to be spam.")
else:
    print("The first email is predicted not to be spam.")
