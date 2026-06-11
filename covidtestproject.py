import mysql.connector
import pandas as pd

# Database Connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="ROOT",
    database="covid_db"
)

print("Connected Successfully!")

# Load Data
df = pd.read_sql("SELECT * FROM covid_data", conn)

print(df.head())
print("Shape:", df.shape)

# Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df["gender"] = le.fit_transform(df["gender"])
df["cough"] = le.fit_transform(df["cough"])
df["city"] = le.fit_transform(df["city"])
df["has_covid"] = le.fit_transform(df["has_covid"])

# # Features and Target
X = df.drop("has_covid", axis=1)
y = df["has_covid"]

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

# # Train Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Prediction
y_pred = model.predict(X_test)

# # Accuracy
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))

#Save Model
import pickle

pickle.dump(model, open("covid_model.pkl", "wb"))

print("Model Saved Successfully")