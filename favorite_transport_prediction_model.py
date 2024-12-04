"""
Favorite Transport Prediction Model

This project uses a decision tree classifier to predict a person's favorite mode of transport
based on their age, gender, and income. The dataset is preprocessed to handle missing values
and encode categorical variables. The user can input their information, and the model will
provide a prediction of their favorite transport type.
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('vehicles.csv')  # Load vehicle-related data from a CSV file.

# Fill missing values in the dataset
df['Income'] = df['Income'].fillna(0.0)  # Replace missing income values with 0.0.
df['Gender'] = df['Gender'].fillna('Unknown')  # Replace missing gender values with 'Unknown'.

# Standardize the Gender column
df['Gender'] = df['Gender'].str.lower()  # Convert gender values to lowercase for consistency.

# Encode the Gender column
label_encoder = LabelEncoder()  # Initialize a label encoder to convert categorical data to numerical.
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Encode gender categories to numerical values.

# Separate features (X) and target (y)
X = df.drop(columns=['Favorite Transport'])  # Features for training (exclude the target column).
y = df['Favorite Transport']  # Target variable representing favorite transport.

# Train the Decision Tree Classifier
model = DecisionTreeClassifier()  # Initialize the decision tree classifier.
model.fit(X, y)  # Train the classifier on the dataset.

# Welcome message and instructions for the user
print("Welcome to the Favorite Transport Prediction Model!")
print("Here you can input the Age, Gender, and Income of a person, and prediction of their favorite transport will be provided!")
print()

# Gather user input for prediction
age = int(input("Enter Age: "))  # Accept age as an integer input.
gender = input("Enter Gender (e.g., Male, Female, Unknown): ").strip().lower()  # Normalize user input to lowercase.
income = float(input("Enter Income: "))  # Accept income as a float input.

# Encode the Gender input using the trained LabelEncoder
try:
    encoded_gender = label_encoder.transform([gender])[0]  # Encode the user-provided gender input.
except ValueError:
    # Handle error if the entered gender is not recognized
    print(f"Error: Gender '{gender}' is not recognized. Please use a valid gender from the dataset.")
    print(f"Available genders: {label_encoder.classes_}")  # Display available gender options.
    exit()  # Exit the program gracefully if the input is invalid.

# Create a test DataFrame for the user's input
test_df = pd.DataFrame({'Age': [age], 'Gender': [encoded_gender], 'Income': [income]})  # Create input in model's format.

# Predict the favorite transport based on the user's input
p = model.predict(test_df)  # Make a prediction using the trained model.

# Display the prediction result
print(f"The predicted favorite transport is: {p[0]}")  # Output the predicted favorite transport.
