# Favorite Transport Prediction Model

This project uses a machine learning model to predict a person's favorite mode of transport based on their age, gender, and income. The model is built using a Decision Tree Classifier and trained on a dataset of vehicle preferences.

## Features
- Predict a user's favorite transport type based on their inputs.
- Handles missing data and encodes categorical variables.
- Provides an interactive interface for user input.
- Offers error handling for invalid gender entries.

## Dataset
The project uses a dataset named `vehicles.csv`, which contains the following columns:
- **Age**: Age of the individual.
- **Gender**: Gender of the individual.
- **Income**: Income of the individual.
- **Favorite Transport**: Target column indicating the preferred mode of transport.

## Requirements
The project requires the following Python libraries:
- `pandas`
- `scikit-learn`

To install the required libraries, run:
```bash
pip install pandas scikit-learn
