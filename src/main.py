from dataGenerator import generate_synthetic_data_point
from modelTrainer import train_model
import pandas as pd
import joblib

# All of the features from the wine-dataset, with their respective ranges based on min- and max-values from the original UC Irvine-data.
columnsAndRanges = {
    'fixed acidity': (3.8, 15.9),
    'volatile acidity': (0.08, 1.58),
    'citric acid': (0.0, 1.66),
    'residual sugar': (0.6, 65.8),
    'chlorides': (0.009, 0.611),
    'free sulfur dioxide': (1.0, 72.0),
    'total sulfur dioxide': (6.0, 289.0),
    'density': (0.98711, 1.03898),
    'pH': (2.72, 4.01),
    'sulphates': (0.22, 2.0),
    'alcohol': (8.0, 14.2),
}

# Initializing the input features dictionary.
inputFeatures = {feature: None for feature in columnsAndRanges.keys()}
inputFeatures['red'] = None
inputFeatures['white'] = None

UI = """

--------------------------------------------------------------------------------------------------
|                                                                                                |                      
|   ##     ##                                                                                    |            
|   ##~~~~~##                                                                                    |                           
|   ##~~~~~##                                                                                    |                       
|    ##~~~##                                                                                     |
|     ## ##                                                                                      |
|      ##   Welcome to the wine quality classifier!                                              |
|      ##   This program will classify the quality of a wine based on features input by the user.|
|      ##   Each feature can also be randomized.                                                 | 
|      ##                                                                                        |
|      ##                                                                                        |
|     #####                                                                                      |
|   #########                                                                                    |
|                                                                                                |
--------------------------------------------------------------------------------------------------

"""

def get_user_input():

    while True:
        wine_type = input("Begin by determining the type of wine you would like to predict ('red' or 'white'):\n")
        if wine_type in ['red', 'white']:
            break
        else:
            print("Invalid input. Please enter 'red' or 'white'.")

    if wine_type == 'red':
        inputFeatures['red'] = 1
        inputFeatures['white'] = 0
    else:
        inputFeatures['red'] = 0
        inputFeatures['white'] = 1

    print("\nNow it is time to start entering the values for the wine features.\n")
    for column, range in columnsAndRanges.items():
        while True:
            user_input = input(f"Enter value for {column} between {range[0]} and {range[1]}, or press ENTER to randomize.\n")
            if user_input == "":
                inputFeatures[column] = generate_synthetic_data_point(column)
                print(f"The randomized value for {column} is {inputFeatures[column]}.\n")
                break
            else:
                try:
                    input_value = float(user_input)
                    if range[0] <= input_value <= range[1]:
                        inputFeatures[column] = input_value
                        break
                    else:
                        print(f"ERROR: The entered value is out of the allowed range ({range[0]} - {range[1]}).\n")
                except ValueError:
                    print("ERROR: The entered value is not a valid floating number.\n")

def train_and_load_model():

    train_model()

    # Loading the scaler, and model.
    scaler = joblib.load(r'src\joblibs\scaler.pkl')
    model = joblib.load( r'src\joblibs\model.pkl')
    
    return scaler, model

def predict_wine_quality(scaler, model):

    # Converting the inputFeatures to a DataFrame.
    input_features_df = pd.DataFrame([inputFeatures])

    # Scaling the data.
    scaled_features = scaler.transform(input_features_df)

    # Predicting the quality of the wine.
    prediction = model.predict(scaled_features)
    if prediction[0] == 1:
        print("Based on the input features, this wine can be considered HIGH-quality, with a confidence of around 84%.")
    else:    
        print("Based on the input features, this wine can be considered LOW-quality, with a confidence of around around 84%.")

def main():

    print(UI)
    get_user_input()
    scaler, model = train_and_load_model()
    predict_wine_quality(scaler, model)

if __name__ == "__main__":
    
    main()