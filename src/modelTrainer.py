from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib


def train_model():

    print("Training the model...\n")
    
    # Function to remove outliers.
    def remove_outliers(data, column):
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    # Loading the data into DataFrames.
    red_wine_data = pd.read_csv(r'data\winequality-red.csv', delimiter=';')
    white_wine_data = pd.read_csv(r'data\winequality-white.csv', delimiter=';')

    # Adding a column to differentiate between red and white wines.
    red_wine_data['type'] = 'red'
    white_wine_data['type'] = 'white'

    # Combining the two DataFrames into one.
    combined_data = pd.concat([red_wine_data, white_wine_data], axis=0)
    combined_data.reset_index(drop=True, inplace=True)
    wine_data = combined_data.copy()

    # One-hot encoding the 'type' column.
    wine_data = pd.concat([wine_data, pd.get_dummies(wine_data['type'])], axis=1)
    wine_data = wine_data.drop('type', axis=1)

    # Adding a boolean flag to differentiate between high and low quality wines.
    wine_data.loc[:, "high quality flag"] = wine_data["quality"].apply(lambda x: 1 if x > 5 else 0)

    # Dropping the original quality column to prevent data leakage and overfitting.
    wine_data = wine_data.drop('quality', axis=1)

    # Removing outliers from the dataset.
    for column in ['residual sugar', 'free sulfur dioxide', 'total sulfur dioxide']:
        wine_data = remove_outliers(wine_data, column)

    # Splitting the data into features and target.
    X = wine_data.drop(['high quality flag'], axis=1)
    y = wine_data['high quality flag']

    # Splitting the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69420)

    # Scaling the data.
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Training the model with the optimal hyperparameters determined in the Jupyter Notebook.
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=69420
    )

    # Fitting the model to the data.
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = round(accuracy_score(y_test, y_pred), 4)
    print(f"Model succesfully trained, accuracy: {accuracy}\n")

    # Saving the scaler and model to be reused in the main program.
    joblib.dump(scaler, r'src\joblibs\scaler.pkl')
    joblib.dump(model, r'src\joblibs\model.pkl')