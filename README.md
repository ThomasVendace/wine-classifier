
# Wine Quality Classifier

In essence, this project is a binary classifier, that predicts whether a wine sample is of **high quality** or **low quality** based on its physicochemical properties. The project uses machine learning techniques to train a Random Forest model and provides a command-line interface for inputting wine features and obtaining predictions.

---

## Features

- **Trainable Machine Learning Model**: Trains a Random Forest Classifier using wine quality datasets from UCI.
- **Feature Randomization**: Allows the user to randomize feature inputs.
- **Classification**: Classifies the wine as high or low quality based on input features.
- **Synthetic Data Generation**: Generates synthetic feature data points for randomization.

---

## Prerequisites

Ensure you have the following installed on your system:

1. Python (>=3.8)
2. Pip (latest version)
3. Git (optional, for cloning the repository)

---

## Installation and Setup

### 1. Clone the Repository
Clone the project from the GitHub repository (if hosted online):
```bash
git clone <repository-url>
cd WineQualityClassifier
```

Alternatively, if you have the folder locally, navigate to it.

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Prepare the Data
Ensure the following files are present in the `data/` directory:
- `winequality-red.csv` (Red wine data)
- `winequality-white.csv` (White wine data)

### 4. Train the Model
Run the `modelTrainer.py` script to train the model:
```bash
python src/modelTrainer.py
```

This script:
- Combines the red and white wine datasets.
- Removes outliers and scales the data.
- Trains a Random Forest model.
- Saves the trained model and scaler to the `src/joblibs/` directory.

### 5. Run the Classifier
Execute the `main.py` script to start the classifier:
```bash
python src/main.py
```

This script:
1. Presents an interactive menu.
2. Accepts user inputs for wine features (or allows randomization).
3. Loads the trained model and scaler.
4. Predicts whether the wine is of high or low quality.

---

## Usage

### Interactive Features
1. The program will prompt you to select the type of wine (`red` or `white`).
2. Enter values for each feature manually or press `Enter` to randomize.
3. The classifier will output whether the wine is **high quality** or **low quality**.

---

## File Descriptions

### `modelTrainer.py`
- Loads and preprocesses the wine datasets.
- Trains a Random Forest Classifier.
- Saves the scaler and model for later use.

### `main.py`
- Provides a user-friendly CLI interface.
- Accepts feature inputs or randomizes them.
- Uses the trained model to classify the wine.

### `dataGenerator.py`
- Generates synthetic data points for feature randomization based on dataset statistics.

---

## Example

### Input:
- `Type`: Red
- `Fixed Acidity`: 7.4
- `Volatile Acidity`: Randomized
- Other features: Randomized

### Output:
```plaintext
Based on the input features, this wine can be considered HIGH-quality, with a confidence of around 84%.
```

---

## Notes

- **Data Source**: The dataset is sourced from the UCI Machine Learning Repository.
- **Dependencies**: The `requirements.txt` includes necessary packages such as `pandas`, `numpy`, `scikit-learn`, and `joblib`.

---

Hopefully this program will guide you on your way to becoming a wine connoisseur! 🍷
