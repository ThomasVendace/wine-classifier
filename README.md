
# Wine Quality Classifier

In essence, this project is a binary classifier, that predicts whether a wine sample is of **high quality** or **low quality** based on its physicochemical properties. The project uses machine learning techniques to train a Random Forest model and provides a command-line interface for inputting wine features and obtaining predictions.

The wineQualityClassifier.ipynb-file, in the notebooks-folder has a step by step walkthrough of the model training, and data wrangling process.

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
3. Git

---

## Installation and Setup

### 1. Clone the project repository (this part can be skipped, if you already have the project directory on your machine)
Clone the project from the GitHub repository, by running the following script in your terminal, after navigating to the directory, that you want to clone the project into.
```bash
git clone https://github.com/ThomasVendace/wine-classifier.git
```

### 2. Navigate to project directory on your machine, and open your terminal.

### 3. Install dependencies
Install the required Python packages, by running the following in your terminal:
```bash
pip install -r requirements.txt
```

### 4. Prepare the data
Ensure the following files are present in the `data/` directory:
- `winequality-red.csv` (Red wine data)
- `winequality-white.csv` (White wine data)


### 5. Run the program
Execute the `main.py` script to start the classifier:
```bash
python src/main.py
```

This program:
1. Presents an interactive menu.
2. Accepts user inputs for wine features (or allows randomization).
3. Loads the trained model and scaler.
4. Predicts whether the wine is of high or low quality.

---

## Usage

### Interactive features
1. The program will prompt you to select the type of wine (`red` or `white`).
2. Enter values for each feature manually or press `Enter` to randomize.
3. The classifier will output whether the wine is **high quality** or **low quality**.

---

## File descriptions

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
