# Crop Recommendation Model

## Overview

This project contains a machine learning model trained on the **Crop Recommendation Dataset** from Kaggle. The model predicts which crops should be recommended based on environmental factors.

The models used in this project are:

* **Random Forest**: Achieved an accuracy of **96%**.
* **Support Vector Machine (SVM)**: Achieved an accuracy of **97%**.

The SVM model is saved as `model.pkl` and can be used for making predictions.

## Dataset

The dataset used for this project is the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset/data), which contains various environmental factors like temperature, humidity, pH, and rainfall to recommend the best crops for a given set of conditions.

You can download the dataset and place it in the `data/` directory. The dataset consists of features such as:

* **N**: Nitrogen content in the soil.
* **P**: Phosphorous content in the soil.
* **K**: Potassium content in the soil.
* **Temperature**: Temperature of the region.
* **Humidity**: Average humidity of the region.
* **PH**: pH value of the soil.
* **Rainfall**: Rainfall in the region.
* **Label (Crop)**: The recommended crop based on these conditions.

## Project Structure

* `Crop Recommendation.ipynb`: in this we used for training the Random Forest and SVM models.
* `model.pkl`: The saved **SVM model** after training.
* `requirements.txt`: Required libraries for running the code.

## Requirements

The following Python libraries are required for this project:

* `scikit-learn`
* `numpy`
* `pandas`
* `joblib`
* `matplotlib`


## Training the Model

1. Download the **Crop Recommendation Dataset** from [Kaggle](https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset/data) and place it in the `data/` directory.

2. Run the `Crop Recommendation.ipynb` script to train both **Random Forest** and **SVM** models.



This will:

1. Train both models using the dataset.
2. Print the accuracy of each model.
3. Save the **SVM model** as `model.pkl` in the root directory.

Example output:

```bash
Random Forest Model Accuracy: 96%
SVM Model Accuracy: 97%
Model saved as 'model.pkl'
```

## Making Predictions

Once the model is trained and saved, you can use the `model.pkl` file to load the model and make predictions on new data.

### Example usage:

```python
import joblib
import numpy as np

# Load the saved model
model = joblib.load("model.pkl")

# Example input for prediction: Replace with actual feature values
new_data = np.array([[90, 50, 40, 30, 80, 6.5, 200]])  # Example values for N, P, K, Temperature, Humidity, pH, Rainfall
predictions = model.predict(new_data)

print("Recommended Crop:", predictions)
```

This will output the recommended crop based on the input environmental factors.

## Saving the Model

After training, the model is saved using `joblib`:

```python
import joblib
joblib.dump(model, "model.pkl")
```

This allows you to easily load the model later for predictions.
