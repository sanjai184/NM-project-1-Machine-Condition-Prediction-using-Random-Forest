````markdown
# Machine Condition Prediction Using Random Forest

**Name:** Sanjai kumar G 
**Year:** II Year  
**Department:** Mechanical Engineering  
**Course:** Data Analysis in Mechanical Engineering  
**College:** ARM College of Engineering & Technology  

---

## Project Overview

This project is focused on predicting the condition of a machine using machine learning. I have used a **Random Forest Classifier** to analyze input data such as temperature, vibration level, oil quality, RPM, and other mechanical parameters to determine whether a machine is functioning normally or has potential faults.

This type of predictive system can help in maintenance planning and early fault detection in mechanical systems.

---

## Required Libraries

Before running the code, make sure to install the required Python libraries. Use the following command:

```bash
pip install -r requirements.txt
````

This will install everything needed to run the prediction script.

---

## Important Files for Prediction

These files are needed to make the prediction work correctly:

* `random_forest_model.pkl` – The trained machine learning model.
* `scaler.pkl` – Used to normalize input data the same way it was done during training.
* `selected_features.pkl` – A list of the features used in the training process, which ensures correct column ordering.

All these files must be in the same directory as your script.

---

## How the Prediction Process Works

1. **Loading the Tools**

   * Load the model with `joblib.load('random_forest_model.pkl')`.
   * Load the scaler with `joblib.load('scaler.pkl')`.
   * Load the list of selected features with `joblib.load('selected_features.pkl')`.

2. **Input Preparation**

   * Create a single-row `DataFrame` using pandas with values for all the required features.
   * Make sure the column names are exactly the same as those in the training data.

3. **Data Scaling**

   * Use the loaded scaler to transform the input data so that it matches the model’s expectations.

4. **Making a Prediction**

   * Use `.predict()` to get the final output class.
   * Use `.predict_proba()` to see the probability scores for each possible class.

---

## How to Use the Prediction Script

Here is a sample code you can use in a Python file (like `predict.py`):

```python
import joblib
import pandas as pd

# Load model, scaler, and selected features
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Example input data
new_data = pd.DataFrame([{
    'Temperature_C': 75,
    'Vibration_mm_s': 2.5,
    'Oil_Quality_Index': 88,
    'RPM': 1500,
    'Pressure_bar': 5.2,
    'Shaft_Misalignment_deg': 0.3,
    'Noise_dB': 70,
    'Load_%': 85,
    'Power_kW': 12.5
}])

# Arrange features in correct order
new_data = new_data[selected_features]

# Apply scaling
scaled_data = scaler.transform(new_data)

# Predict machine condition
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Condition:", prediction[0])
print("Prediction Confidence:", prediction_proba[0])
```

---

## Things to Keep in Mind

* Your input must contain **all the same features** used in training.
* The values should be realistic and within the usual operating range.
* The order of features in the data matters — do not change the column sequence.

---

## Future Scope: Retraining the Model

If needed, the model can be retrained with new data. For that:

* Follow the same steps for preprocessing.
* Use consistent feature selection and scaling.
* Save the new model and tools using `joblib`.

---

## Practical Applications

* Predict whether a machine is operating **normally** or showing **faulty behavior**.
* Useful in industrial setups, maintenance departments, and smart monitoring systems using IoT.

---

This project helped me understand how machine learning can be applied to real mechanical systems to make smarter maintenance decisions.
