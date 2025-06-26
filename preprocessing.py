import joblib
import numpy as np
import pandas as pd
import os

scaler = joblib.load("model/scaler.pkl")

def preprocess_input(data_dict):
    categorical = [
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
    ]
    numerik = ["BMI", "MentHlth", "PhysHlth", "GenHlth"]

    input_cat = [int(data_dict.get(col, 0)) for col in categorical]
    
    input_num_dict = {col: [float(data_dict.get(col, 0.0))] for col in numerik}
    input_num_df = pd.DataFrame(input_num_dict)
    input_num_scaled = scaler.transform(input_num_df)[0]

    return np.array(input_cat + list(input_num_scaled))
