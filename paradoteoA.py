import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plot 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold


file = "alzheimers_disease_data.csv"

if not os.path.exists(file):
    print(f"File {file} not found")
    exit()
else:
    try:
        df = pd.read_csv(file, encoding="utf-8")
        print("File loaded successfully")
        
        #print(df.head())
        #print(df.info())
        #print(df.describe())
        # TODO Make this into a function
        # Apply Normalization
        columns_for_normalization = ['AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality','ADL']
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[columns_for_normalization] = scaler.fit_transform(df[columns_for_normalization])

        # Apply Standardization
        columns_for_standardization = [
            'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
            'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 
            'MMSE', 'FunctionalAssessment'
        ]
        scaler_standard = StandardScaler()
        df[columns_for_standardization] = scaler_standard.fit_transform(df[columns_for_standardization])

        # Apply One-Hot Encoding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        df["Ethnicity"] = df["Ethnicity"].astype("category")
        df["EducationLevel"] = df["EducationLevel"].astype("category")
        df = pd.get_dummies(df, columns=["Ethnicity"], prefix="Ethnicity", dtype=int)
        df = pd.get_dummies(df, columns=["EducationLevel"], prefix="EducationLevel", dtype=int)

        # Save the processed dataset
        df.to_csv("processed_data.csv", index=False)
    
        # Define features and target variable
        dataForTargetColumn = df.drop(columns=['Diagnosis','DoctorInCharge'])  # These are the variables used to predict the target
        targetColumn = df['Diagnosis']  # This is the target 
        
        # Convert to numpy arrays if necessary
        dataForTargetColumn = dataForTargetColumn.to_numpy()
        targetColumn = targetColumn.to_numpy()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        python -c "import tensorflow as tf; print(tf.__version__)"
        
        
    
    except FileNotFoundError:
        print(f"❌ Error: File '{file}' not found.")
    except PermissionError:
        print(f"❌ Error: Not permitted to use '{file}'.")
    except UnicodeDecodeError:
        print(f"❌ Error: Encoding error, please try again!")
    except Exception as e:
        print(f"❌ An error occured: {e}")