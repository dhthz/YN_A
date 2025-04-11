import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import keras
import tensorflow 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.src.layers import Dense, Input
from keras.src.callbacks import EarlyStopping

file = "alzheimers_disease_data.csv"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable GPU , i dont have one but if you do you can remove this line

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
        
        # Ορισμός αριθμού εισόδων (I) και στόχου
        I = dataForTargetColumn.shape[1]  # Αριθμός εισόδων

        # Ορισμός αριθμών νευρώνων για τα κρυφά επίπεδα
        hidden_layer_configs = [I//2, 2*I//3, I, 2*I]
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        input_features = dataForTargetColumn.shape[1]  # Number of input features
        results = {}

        # Επανάληψη για κάθε αριθμό κρυφών κόμβων
        for hidden_units in hidden_layer_configs:
            print(f"Training with {hidden_units} hidden neurons:")
            fold_accuracies = []
            fold_losses = []
            val_accuracies = []

            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            for train_idx, val_idx in skf.split(dataForTargetColumn, targetColumn):
                X_train, X_val = dataForTargetColumn[train_idx], dataForTargetColumn[val_idx]
                y_train, y_val = targetColumn[train_idx], targetColumn[val_idx]

                model = Sequential([
                    Input(shape=(input_features,)),  # Correct way to specify input shape
                    Dense(19, activation='LeakyReLU'),    # Hidden layer with 19 neurons
                    Dense(1, activation='sigmoid')   # Output layer
                ])

                # Compile and fit the model
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                
                # Εκπαίδευση του μοντέλου με early stopping
                history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])

                # Καταγραφή της ακριβείας και του κόστους (loss) στο validation set
                val_accuracies.append(history.history['val_accuracy'][-1])
                fold_accuracies.append(history.history['accuracy'][-1])
                fold_losses.append(history.history['loss'][-1])

            # Αποθήκευση των αποτελεσμάτων για κάθε αριθμό κρυφών κόμβων
            results[hidden_units] = {
                'accuracy': np.mean(fold_accuracies),
                'loss': np.mean(fold_losses),
                'val_accuracy': np.mean(val_accuracies)
            }

            # Γραφικές παραστάσεις σύγκλισης (M.O. ανά κύκλο εκπαίδευσης)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title(f"Model Accuracy for {hidden_units} hidden neurons")
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

        # Εμφάνιση των αποτελεσμάτων
        print("Results for each hidden layer configuration:")
        for hidden_units, metrics in results.items():
            print(f"Hidden Neurons: {hidden_units} -> Accuracy: {metrics['accuracy']:.4f}, Loss: {metrics['loss']:.4f}")
        
    
    except FileNotFoundError:
        print(f"❌ Error: File '{file}' not found.")
    except PermissionError:
        print(f"❌ Error: Not permitted to use '{file}'.")
    except UnicodeDecodeError:
        print(f"❌ Error: Encoding error, please try again!")
    except Exception as e:
        print(f"❌ An error occured: {e}")