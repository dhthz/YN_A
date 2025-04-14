import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import keras
import tensorflow 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.src.layers import Dense, Input, LeakyReLU
from keras.src.callbacks import EarlyStopping

file = "alzheimers_disease_data.csv"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU, you can remove this line if you have one

# Create the 'plots' directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

if not os.path.exists(file):
    print(f"File {file} not found")
    exit()
else:
    try:
        # Load dataset
        df = pd.read_csv(file, encoding="utf-8")
        print("File loaded successfully")

        # Apply Normalization and Standardization
        columns_for_normalization = ['AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality','ADL']
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[columns_for_normalization] = scaler.fit_transform(df[columns_for_normalization])

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

        # Define features and target variable
        dataForTargetColumn = df.drop(columns=['PatientID','Diagnosis','DoctorInCharge'])
        targetColumn = df['Diagnosis']
        dataForTargetColumn = dataForTargetColumn.to_numpy()
        targetColumn = targetColumn.to_numpy()

        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        # Hyperparameters
        I = dataForTargetColumn.shape[1]  # Number of input features
        hidden_layer_configs = [I//2, 2*I//3, I, 2*I]
        input_activation_function_list = ['relu', 'tanh', 'silu']
        loss_function_list = ['binary_crossentropy', 'mean_squared_error']

        # Store results
        results = {}

        for loss_fn in loss_function_list:
            for input_activation_fn in input_activation_function_list:
                for hidden_units in hidden_layer_configs:
                    print(f"Training with {hidden_units} hidden neurons:")

                    # Initialize lists to store metrics
                    fold_losses = []
                    fold_accuracies = []
                    val_accuracies = []
                    fold_bces = []
                    fold_mces = []
                    train_loss_epochs = []
                    val_loss_epochs = []

                    # Early stopping
                    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                    # Cross-validation
                    for train_idx, val_idx in skf.split(dataForTargetColumn, targetColumn):
                        X_train, X_val = dataForTargetColumn[train_idx], dataForTargetColumn[val_idx]
                        y_train, y_val = targetColumn[train_idx], targetColumn[val_idx]

                        y_train = y_train.reshape(-1, 1)
                        y_val = y_val.reshape(-1, 1)
                        
                        # Define model
                        model = Sequential([
                            Input(shape=(I,)),
                            Dense(hidden_units, activation=input_activation_fn),
                            Dense(1, activation='sigmoid')
                        ])
                        # Compile and fit the model
                        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy', 'binary_crossentropy', 'mean_squared_error'])
                        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])

                        # Get metrics
                        train_metrics = model.evaluate(X_train, y_train, verbose=0)
                        val_metrics = model.evaluate(X_val, y_val, verbose=0)

                        # Store metrics
                        fold_losses.append(train_metrics[0])
                        fold_accuracies.append(train_metrics[1])
                        fold_bces.append(train_metrics[2])
                        fold_mces.append(train_metrics[3])
                        val_accuracies.append(val_metrics[1])

                        # Collect epoch-wise loss data
                        train_loss_epochs.append(history.history['loss'])
                        val_loss_epochs.append(history.history['val_loss'])

                    # Calculate average metrics
                    avg_train_loss = np.mean(fold_losses)
                    avg_train_accuracy = np.mean(fold_accuracies)
                    avg_train_bce = np.mean(fold_bces)
                    avg_train_mse = np.mean(fold_mces)
                    avg_val_accuracy = np.mean(val_accuracies)

                    # Save results
                    results[(loss_fn, input_activation_fn, hidden_units)] = {
                        'Train Accuracy': avg_train_accuracy,
                        'Train Loss': avg_train_loss,
                        'Train BCE': avg_train_bce,
                        'Train MSE': avg_train_mse,
                        'Validation Accuracy': avg_val_accuracy,
                    }

                    # Ensure all epochs have the same length
                    min_epochs = min(len(hist) for hist in train_loss_epochs)
                    train_loss_epochs = [hist[:min_epochs] for hist in train_loss_epochs]
                    val_loss_epochs = [hist[:min_epochs] for hist in val_loss_epochs]

                    # Average the losses over folds for plotting
                    avg_train_loss_epochs = np.mean(train_loss_epochs, axis=0)
                    avg_val_loss_epochs = np.mean(val_loss_epochs, axis=0)

                    # Plot loss convergence
                    plt.figure(figsize=(8, 6))
                    plt.plot(avg_train_loss_epochs, label='Avg Training Loss')
                    plt.plot(avg_val_loss_epochs, label='Avg Validation Loss')
                    plt.title(f"Loss Convergence\nLoss: {loss_fn}, Input: {input_activation_fn}, Hidden: {hidden_units}")
                    plt.xlabel("Epochs")
                    plt.ylabel("Loss")
                    plt.legend()

                    # Save plot with a descriptive filename
                    plot_filename = f"plots/{loss_fn}_{input_activation_fn}_{hidden_units}_neurons.png"
                    plt.savefig(plot_filename)  # Save the plot as PNG
                    plt.close()  # Close the plot to prevent memory issues

                    print(f"Configuration {hidden_units} neurons, Loss Function: {loss_fn}, Activation: {input_activation_fn}")
                    print(f"Training converges at epoch: {np.argmin(avg_train_loss_epochs)}")
                    print(f"Validation converges at epoch: {np.argmin(avg_val_loss_epochs)}")

        # Prepare results for saving
        results_list = []
        for (loss_fn, input_activation_fn, hidden_units), metrics in results.items():
            results_list.append({
                'Loss Function': loss_fn,
                'Input Activation Function': input_activation_fn,
                'Hidden Neurons': hidden_units,
                **metrics
            })

        # Create DataFrame
        results_df = pd.DataFrame(results_list)

        # Sort by Validation Accuracy
        results_df = results_df.sort_values(by='Validation Accuracy', ascending=False)

        # Save results to CSV
        results_df.to_csv("model_comparison_results.csv", index=False)

        print("✅ Results saved to model_comparison_results.csv")
        print(results_df.head())

    except FileNotFoundError:
        print(f"❌ Error: File '{file}' not found.")
    except PermissionError:
        print(f"❌ Error: Not permitted to use '{file}'.")
    except UnicodeDecodeError:
        print(f"❌ Error: Encoding error, please try again!")
    except Exception as e:
        print(f"❌ An error occured: {e}")
