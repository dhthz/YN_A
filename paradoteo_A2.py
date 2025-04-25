# Import required libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.src.layers import Dense, Input
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import SGD


def plot_convergence(histories, loss_fn, input_activation_fn, output_activation_fn, hidden_units):
    # Calculate average losses
    min_epochs = min(len(h.history['loss']) for h in histories)
    train_losses = np.array([h.history['loss'][:min_epochs]
                            for h in histories])
    val_losses = np.array([h.history['val_loss'][:min_epochs]
                          for h in histories])

    avg_train_loss = np.mean(train_losses, axis=0)
    avg_val_loss = np.mean(val_losses, axis=0)

    # Find convergence
    def find_convergence(losses):
        window = 5
        min_epochs = 20
        rel_threshold = 0.005

        if len(losses) < min_epochs + window:
            return len(losses) - 1

        # Add minimum loss check
        min_loss_idx = np.argmin(losses[min_epochs:]) + min_epochs

        for i in range(min_epochs, len(losses) - window):
            window_vals = losses[i:i+window]
            rel_change = np.std(window_vals) / np.mean(window_vals)

            if rel_change < rel_threshold and i >= min_loss_idx:
                return i

        return len(losses) - 1

    train_epoch = find_convergence(avg_train_loss)
    val_epoch = find_convergence(avg_val_loss)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(avg_train_loss, label='Training Loss')
    plt.plot(avg_val_loss, label='Validation Loss')
    plt.axvline(x=train_epoch, color='b', linestyle='--',
                label='Train Convergence')
    plt.axvline(x=val_epoch, color='r', linestyle='--',
                label='Val Convergence')
    plt.title(
        f"Loss: {loss_fn}, Input Activation: {input_activation_fn}, Output Activation: {output_activation_fn}, Neurons: {hidden_units}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        f"plots/{loss_fn}_{input_activation_fn}_{hidden_units}_neurons_{output_activation_fn}.png")
    plt.close()

    return train_epoch, val_epoch


def main():
    FILE = "alzheimers_disease_data.csv"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # If you have a GPU uncomment this

    # Create plots directory
    if not os.path.exists('plots'):
        os.makedirs('plots')

    try:
        # Load and preprocess data
        df = pd.read_csv(FILE, encoding="utf-8")
        print("File loaded successfully")

        # Apply Normalization
        columns_for_normalization = ['AlcoholConsumption', 'PhysicalActivity',
                                     'DietQuality', 'SleepQuality', 'ADL']
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[columns_for_normalization] = scaler.fit_transform(
            df[columns_for_normalization])

        # Apply Standardization
        columns_for_standardization = ['Age', 'BMI', 'SystolicBP', 'DiastolicBP',
                                       'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
                                       'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment']
        scaler_standard = StandardScaler()
        df[columns_for_standardization] = scaler_standard.fit_transform(
            df[columns_for_standardization])

        # Apply One-hot encoding
        df["Ethnicity"] = df["Ethnicity"].astype("category")
        df["EducationLevel"] = df["EducationLevel"].astype("category")
        df = pd.get_dummies(df, columns=["Ethnicity"], prefix="Ethnicity")
        df = pd.get_dummies(
            df, columns=["EducationLevel"], prefix="EducationLevel")

        # Save preprocessed data
        processed_File = "processed_data.csv"
        df.to_csv(processed_File, index=False)

        # Prepare data for modeling
        dataForTargetColumn = df.drop(
            columns=['PatientID', 'Diagnosis', 'DoctorInCharge'])
        targetColumn = df['Diagnosis']
        dataForTargetColumn = dataForTargetColumn.to_numpy()
        targetColumn = targetColumn.to_numpy()
        # Convert boolean to int
        bool_columns = df.select_dtypes(include=['bool']).columns
        df[bool_columns] = df[bool_columns].astype('int32')

        # Convert all numeric to float32
        numeric_columns = df.select_dtypes(
            include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].astype('float32')

        # Numpy conversion
        dataForTargetColumn = dataForTargetColumn.astype('float32')
        targetColumn = targetColumn.astype('float32')

        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        # Hyperparameters
        I = dataForTargetColumn.shape[1]
        hidden_layer_configs = [I//2, 2*I//3, I, 2*I]
        input_activation_functions = ['tanh', 'elu', 'relu', 'silu']
        loss_functions = ['binary_crossentropy', 'mean_squared_error']
        output_activation_fns = ['sigmoid']

        # Store results
        results = {}

        for loss_fn in loss_functions:
            for input_activation_fn in input_activation_functions:
                for output_activation_fn in output_activation_fns:
                    for hidden_units in hidden_layer_configs:
                        print(f"Training with {hidden_units} hidden neurons:")

                        # Initialize metrics storage
                        fold_losses = []
                        fold_accuracies = []
                        val_accuracies = []
                        fold_bces = []
                        fold_mces = []
                        histories = []

                        # Early stopping
                        early_stopping = EarlyStopping(
                            monitor='val_loss',
                            patience=5,
                            min_delta=0.001,
                            restore_best_weights=True,
                            mode='min',
                            verbose=1
                        )

                        # Cross-validation
                        for train_idx, val_idx in skf.split(dataForTargetColumn, targetColumn):
                            x_train, x_val = dataForTargetColumn[train_idx], dataForTargetColumn[val_idx]
                            y_train, y_val = targetColumn[train_idx], targetColumn[val_idx]

                            y_train = y_train.reshape(-1, 1)
                            y_val = y_val.reshape(-1, 1)

                            # Define model
                            model = Sequential([
                                Input(shape=(I,)),
                                Dense(hidden_units,
                                      activation=input_activation_fn),
                                Dense(1, activation=output_activation_fn)
                            ])

                            # Compile and fit model
                            model.compile(
                                optimizer=SGD(learning_rate=0.001),
                                loss=loss_fn,
                                metrics=['accuracy', 'binary_crossentropy',
                                         'mean_squared_error']
                            )

                            history = model.fit(
                                x_train, y_train,
                                epochs=100,
                                batch_size=32,
                                validation_data=(x_val, y_val),
                                verbose=0,
                                callbacks=[early_stopping]
                            )

                            # Store results
                            train_metrics = model.evaluate(
                                x_train, y_train, verbose=0)
                            val_metrics = model.evaluate(
                                x_val, y_val, verbose=0)

                            fold_losses.append(train_metrics[0])
                            fold_accuracies.append(train_metrics[1])
                            fold_bces.append(train_metrics[2])
                            fold_mces.append(train_metrics[3])
                            val_accuracies.append(val_metrics[1])
                            histories.append(history)

                        # Calculate averages
                        avg_train_loss, avg_val_loss = plot_convergence(
                            histories, loss_fn, input_activation_fn, output_activation_fn, hidden_units)

                        # Store configuration results
                        results[(loss_fn, input_activation_fn, hidden_units)] = {
                            'Train Accuracy': np.mean(fold_accuracies),
                            'Train Loss': np.mean(fold_losses),
                            'Train BCE': np.mean(fold_bces),
                            'Train MSE': np.mean(fold_mces),
                            'Validation Accuracy': np.mean(val_accuracies)
                        }

                        print(
                            f"Configuration {hidden_units} neurons, Loss: {loss_fn}, Activation: {input_activation_fn}")
                        print(
                            f"Training converges at epoch: {np.argmin(avg_train_loss)}")
                        print(
                            f"Validation converges at epoch: {np.argmin(avg_val_loss)}")

        # Save results to CSV
        results_df = pd.DataFrame([
            {
                'Loss Function': loss_fn,
                'Input Activation Function': act_fn,
                'Hidden Neurons': units,
                **metrics
            }
            for (loss_fn, act_fn, units), metrics in results.items()
        ])

        results_df = results_df.sort_values(
            by='Validation Accuracy', ascending=False)
        results_df.to_csv("model_comparison_results.csv", index=False)
        print("\n✅ Results saved to model_comparison_results.csv")
        print(results_df.head())

    except FileNotFoundError:
        print(f"❌ Error: File '{FILE}' not found.")
    except PermissionError:
        print(f"❌ Error: Not permitted to use '{FILE}'.")
    except UnicodeDecodeError:
        print("❌ Error: Encoding error, please try again!")
    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
