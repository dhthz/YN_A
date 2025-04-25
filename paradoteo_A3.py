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


def plot_convergence(histories, lr, momentum):
    plt.figure(figsize=(10, 6))

    min_epochs = min(len(h.history['loss']) for h in histories)
    train_losses = np.array([h.history['loss'][:min_epochs]
                            for h in histories])
    val_losses = np.array([h.history['val_loss'][:min_epochs]
                          for h in histories])

    avg_train_loss = np.mean(train_losses, axis=0)
    avg_val_loss = np.mean(val_losses, axis=0)

    plt.plot(avg_train_loss, label='Train Loss')
    plt.plot(avg_val_loss, label='Validation Loss')
    plt.title(
        f'Average Loss Convergence (lr={lr}, m={momentum})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/convergence_lr{lr}_m{momentum}.png')
    plt.close()


def create_model(input_shape, hidden_units, learning_rate, momentum):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(hidden_units, activation='elu'),
        Dense(1, activation='sigmoid')
    ])

    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy', 'mean_squared_error']
    )
    return model


def main():
    # File paths and setup
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

        # One-hot encoding
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

        # Then proceed with numpy conversion
        dataForTargetColumn = dataForTargetColumn.astype('float32')
        targetColumn = targetColumn.astype('float32')

        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        # Hyperparameters
        configs = [
            {'lr': 0.001, 'm': 0.2, },
            {'lr': 0.001, 'm': 0.6, },
            {'lr': 0.05, 'm': 0.6, },
            {'lr': 0.1, 'm': 0.6, },
        ]

        hidden_units = 76
        # Store results
        results = {}

        for config in configs:
            lr, m, = config['lr'], config['m']
            print(f"\nTraining with lr={lr}, momentum={m}")
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
                model = create_model(
                    dataForTargetColumn.shape[1], hidden_units, lr, m)
                # Compile and fit model

                history = model.fit(
                    x_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(x_val, y_val),
                    verbose=0,
                    callbacks=[early_stopping]
                )
                # Store results
                train_metrics = model.evaluate(x_train, y_train, verbose=0)
                val_metrics = model.evaluate(x_val, y_val, verbose=0)

                fold_losses.append(train_metrics[0])
                fold_accuracies.append(train_metrics[1])
                fold_bces.append(train_metrics[2])
                fold_mces.append(train_metrics[3])
                val_accuracies.append(val_metrics[1])
                histories.append(history)
            # Plot convergence
            plot_convergence(histories, lr, m)

            results[(lr, m)] = {
                'Train Accuracy': np.mean(fold_accuracies),
                'Train Loss': np.mean(fold_losses),
                'Train BCE': np.mean(fold_bces),
                'Train MSE': np.mean(fold_mces),
                'Validation Accuracy': np.mean(val_accuracies)
            }

        # Save results to CSV
        results_df = pd.DataFrame([
            {
                'Learning Rate': lr,
                'Momentum': m,
                **metrics
            }
            for (lr, m), metrics in results.items()
        ])

        results_df = results_df.sort_values(
            by='Validation Accuracy', ascending=False)
        results_df.to_csv("A4.csv", index=False)
        print("\n✅ Results saved to A3.csv")
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
