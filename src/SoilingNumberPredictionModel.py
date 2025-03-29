from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import pandas as pd
import rdtools

def SVC(data):

    train_df, test_df = train_test_split(data, test_size=0.3, random_state=1)
    train_x = train_df[["ac_power", "poa_irradiance", "ambient_temp", "wind_speed"]]
    train_y = train_df["soiling"]

    test_x = test_df[["ac_power", "poa_irradiance", "ambient_temp", "wind_speed"]]
    test_y = test_df["soiling"]

    # Support Vector Machine
    svc = SVR(gamma=9.999999999999999e-05, epsilon=0.05159408053585679, kernel='rbf', degree=1000)
    svc.fit(train_x, train_y)
    y_pred = svc.predict(test_x)
    plt.scatter(test_y.index, test_y, color="red", label="Actual")
    plt.scatter(test_y.index, y_pred, color="blue", label="Predicted")
    plt.legend()
    plt.show()

def NeuralNetwork(data):
    train_df, test_df = train_test_split(data, test_size=0.3, random_state=1)
    train_x = train_df[["ac_power", "poa_irradiance", "ambient_temp", "wind_speed"]]
    train_y = train_df["soiling"]

    test_x = test_df[["ac_power", "poa_irradiance", "ambient_temp", "wind_speed"]]
    test_y = test_df["soiling"]
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    mean_y = test_y.mean()

    model = Sequential([
        tf.keras.Input(shape=(4,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation= 'relu'),
        tf.keras.layers.Dropout(rate= 0.1),
        Dense(1, activation=None,
              kernel_initializer=tf.keras.initializers.HeUniform(),
              bias_initializer=tf.keras.initializers.Constant(mean_y))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),
                  loss='mse',
                  metrics=['mae', tf.keras.metrics.MeanAbsolutePercentageError()])
    model.fit(train_x, train_y, batch_size=64, epochs=10)
    y_pred = model.predict(test_x)
    return y_pred, model

def DegredationRate(data):
    data = data.resample('D').ffill()
    data = data.asfreq("D")
    data.index = pd.to_datetime(data.index)
    degredation_result = rdtools.degradation_classical_decomposition(data["ac_power"])
    return degredation_result
    # diff = data.diff()
    # diff.dropna(inplace = True)
    # diff = diff[diff["soiling"] < 0]
    # return diff["soiling"].mean()

def hyperparameterTuning(data):
    train_df, test_df = train_test_split(data, test_size=0.3, random_state=1)
    train_x = train_df[["ac_power", "poa_irradiance", "ambient_temp", "wind_speed"]]
    train_y = train_df["soiling"]

    test_x = test_df[["ac_power", "poa_irradiance", "ambient_temp", "wind_speed"]]
    test_y = test_df["soiling"]
    mean_y = test_y.mean()

    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(4,)))

        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                                   activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh'])),
            model.add(layers.Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1))))

        model.add(layers.Dense(1, activation= None,
              kernel_initializer=tf.keras.initializers.HeUniform(),
              bias_initializer=tf.keras.initializers.Constant(mean_y)))

        optimizer = keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [3e-3, 3e-4, 3e-5]))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', tf.keras.metrics.R2Score(), tf.keras.metrics.MeanAbsolutePercentageError()])

        return model

    # Use Keras Tuner
    tuner = kt.Hyperband(build_model,
                          objective='val_loss',
                          max_epochs=20,
                          factor=3,
                          directory='hyperparam_tuning_HB',
                          project_name='regression_nn')

    # Run the hyperparameter search
    tuner.search(train_x, train_y, epochs=20, verbose=1, validation_data=(test_x, test_y))

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters:")
    print(f"Number of layers: {best_hps.get('num_layers')}")
    print(f"Dropout rate: {best_hps.get('dropout')}")
    print(f"Learning rate: {best_hps.get('learning_rate')}")
    for i in range(best_hps.get('num_layers')):
        print(f"Units in layer {i}: {best_hps.get(f'units_{i}')}")






