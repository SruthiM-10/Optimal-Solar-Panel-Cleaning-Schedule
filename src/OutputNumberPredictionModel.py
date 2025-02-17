import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from sklearn.model_selection import train_test_split
import random
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras import layers
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def NeuralNetwork(data):
      train_df, test_df = train_test_split(data, test_size=0.3, random_state=1)
      train_x = train_df[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]]
      train_y = train_df["ac_power"]

      test_x = test_df[["poa_irradiance", "ambient_temp", "wind_speed", "soiling"]]
      test_y = test_df["ac_power"]
      mean_y = train_y.mean()
      print(mean_y)

      SEED = 42
      random.seed(SEED)
      np.random.seed(SEED)
      tf.random.set_seed(SEED)
      model = Sequential([
        tf.keras.Input(shape=(4,)),
        Dense(224, activation='tanh'),
        Dense(256, activation='relu'),
        Dense(1, activation=None,
              kernel_initializer=tf.keras.initializers.HeUniform(),
              bias_initializer=tf.keras.initializers.Constant(mean_y))
      ])
      model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(3e-4), metrics=["mse", tf.keras.metrics.MeanAbsolutePercentageError()])
      model.fit(train_x, train_y, epochs=15, validation_data= (test_x, test_y))
      y_pred = model.predict(test_x)
      print(mean_absolute_percentage_error(test_y, y_pred))
      plt.scatter(np.arange(0, y_pred.size), y_pred, label = "predicted")
      plt.scatter(np.arange(0, test_y.size), test_y, label = "actual")
      plt.legend()
      plt.show()

      return model

def hyperparameterTuning(original_data):
    scaler = MinMaxScaler()
    data = original_data.copy()
    data["ac_power"] = scaler.fit_transform(data[["ac_power"]])
    train_df, test_df = train_test_split(data, test_size=0.3, random_state=1)
    train_x = train_df[["soiling", "poa_irradiance", "ambient_temp", "wind_speed"]]
    train_y = train_df["ac_power"]

    test_x = test_df[["soiling", "poa_irradiance", "ambient_temp", "wind_speed"]]
    test_y = test_df["ac_power"]
    mean_y = test_y.mean()

    expected_loss = np.var(test_y)
    print(expected_loss)

    def build_model(hp):
        model = Sequential()
        model.add(layers.Input(shape=(4,)))

        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                                   activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh'])),
            model.add(layers.Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1))))

        model.add(layers.Dense(1, activation= None,
              kernel_initializer=tf.keras.initializers.HeUniform(),
              bias_initializer=tf.keras.initializers.Constant(mean_y)))

        optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [3e-3, 3e-4, 3e-5]))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', tf.keras.metrics.R2Score(), tf.keras.metrics.MeanAbsolutePercentageError()])

        return model

    # Use Keras Tuner
    tuner = kt.Hyperband(build_model,
                          objective='val_loss',
                          max_epochs=40,
                          factor=3,
                          directory='hyperparam_tuning_HB_onpm',
                          project_name='regression_nn2')

    # Run the hyperparameter search
    tuner.search(train_x, train_y, epochs=40, verbose=1, validation_data=(test_x, test_y))

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters:")
    print(f"Number of layers: {best_hps.get('num_layers')}")
    print(f"Dropout rate: {best_hps.get('dropout')}")
    print(f"Learning rate: {best_hps.get('learning_rate')}")
    for i in range(best_hps.get('num_layers')):
        print(f"Units in layer {i}: {best_hps.get(f'units_{i}')}")

    '''
    Best val_loss So Far: 0.015745321288704872
    Total elapsed time: 00h 01m 39s
    Best hyperparameters:
    Number of layers: 2
    Dropout rate: 0.0
    Learning rate: 0.0003
    Units in layer 0: 224
    Units in layer 1: 256
    '''

