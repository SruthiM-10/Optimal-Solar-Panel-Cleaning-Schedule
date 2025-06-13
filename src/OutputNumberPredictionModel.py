import tensorflow as tf
from keras.src.callbacks import early_stopping
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
import random
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
# import keras_tuner as kt

def smape(y_true, y_pred):
    epsilon = tf.keras.backend.epsilon()
    numerator = tf.abs(y_true - y_pred)
    denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2 + epsilon  # Prevent division by zero
    return tf.reduce_mean(numerator / denominator) * 100

def log_cosh_mape(y_true, y_pred):
    epsilon = tf.keras.backend.epsilon()
    y_true = tf.where(tf.abs(y_true) < epsilon, epsilon, y_true)
    diff = (y_true - y_pred) / tf.abs(y_true)
    diff = tf.clip_by_value(diff, -10.0, 10.0)
    return tf.reduce_mean(tf.math.log(tf.cosh(diff))) * 100  # Scale result

def msle(y_true, y_pred):
    epsilon = tf.keras.backend.epsilon()
    y_true = tf.where(y_true < 0, 0.0, y_true)
    y_pred = tf.where(y_pred < 0, 0.0, y_pred)
    return tf.reduce_mean(tf.square(tf.math.log1p(y_true + epsilon) - tf.math.log1p(y_pred + epsilon)))

def FeedForwardNetwork(data):
      arr = data["ac_power"]
      mean_y = arr.mean()

      expected_loss = np.var(arr)

      SEED = 23
      random.seed(SEED)
      np.random.seed(SEED)
      tf.random.set_seed(SEED)
      model = Sequential([
          tf.keras.Input(shape=(6,)),
          Dense(128, activation="relu"),
          Dense(200),
          BatchNormalization(),
          tf.keras.layers.Activation("tanh"),
          Dense(200, activation="relu"),
          Dense(128, activation="relu"),
          Dense(1, kernel_initializer=tf.keras.initializers.HeUniform(),
                bias_initializer=tf.keras.initializers.Constant(mean_y)),
      ])
      model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(3e-4, clipnorm= 1.0),
                    metrics=[tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.RootMeanSquaredError()])
      return model

def get_sample_weights(length):
    weights = np.linspace(1, 3, length)  # Linear increasing weights
    return weights / weights.sum()

def train(days_data):
    seed = 42
    np.random.seed(seed)
    bootstrapped_df = days_data.sample(n=40000, replace=True, random_state=seed)
    bootstrapped_df.sort_index(inplace=True)

    def generate_jittered_data(df, num_copies=9, noise_level=0.02):
        augmented_data = []
        for _ in range(num_copies):
            std_dev = df.std()
            noise = np.random.normal(loc=0, scale=noise_level * std_dev, size=df.shape)
            jittered_df = df + noise
            augmented_data.append(jittered_df)
        df_expanded = pd.concat([df] + augmented_data, ignore_index=True)
        return df_expanded
    jittered_data = generate_jittered_data(days_data)

    data = jittered_data

    window_size = 6000
    step_size = 2000
    model = FeedForwardNetwork(data)
    for i in range(0, len(data) - window_size - 2 * step_size, step_size):
        train_df = data[:i + window_size]
        val_df = data[i + window_size:i + window_size + step_size]
        train_x = train_df[["poa_irradiance", "ambient_temp", "wind_speed", "soiling", 'day', 'month']]
        train_y = train_df["ac_power"]

        val_x = val_df[["poa_irradiance", "ambient_temp", "wind_speed", "soiling", 'day', 'month']]
        val_y = val_df["ac_power"]
        early_stop = EarlyStopping(monitor= 'val_loss', patience=10, restore_best_weights=True)
        sample_weights = get_sample_weights(len(train_y))
        model.fit(train_x, train_y, sample_weight= sample_weights, epochs=200, batch_size=64, verbose= 1, validation_data= (val_x, val_y), callbacks= [early_stop])

        test_df = data[i + window_size + step_size: i + window_size + 2 * step_size]
        test_x = test_df[["poa_irradiance", "ambient_temp", "wind_speed", "soiling", 'day', 'month']]
        test_y = test_df["ac_power"]
        y_pred = model.predict(test_x).flatten()
        print(f"MSE: {mean_squared_error(test_y, y_pred, squared= False)}")
        print(f"MAPE: {mean_absolute_percentage_error(test_y, y_pred)}")
        print(f'R^2: {r2_score(test_y, y_pred)}')

    y_pred = model.predict(data[["poa_irradiance", "ambient_temp", "wind_speed", "soiling", 'day', 'month']]).flatten()
    print(f"MSE: {mean_squared_error(data['ac_power'], y_pred, squared= False)}")
    print(f"MAPE: {mean_absolute_percentage_error(data['ac_power'], y_pred)}")
    print(f'R^2: {r2_score(data["ac_power"], y_pred)}')

    return model

def hyperparameterTuning(train_df, test_df):
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
                                   activation=hp.Choice(f'activation_{i}', ['relu', 'sigmoid', 'tanh'])),
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
                          max_epochs=20,
                          factor=3,
                          directory='hyperparam_tuning_HB_onpm',
                          project_name='regression_nn5')

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

'''
Log notes:
Model 2 - there were bugs in code, underfitting
Model 3 - shows signs of overfitting (maybe try deleting some layers)

Model 5 (best so far) - added day
Model 6 - added month

Model 7 - pretty good! (5 copies of jittered data)
Model 8 - 9 copies
Model 9 - Dropout 0.3
Model 10 - pretty good! (Dropout 0.2, Activation after Batch Normalization, rmse = 22.75)
Model 11 - No dropout, rmse = 18!
Tried - Removed latter kernel_regularizer.. basically the same but marginally better
'''


