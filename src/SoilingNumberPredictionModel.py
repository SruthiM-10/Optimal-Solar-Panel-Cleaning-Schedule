from sklearn.model_selection import train_test_split
from accessData import accessData
from sklearn.svm import SVR

import matplotlib.pyplot as plt

def prediction_model(path):
    data = accessData(path)

    train_df, test_df = train_test_split(data, test_size=0.8, random_state=1)
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



