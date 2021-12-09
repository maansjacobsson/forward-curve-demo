import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import holidays
from datetime import timedelta

from sklearn.model_selection import train_test_split


import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Lambda, Multiply


def load_data(path: str) -> pd.DataFrame:
    """Load and process dataframe"""
    # Read data
    df = pd.read_csv(path, index_col=0)
    df = df[["DE_SPOT_PX"]].dropna()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.tz_convert("CET")

    # Hourly resolution
    df = df.resample("1H").asfreq()

    # Daily resolution
    df_daily = pd.DataFrame(
        {"date": df.index.date, "hour": df.index.hour, "price": df["DE_SPOT_PX"]}
    ).copy()
    df_daily.reset_index(drop=True, inplace=True)

    df_daily = df_daily.pivot_table(values="price", columns="hour", index="date")
    df_daily.index = pd.to_datetime(df_daily.index)
    df_daily = df_daily.dropna()
    df_daily["base"] = df_daily.mean(axis=1)

    df["start_of_week"] = df.index.date
    df["start_of_week"].loc[df.index.dayofweek != 0] = 0
    df["start_of_week"].replace(to_replace=0, method="ffill", inplace=True)
    df.drop(df[df["start_of_week"] == 0].index, inplace=True)

    # Binary holiday feature
    h = holidays.CountryHoliday("DE")
    df["holiday"] = np.array(df.index.map(lambda x: x in h)).astype("int8")

    df_weekly = pd.DataFrame(
        {
            "start_of_week": df["start_of_week"],
            "dayofweek": df.index.dayofweek,
            "price": df["DE_SPOT_PX"],
            "holiday": df["holiday"],
        }
    ).copy()

    # Weekly resolution with columns representing the base of each day.
    df_price = df_weekly.pivot_table(
        values="price", columns="dayofweek", index="start_of_week"
    )
    df_price = df_price.dropna()

    # Weekly resolution with columns representing binary holiday feature of each day.
    df_holiday = df_weekly.pivot_table(
        values="holiday", columns="dayofweek", index="start_of_week"
    )
    df_holiday = df_holiday[df_holiday.index.isin(df_price.index)]

    # First 7 columns is price mon-sun, following 7 columns is binary holiday mon-sun
    df_weekly = pd.concat([df_price, df_holiday], axis=1)

    # Weekly base price
    df_weekly["base"] = df_weekly.iloc[:, 0:7].mean(axis=1)

    # Column of ones
    df_weekly["ones"] = 1

    # Renaming columns
    for i in range(7):
        df_weekly.columns.values[i + 7] = str(i) + " Holiday"

    # same values plotted with different granularity.
    plt.step(df.index, df["DE_SPOT_PX"], where="post", label="hourly")
    plt.step(df_daily.index, df_daily.base, where="post", label="daily")
    plt.step(df_weekly.index, df_weekly.base, where="post", label="weekly")
    plt.legend()
    plt.show()

    return df_weekly


def stable_softmax(x):
    """Stable softmax for scaling labels/targets."""
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def create_xarray(df: pd.DataFrame) -> xr.Dataset:
    """Scaling and creating xarrays with features and labels/targets. """
    # Features: Holidays and base price. Labels: Daily base price.
    # x_features = [df.columns.values[i] for i in range(7, 15)]
    x_features = ["ones"]
    y_labels = [df.columns.values[i] for i in range(7)]

    # Scaling daily prices by subtracting mean, dividing by 20.0 and then stable softmax.
    df_norm = df.copy()
    for row in range(len(df_norm)):
        data = df_norm.iloc[row, 0:7].values.T
        data = (data - data.mean()) / 20.0
        df_norm.iloc[row, 0:7] = stable_softmax(data).T

    # Scaling week base price.
    df_norm["base"] = df_norm["base"] / 30.0

    # Create xarrays
    ds = xr.Dataset()
    ds["x"] = xr.DataArray(
        df_norm[x_features].values,
        dims=["datetime", "x_features"],
        coords=[df_norm.index.values, x_features],
    )
    ds["y"] = xr.DataArray(
        df_norm[y_labels].values,
        dims=["datetime", "y_labels"],
        coords=[df_norm.index.values, y_labels],
    )

    return ds


def train_validation_split(ds: xr.Dataset, size: float):
    """Splitting arrays randomly into training and validation"""
    x_train, x_validation, y_train, y_validation = train_test_split(
        ds.x, ds.y, test_size=size
    )

    x_train = x_train.sortby("datetime")
    x_validation = x_validation.sortby("datetime")
    y_train = y_train.sortby("datetime")
    y_validation = y_validation.sortby("datetime")

    return x_train, x_validation, y_train, y_validation


def create_model(x_inp, y_target):
    """Creating the model. """

    # declear weights as non-trainable
    # Input.
    # x = Input(shape=1, dtype="float64", name=)
    input_one = Input(shape=x_inp.shape[1], dtype="float64", name="1")

    x = Dense(512, activation="tanh", dtype="float64", name="dense1")(input_one)
    x = Dense(32, activation="tanh", dtype="float64", name="dense2")(x)

    predictions = Dense(
        y_target.shape[1], activation="softmax", dtype="float64", name="output"
    )(x)

    model = Model(inputs=input_one, outputs=predictions)
    model.compile(loss="mae", optimizer="adam")
    model.summary()

    return model


def df_plots(
    df: pd.DataFrame,
    ds: xr.Dataset,
    y_train_pred: np.ndarray,
    y_train: xr.DataArray,
    y_val_pred: np.ndarray,
    y_val: xr.DataArray,
) -> dict:
    """Creating post processed time series for plotting. """
    # Creating dictionary for the dataframes.
    dfs = dict()

    # Targets data frame
    df_target = df.iloc[
        :, 0:7
    ].copy()  # TODO: scaling the targets directly becomes a problem when plotting.
    for i in range(len(ds.y)):
        data = df_target.iloc[[i]].values.T
        data = (data - np.mean(data)) / 20.0
        ds.y[[i]] = data.T

    df_target = ds.y.to_dataframe()
    df_target.reset_index(inplace=True)
    for i in range(len(df_target)):
        df_target.iloc[i, 0] = df_target.iloc[i, 0] + timedelta(
            days=df_target.iloc[i, 1].astype(float)
        )

    df_target.set_index(df_target["datetime"], inplace=True)
    df_target.drop(columns=["datetime", "y_labels"], inplace=True)
    df_target.index = pd.to_datetime(df_target.index.values)

    dfs["df_target"] = df_target["y"]

    # Training data frame
    for i in range(len(y_train_pred)):
        y_train_pred[[i]] = (np.log(y_train_pred[[i]]) - (1 / 7) * np.sum(
            np.log(y_train_pred[[i]])
        ))

    df_train_pred = y_train.to_dataframe()
    df_train_pred.reset_index(inplace=True)
    for i in range(len(df_train_pred)):
        df_train_pred.iloc[i, 0] = df_train_pred.iloc[i, 0] + timedelta(
            days=df_train_pred.iloc[i, 1].astype(float)
        )

    df_train_pred.set_index(df_train_pred["datetime"], inplace=True)
    df_train_pred.drop(columns=["datetime", "y_labels"], inplace=True)
    df_train_pred.index = pd.to_datetime(df_train_pred.index.values)

    df_train_pred["y"] = y_train_pred.ravel()
    dfs["df_train_pred"] = df_train_pred["y"].resample("1D").asfreq()

    # Validation data frame
    for i in range(len(y_val)):
        y_val_pred[[i]] = (np.log(y_val_pred[[i]]) - (1 / 7) * np.sum(
            np.log(y_val_pred[[i]])
        ))

    df_val_pred = y_val.to_dataframe()
    df_val_pred.reset_index(inplace=True)
    for i in range(len(df_val_pred)):
        df_val_pred.iloc[i, 0] = df_val_pred.iloc[i, 0] + timedelta(
            days=df_val_pred.iloc[i, 1].astype(float)
        )

    df_val_pred.set_index(df_val_pred["datetime"], inplace=True)
    df_val_pred.drop(columns=["datetime", "y_labels"], inplace=True)
    df_val_pred.index = pd.to_datetime(df_val_pred.index.values)

    df_val_pred["y"] = y_val_pred.ravel()
    dfs["df_val_pred"] = df_val_pred["y"].resample("1D").asfreq()

    return dfs


def make_plots(dfs, df, history=None):
    """Plotting post processed results. """
    # Fitting plot to check over-fitting (if data has been used for training).
    if history is not None:
        plt.plot(history.history["loss"], label="trian")
        plt.plot(history.history["val_loss"], label="validation")
        plt.legend()
        plt.grid()
        plt.show()

    # plot time series.
    plt.plot(dfs["df_target"], marker="o", color="k")
    plt.plot(dfs["df_train_pred"], marker="o", color="b")
    plt.plot(dfs["df_val_pred"], marker="o", color="g")

    h = holidays.CountryHoliday("DE")
    df_h = dfs["df_target"].to_frame()
    df_h["holiday"] = np.array(df_h.index.map(lambda x: x in h)).astype("int8")
    df_h = df_h["holiday"].loc[df_h["holiday"] == 1].index
    for i in df_h:
        plt.axvspan(i, (i + timedelta(days=1)), alpha=0.5, fill=True, color="r")

    plt.legend(["Targets", "Train", "Validation", "Holiday"])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Load parsed data with features.
    PATH = "C:\\Users\\xjacobmaa\\PycharmProjects\\CPO-opt\\Spot prices data\\DE.csv"
    df = load_data(PATH)

    # Create xarrays.
    ds = create_xarray(df)

    # Split data into training and validation.
    x_train, x_val, y_train, y_val = train_validation_split(ds, 0.10)

    # Create model.
    model = create_model(ds["x"], ds["y"])

    # Fit model.
    history = model.fit(
        x_train.values,
        y_train.values,
        epochs=100,
        batch_size=32,
        validation_data=(x_val.values, y_val.values),
    )

    # Predict values on training and validation data.
    y_train_pred = model.predict(x_train.values)
    y_val_pred = model.predict(x_val.values)

    dfs = df_plots(df, ds, y_train_pred, y_train, y_val_pred, y_val)

    make_plots(dfs, df, history)
    print("testing testing testing testing testing testing ")