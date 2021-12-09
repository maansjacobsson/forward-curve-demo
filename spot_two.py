import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import timedelta
from scipy.optimize import curve_fit

from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Dense, Input, Concatenate, Lambda, Multiply

import holidays


class SpotProfile:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @staticmethod
    def load_data(path: str, file_name) -> pd.DataFrame:
        """Load dataframe."""
        # Read data
        df = pd.read_csv(path + file_name, index_col=0)
        df = df[["DE_SPOT_PX"]].dropna()

        if str(df.index[0]).find("+") != -1:
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.tz_convert("CET")
        else:
            df.index = pd.to_datetime(df.index)

        # Pivoting the table so that each row represents one day
        df = pd.DataFrame(
            {"date": df.index.date, "hour": df.index.hour, "price": df["DE_SPOT_PX"]}
        ).copy()
        df.reset_index(drop=True, inplace=True)

        df = df.pivot_table(values="price", columns="hour", index="date")
        df.index = pd.to_datetime(df.index)
        df = df.dropna()

        # Day of the week sinus and cosinus feature;
        df["Weekday_SIN"] = np.sin(
            (pd.to_datetime(df.index).day_of_week / 7.0) * 2 * np.pi
        )
        df["Weekday_COS"] = np.cos(
            (pd.to_datetime(df.index).day_of_week / 7.0) * 2 * np.pi
        )

        # Binary holiday feature
        h = holidays.CountryHoliday("DE")
        df["Holiday"] = np.array(df.index.map(lambda x: x in h)).astype("int8")

        # Peak and off-peak value features (scaled).
        df["Peak_mean"] = (df.iloc[:, 8:20].mean(axis=1)) / 30.0
        df["Off-peak_mean"] = (
            df.iloc[:, 0:8].mean(axis=1) * (2 / 3)
            + df.iloc[:, 20:24].mean(axis=1) * (1 / 3)
        ) / 30.0

        return df

    @staticmethod
    def mean_std(df: pd.DataFrame):

        # testing some linear regression stuff...
        def linear(x, a, b):
            return a * x + b

        df = df.copy()

        df["Peak_mean"] = df.iloc[:, 8:20].mean(axis=1)
        df["Peak_std"] = df.iloc[:, 8:20].std(axis=1)
        df["Off-peak_mean"] = df.iloc[:, 0:8].mean(axis=1) * (2 / 3) + df.iloc[
            :, 20:24
        ].mean(axis=1) * (1 / 3)
        df["Off-peak_std"] = df.iloc[:, 0:8].std(axis=1) * (2 / 3) + df.iloc[
            :, 20:24
        ].std(axis=1) * (1 / 3)
        df["Off-peak_m_mean"] = df.iloc[:, 0:8].mean(axis=1)
        df["Off-peak_e_mean"] = df.iloc[:, 20:24].mean(axis=1)
        df["Off-peak_m_std"] = df.iloc[:, 0:8].std(axis=1)
        df["Off-peak_e_std"] = df.iloc[:, 20:24].std(axis=1)

        df = df.loc[
            :,
            [
                "mean",
                "std",
                "Peak_mean",
                "Peak_std",
                "Off-peak_mean",
                "Off-peak_std",
                "Off-peak_m_mean",
                "Off-peak_m_std",
                "Off-peak_e_mean",
                "Off-peak_e_std",
            ],
        ].copy()

        # Linear regression for mean price values from negative to 10
        constanst_neg_ten = curve_fit(
            linear,
            df["mean"].loc[df["mean"] <= 10.0].values,
            df["std"].loc[df["mean"] <= 10.0].values,
        )
        a_fit_neg = constanst_neg_ten[0][0]
        b_fit_neg = constanst_neg_ten[0][1]
        fit_neg_ten = []
        for i in df["mean"].loc[df["mean"] <= 10.0]:
            fit_neg_ten.append(linear(i, a_fit_neg, b_fit_neg))

        plt.plot(
            df["mean"].loc[df["mean"] <= 10.0],
            fit_neg_ten,
            "k",
            label="Linear regression  ]10,]",
        )

        # Linear regression for mean price values greater then 10
        constants_ten = curve_fit(
            linear,
            df["mean"].loc[df["mean"] > 10.0].values,
            df["std"].loc[df["mean"] > 10.0].values,
        )
        a_fit_ten = constants_ten[0][0]
        b_fit_ten = constants_ten[0][1]
        fit_ten = []
        for i in df["mean"].loc[df["mean"] > 10.0]:
            fit_ten.append(linear(i, a_fit_ten, b_fit_ten))

        plt.plot(
            df["mean"].loc[df["mean"] > 10.0],
            fit_ten,
            "k",
            label="Linear regression [, 10]",
        )

        # Linear regression for all mean price values
        constants_all = curve_fit(linear, df["mean"].values, df["std"].values)
        a_fit = constants_all[0][0]
        b_fit = constants_all[0][1]
        fit_all = []
        for i in df["mean"].values:
            fit_all.append(linear(i, a_fit, b_fit))

        plt.plot(df["mean"], fit_all, "k", label="Linear regression all values")

        plt.scatter(df["mean"], df["std"], marker=".", label="all 24h")
        plt.xlabel("Daily Price Mean")
        plt.ylabel("Daily Price Std")
        plt.legend()
        plt.grid()
        plt.show()

        plt.scatter(
            df["Peak_mean"],
            df["Peak_std"],
            marker=".",
            alpha=0.5,
            color="g",
            label="Peak hours",
        )
        plt.scatter(
            df["Off-peak_m_mean"],
            df["Off-peak_m_std"],
            marker=".",
            alpha=0.5,
            color="b",
            label="Morning Off-peak hours",
        )
        plt.scatter(
            df["Off-peak_e_mean"],
            df["Off-peak_e_std"],
            marker=".",
            alpha=0.5,
            color="r",
            label="Evening Off-peak hours",
        )
        plt.xlabel("Mean")
        plt.ylabel("Standard deviation")
        plt.legend()
        plt.grid()
        plt.show()

        plt.scatter(
            df["Peak_mean"],
            df["Peak_std"],
            marker=".",
            alpha=0.3,
            color="b",
            label="Peak hours",
        )
        plt.scatter(
            df["Off-peak_mean"],
            df["Off-peak_std"],
            marker=".",
            alpha=0.3,
            color="g",
            label="Off-peak hours",
        )
        plt.xlabel("Mean")
        plt.ylabel("Standard deviation")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def stable_softmax(x):
        """Stable softmax for scaling labels/targets."""
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    @staticmethod
    def create_xarray(df: pd.DataFrame) -> xr.Dataset:
        """Construct dataset with input (X) and target (y) arrays."""
        x_features = [
            "Weekday_SIN",
            "Weekday_COS",
            "Peak_mean",
            "Off-peak_mean",
            "Holiday",
            "Peak_quota",
            "Off-peak_quota",
        ]
        y_labels = [i for i in range(24)]

        df_norm = df.drop(
            columns=[
                "Weekday_SIN",
                "Weekday_COS",
                "Holiday",
                "Peak_mean",
                "Off-peak_mean",
            ]
        ).copy()

        # Scaling the labels/targets.
        for row in range(len(df_norm)):
            data = df_norm.iloc[[row]].values.T
            data = (data - data.mean()) / 20.0
            df_norm.iloc[[row]] = SpotProfile.stable_softmax(data).T

        # Creating the input peak and off-peak quotas for every day.
        df_norm["Peak_quota"] = 0
        df_norm["Off-peak_quota"] = 0

        for i in range(len(df_norm)):
            df_norm.iloc[i, 24] = df_norm.iloc[i, 8:20].sum()
            df_norm.iloc[i, 25] = (
                df_norm.iloc[i, 0:8].sum() + df_norm.iloc[i, 20:24].sum()
            )

        df_norm[
            ["Weekday_SIN", "Weekday_COS", "Holiday", "Peak_mean", "Off-peak_mean"]
        ] = df[["Weekday_SIN", "Weekday_COS", "Holiday", "Peak_mean", "Off-peak_mean"]]

        # Creating xarrays.
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

    @staticmethod
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

    @staticmethod
    def create_model(x_inp, y_target):
        """Creating the model."""

        # Input.
        inputs = Input(shape=(x_inp.shape[1] - 2), dtype="float64", name="INPUT")
        peak_input = Input(shape=12, dtype="float64", name="PEAK_QUOTA")
        off_peak_input = Input(shape=12, dtype="float64", name="OFF-PEAK_QUOTA")

        # Two dense layers.
        x = Dense(
            y_target.shape[1] * 3, activation="tanh", dtype="float64", name="DENSE1"
        )(inputs)
        x = Dense(48, activation="tanh", dtype="float64", name="DENSE2")(x)

        # Splitting the output from the last dense layer.
        x_p = Lambda(lambda i: i[:, 0:24], dtype="float64", name="SPLIT_PEAK")(x)
        x_op = Lambda(lambda i: i[:, 24:48], dtype="float64", name="SPLIT_OFF-PEAK")(x)

        # Creating the parallel peak and off-peak layers with softmax.
        x_p = Dense(12, activation="softmax", dtype="float64", name="PEAK")(x_p)
        x_op = Dense(12, activation="softmax", dtype="float64", name="OFF-PEAK")(x_op)

        # Multiplying the output with the corresponding percentage for peak and off-peak hours.
        x_p = Multiply(dtype="float64", name="PEAK_SUM")([x_p, peak_input])
        x_op = Multiply(dtype="float64", name="OFF-PEAK_SUM")([x_op, off_peak_input])

        x_op_m = Lambda(
            lambda i: i[:, 0:8], dtype="float64", name="SPLIT_OFF-PEAK_MORNING"
        )(x_op)
        x_op_e = Lambda(
            lambda i: i[:, 8:12], dtype="float64", name="SPLIT_OFF-PEAK_EVENING"
        )(x_op)

        # Merging the outputs to create a 24 hour output with peak hours being the 12 first values.
        predictions = Concatenate(axis=1, dtype="float64", name="OUTPUT")(
            [x_op_m, x_p, x_op_e]
        )

        # Creating the model that includes all the layers above
        model = Model(
            inputs=[inputs, peak_input, off_peak_input],
            outputs=predictions,
            name="Michel_Architecture",
        )
        model.compile(loss="mae", optimizer="adam")
        model.summary()

        return model

    @staticmethod
    def df_plots(
        df: pd.DataFrame,
        ds: xr.Dataset,
        y_train: xr.DataArray,
        y_train_pred: np.ndarray,
        y_validation: xr.DataArray,
        y_validation_pred: np.ndarray,
    ) -> dict:
        # Create dictionary of dataframes for plotting
        dfs = dict()

        # Targets scaled (mean off the day subtracted then divided by 20)
        df_target = df.drop(
            columns=[
                "Weekday_SIN",
                "Weekday_COS",
                "Holiday",
                "Peak_mean",
                "Off-peak_mean",
            ]
        ).copy()
        for i in range(len(ds.y)):
            data = df_target.iloc[[i]].values.T
            data = (data - np.mean(data)) / 20.0
            ds.y[[i]] = data.T

        df_target = ds.y.to_dataframe()
        df_target.index = (
            df_target.index.get_level_values(0).astype(str)
            + " "
            + df_target.index.get_level_values(1).astype(str)
            + ":00"
        )
        df_target.index = pd.to_datetime(df_target.index, format="%Y-%m-%d %H:%M")
        df_target = df_target.sort_index()
        dfs["df_actuals"] = df_target["y"]

        # Models training data predictions post processed
        for i in range(len(y_train_pred)):
            y_train_pred[[i]] = np.log(y_train_pred[[i]]) - (1 / 24) * np.sum(
                np.log(y_train_pred[[i]])
            )

        df_train_pred = y_train.to_dataframe()
        df_train_pred.index = (
            df_train_pred.index.get_level_values(0).astype(str)
            + " "
            + df_train_pred.index.get_level_values(1).astype(str)
            + ":00"
        )
        df_train_pred.index = pd.to_datetime(
            df_train_pred.index.values, format="%Y-%m-%d %H:%M"
        )
        df_train_pred["y"] = y_train_pred.ravel()
        dfs["df_train_pred"] = df_train_pred.resample("1H").asfreq()

        # Models validation data prediction post processed
        for j in range(len(y_validation_pred)):
            y_validation_pred[[j]] = np.log(y_validation_pred[[j]]) - (1 / 24) * np.sum(
                np.log(y_validation_pred[[j]])
            )

        df_validation_pred = y_validation.to_dataframe()
        df_validation_pred.index = (
            df_validation_pred.index.get_level_values(0).astype(str)
            + " "
            + df_validation_pred.index.get_level_values(1).astype(str)
            + ":00"
        )
        df_validation_pred.index = pd.to_datetime(
            df_validation_pred.index.values, format="%Y-%m-%d %H:%M"
        )
        df_validation_pred["y"] = y_validation_pred.ravel()
        dfs["df_validation_pred"] = df_validation_pred.resample("1H").asfreq()

        return dfs

    @staticmethod
    def make_plots(dfs, df, history=None):
        """Plot results."""

        # Make fitting plot to check over-fitting and bias variance trade-off
        if history is not None:
            plt.plot(history.history["loss"], label="train")
            plt.plot(history.history["val_loss"], label="validation")
            plt.legend()
            plt.grid()
            plt.show()

        # Plot timeseries
        plt.plot(dfs["df_actuals"], "k")
        plt.plot(dfs["df_train_pred"], "b")
        plt.plot(dfs["df_validation_pred"], "g")

        h = df["Holiday"].loc[df["Holiday"] == 1].index
        for i in h:
            plt.axvspan(i, (i + timedelta(days=1)), alpha=0.5, fill=True, color="r")

        plt.legend(["Actuals", "Train", "Validation", "Holiday"])
        plt.grid()
        plt.show()

        # # Plot days of week
        # days = [
        #     "Monday",
        #     "Tuesday",
        #     "Wednesday",
        #     "Thursday",
        #     "Friday",
        #     "Saturday",
        #     "Sunday",
        # ]
        # x = [i for i in range(24)]
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # fig.suptitle("Day of week comparison")
        # ax1.set_title("Targets (random days)")
        # ax2.set_title("Training prediction")
        # ax3.set_title("Validation prediction")
        #
        # df_week = dfs["df_actuals"]
        # df_week_train = dfs["df_train_pred"].dropna()
        # df_week_validation = dfs["df_validation_pred"].dropna()
        #
        # for day in range(7):
        #     for i in range(0, len(df_week), 24):
        #         if df_week.index[i].dayofweek == day:
        #             ax1.plot(x, df_week.iloc[i : i + 24], label=days[day])
        #             ax1.legend()
        #             break
        #
        #     for j in range(0, len(df_week_train), 24):
        #         if df_week_train.index[j].dayofweek == day:
        #             ax2.plot(x, df_week_train.y[j : j + 24], label=days[day])
        #             ax2.legend()
        #             break
        #
        #     for k in range(0, len(df_week_validation), 24):
        #         if df_week_validation.index[k].dayofweek == day:
        #             ax3.plot(x, df_week_validation.y[k : k + 24], label=days[day])
        #             ax3.legend()
        #             ax3.set_title("Validation prediction")
        #             break

        plt.show()
