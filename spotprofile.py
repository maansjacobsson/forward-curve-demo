import numpy as np
from keras.utils import plot_model

from spot_two import SpotProfile as SP

if __name__ == "__main__":
    # Load and parse data, and include calendar features________________________________________________________________
    PATH = "/\\"
    df = SP.load_data(PATH, "DE.csv")

    # Create normalized numpy arrays for the model fitting______________________________________________________________
    ds = SP.create_xarray(df)

    # Create Model______________________________________________________________________________________________________
    model = SP.create_model(ds["x"], ds["y"])

    # Split training and validation data________________________________________________________________________________
    X_train, X_validation, y_train, y_validation = SP.train_validation_split(ds, 0.01)

    # # Max resp. min peak and off-peak values used as input for training the model_____________________________________
    # max_peak = X_train[:, 2].values.max()
    # min_peak = X_train[:, 2].values.min()
    # max_op = X_train[:, 3].values.max()
    # min_op = X_train[:, 3].values.min()
    #
    # # Changes peak input if it is bigger than seen before_____________________________________________________________
    # peak_plus = X_validation.loc[X_validation[:, 2] > max_peak].coords["datetime"].values
    # X_validation.loc[peak_plus, "Peak_mean"] = max_peak
    # # Change peak input if it is smaller than seen before
    # peak_minus = X_validation.loc[X_validation[:, 2] < min_peak].coords["datetime"].values
    # X_validation.loc[peak_minus, "Peak_mean"] = min_peak
    # # Change off-peak input if it is bigger than seen before
    # op_plus = X_validation.loc[X_validation[:, 3] > max_op].coords["datetime"].values
    # X_validation.loc[op_plus, "Off-peak_mean"] = max_op
    # # Change off-peak input if it is smaller than seen before
    # op_minus = X_validation.loc[X_validation[:, 3] < min_op].coords["datetime"].values
    # X_validation.loc[op_minus, "Off-peak_mean"] = min_op

    # Fit model_________________________________________________________________________________________________________
    history = model.fit(
        [
            X_train.values[:, :-2],
            np.tile(X_train.values[:, -2], (12, 1)).transpose(),
            np.tile(X_train.values[:, -1], (12, 1)).transpose(),
        ],
        y_train.values,
        epochs=200,
        batch_size=128,
        verbose=1,
        validation_data=(
            [
                X_validation.values[:, :-2],
                np.tile(X_validation.values[:, -2], (12, 1)).transpose(),
                np.tile(X_validation.values[:, -1], (12, 1)).transpose(),
            ],
            y_validation.values,
        ),
    )

    # Predict/Infer train and validation model results, and un-scale them back__________________________________________
    y_train_pred = model.predict(
        [
            X_train.values[:, :-2],
            np.tile(X_train.values[:, -2], (12, 1)).transpose(),
            np.tile(X_train.values[:, -1], (12, 1)).transpose(),
        ]
    )
    y_validation_pred = model.predict(
        [
            X_validation.values[:, :-2],
            np.tile(X_validation.values[:, -2], (12, 1)).transpose(),
            np.tile(X_validation.values[:, -1], (12, 1)).transpose(),
        ]
    )

    dfs = SP.df_plots(df, ds, y_train, y_train_pred, y_validation, y_validation_pred)
    """
    cce = CategoricalCrossentropy()
    mae = MeanAbsoluteError()
    mse = MeanSquaredError()
    h = holidays.CountryHoliday("DE")

    val_targets = dfs["df_actuals"][
        dfs["df_actuals"].index.isin(dfs["df_validation_pred"].dropna().index)
    ].to_frame()
    val_pred = dfs["df_validation_pred"].dropna().copy()
    val_targets["Holiday"] = np.array(val_targets.index.map(lambda x: x in h)).astype(
        "int8"
    )
    val_pred["Holiday"] = np.array(val_pred.index.map(lambda i: i in h)).astype("int8")

    val_target_h = val_targets.loc[val_targets["Holiday"] == 1]["y"].values
    val_pred_h = val_pred.loc[val_pred["Holiday"] == 1]["y"].values

    val_target_nh = val_targets.loc[val_targets["Holiday"] == 0]["y"].values
    val_pred_nh = val_pred.loc[val_pred["Holiday"] == 0]["y"].values

    # print("Cathegorical cross entropy loss,validation data: ")
    # cce_loss = cce(val_targets, val_pred).numpy()
    # print(cce_loss)

    # print("Mean absolute error,validation data: ")
    mae_loss = mae(val_targets["y"].values, val_pred["y"].values).numpy()
    print(mae_loss)

    # print("Mean absolute error, non-holidays, validation data: ")
    mae_loss_n_h = mae(val_target_nh, val_pred_nh).numpy()
    print(mae_loss_n_h)

    # print("Mean absolute error, holidays, validation data: ")
    mae_loss_h = mae(val_target_h, val_pred_h).numpy()
    print(mae_loss_h)

    print(len(val_pred_h) / 24)

    # print("Mean squared error, validation data: ")
    # mse_loss = mse(val_targets, val_pred).numpy()
    # print(mse_loss)
    """

    # Plotting results.
    SP.make_plots(dfs, df, history)
    plot_model(
        model,
        to_file="/Spotmodel fig/model_architecture_2.0.png",
        show_shapes=True,
    )

    # Evaluating model on data from 1st of Sep 2021 to 1 of Dec 2021
    df_ev = SP.load_data(PATH, "DE_spot_ev.csv")

    # Create normalized numpy arrays for the model fitting
    ds_ev = SP.create_xarray(df_ev)

    # Split training and validation data
    Xd, X_ev, y_train, y_ev = SP.train_validation_split(ds_ev, 0.98)

    y_train_pred_ev = model.predict(
        [
            Xd.values[:, :-2],
            np.tile(Xd.values[:, -2], (12, 1)).transpose(),
            np.tile(Xd.values[:, -1], (12, 1)).transpose(),
        ]
    )
    y_validation_pred_ev = model.predict(
        [
            X_ev.values[:, :-2],
            np.tile(X_ev.values[:, -2], (12, 1)).transpose(),
            np.tile(X_ev.values[:, -1], (12, 1)).transpose(),
        ]
    )

    dfs_ev = SP.df_plots(
        df_ev, ds_ev, y_train, y_train_pred_ev, y_ev, y_validation_pred_ev
    )
    SP.make_plots(dfs_ev, df_ev)
