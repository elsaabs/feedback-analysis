from unittest import result

import numpy as np
import pandas as pd
import pyreadr
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import ttest_rel

#########################
#Create lagged matrix
#########################
def build_lagged_matrix(x, y=None, lag=10):
    n = len(x)
    rows = n - lag

    X_lags = np.column_stack(
        [x[lag - l : n - l] for l in range(1, lag + 1)]
    )

    if y is not None:
        Y_lags = np.column_stack(
            [y[lag - l : n - l] for l in range(1, lag + 1)]
        )
        design = np.hstack([X_lags, Y_lags])
    else:
        design = X_lags

    target = x[lag:n]

    return design, target

####################
#Build neural network
####################
def build_nn(input_dim, hidden_layers=(8,4), lr=0.01):
    model = keras.Sequential()
    
    model.add(layers.Dense(hidden_layers[0],
                           activation="relu",
                           input_shape=(input_dim,)))
    
    for units in hidden_layers[1:]:
        model.add(layers.Dense(units, activation="relu"))
        
    model.add(layers.Dense(1))
    
    model.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(learning_rate=lr)
    )
    
    return model


######################################################
#Compute Jacobian of NN model at reference input
######################################################

def compute_block_jacobian(model, X_ref, lag):
    """
    Compute lag-aggregated local sensitivities
    evaluated at reference input X_ref (1D array).
    """

    x_tf = tf.convert_to_tensor(X_ref.reshape(1, -1), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        y_pred = model(x_tf)

    grad = tape.gradient(y_pred, x_tf).numpy().flatten()

    # First lag entries = own lags
    own_block = grad[:lag]

    # Remaining entries = cross lags
    cross_block = grad[lag:]

    return np.mean(own_block), np.mean(cross_block)

########################################
#Time series cross validated  NN Granger
########################################


def cv_nn_granger_ts(x, y,
                     lag=10,
                     hidden_univ=(4,2),
                     hidden_biv=(8,4),
                     n_splits=5,
                     epochs=100,
                     batch_size=32,
                     seed=5):

    tf.keras.utils.set_random_seed(seed)

    #Prepare lagged matrices for univariate (resctricted) and bivariate (full) models
    X_univ, y_univ = build_lagged_matrix(x, None, lag)
    X_biv, y_biv   = build_lagged_matrix(x, y, lag)

    # TimeSeriesSplit for cross-validation: ensure future data is not used to predict past
    tscv = TimeSeriesSplit(n_splits=n_splits)

    mse_univ = []
    mse_biv = []
    #For each fold, train both models and evaluate on the test set
    for fold_num, (train_index, test_index) in enumerate(tscv.split(X_univ), start=1):
        print(f"\n--- Fold {fold_num} ---")
        print(f"Train indices: {train_index[0]}..{train_index[-1]}, Test indices: {test_index[0]}..{test_index[-1]}")

        
        # Split into train/test
        X_train_u, X_test_u = X_univ[train_index], X_univ[test_index]
        y_train_u, y_test_u = y_univ[train_index], y_univ[test_index]

        X_train_b, X_test_b = X_biv[train_index], X_biv[test_index]
        y_train_b, y_test_b = y_biv[train_index], y_biv[test_index]

        
        #Further split training set into train/validation for early stopping during NN training
        # Validation split from training set (last 20%)
        val_split = int(0.8 * len(X_train_u))

        X_tr_u, X_val_u = X_train_u[:val_split], X_train_u[val_split:]
        y_tr_u, y_val_u = y_train_u[:val_split], y_train_u[val_split:]

        X_tr_b, X_val_b = X_train_b[:val_split], X_train_b[val_split:]
        y_tr_b, y_val_b = y_train_b[:val_split], y_train_b[val_split:]

        print(f"X_tr_u shape: {X_tr_u.shape}, X_val_u shape: {X_val_u.shape}")
        tf.keras.backend.clear_session()

        #Build and train NN for both models with early stopping based on validation loss (prevents overfitting)
        # Restricted model: predicts x using only its own lags
        model_u = build_nn(X_tr_u.shape[1], hidden_univ) #build_nn() builds a fully connected feedforward network
        model_u.fit(
            X_tr_u, y_tr_u,
            validation_data=(X_val_u, y_val_u),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True)]
        )
        
        print("Restricted model training complete")

        # Full model: predicts x using its own lags and lags of y
        model_b = build_nn(X_tr_b.shape[1], hidden_biv)
        model_b.fit(
            X_tr_b, y_tr_b,
            validation_data=(X_val_b, y_val_b),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True)]
        )
        print("Full model training complete")

        
        # Test MSE
        #Predict on test set and calculate MSE for both models
        pred_u = model_u.predict(X_test_u, verbose=0).flatten()
        pred_b = model_b.predict(X_test_b, verbose=0).flatten()

        mse_univ.append(np.mean((y_test_u - pred_u)**2))
        mse_biv.append(np.mean((y_test_b - pred_b)**2))

    mse_r = np.mean(mse_univ)
    mse_f = np.mean(mse_biv)


    #Compare MSE predictions of restricted vs full model
    delta_mse = mse_r - mse_f

    # Paired t-test across CV folds
    t_stat, p_value = ttest_rel(mse_univ, mse_biv)

    # --- Compute NN Jacobian at mean state of last fold ---
    X_ref = np.mean(X_tr_b, axis=0)

    J_xx, J_xy = compute_block_jacobian(model_b, X_ref, lag)


    print("\n=== Final Results ===")
    print(f"MSE restricted (avg over folds): {mse_r:.5f}")
    print(f"MSE full (avg over folds): {mse_f:.5f}")
    print(f"Delta MSE: {delta_mse:.5f}")
    print(f"Granger causality detected: {delta_mse > 0}")
    print(f"Own sensitivity (J_own): {J_xx}")
    print(f"Cross sensitivity (J_cross): {J_xy}")

    return {
        "mse_restricted": mse_r,
        "mse_full": mse_f,
        "delta_mse": delta_mse,
        #Granger: If including y lags improves prediction of x, then mse_f < mse_r, so delta_mse > 0 indicates Granger causality
        "granger": delta_mse > 0,
        "J_own": J_xx,
        "J_cross": J_xy
    }



########################################################################
# Function to: read in the data
# 	      run NN granger
#	      create table with results from NN granger and Jacob matrix
########################################################################

def NL_Granger_test(obs_file, param_grid_file, J_star_list_file, base_dir="."):

    # Load data
    obs_all = pd.read_csv(os.path.join(base_dir, obs_file))
    param_grid = pd.read_csv(os.path.join(base_dir, param_grid_file))

    # Load jacobian list
    Jstar_in = pd.read_csv(os.path.join(base_dir, J_star_list_file))
    # Reconstruct list of matrices
    J_star_list = []
    for mid in Jstar_in['matrix_id'].unique():
        # Select rows for this matrix
        sub = Jstar_in[Jstar_in['matrix_id'] == mid]
        
        # Find max row and col to know matrix shape
        n_rows = sub['row'].max()
        n_cols = sub['col'].max()
        
        # Initialize empty matrix
        mat = np.zeros((n_rows, n_cols))
        
        # Fill matrix
        for _, r in sub.iterrows():
            mat[int(r['row'])-1, int(r['col'])-1] = r['value']
        
        J_star_list.append(mat)

    # Test
    #print(f"Number of matrices: {len(J_star_list)}")
    #print("First matrix:\n", J_star_list[0])

    param_set_num = obs_all["param_set"].unique()
    
    #Decide on what lags to test
    lag_in_list = [10, 20]

    results = []

    for lag_in in lag_in_list:
        print("Lag:", lag_in)

        for i in param_set_num:

            ts_data = obs_all[obs_all["param_set"] == i]

            X_series = ts_data["Xobs"].values
            Y_series = ts_data["Yobs"].values

            result_XY = cv_nn_granger_ts(
                x=X_series,
                y=Y_series,
                lag=lag_in
            )

            result_YX = cv_nn_granger_ts(
                x=Y_series,
                y=X_series,
                lag=lag_in
            )

            J = J_star_list[i-1]  # adjust indexing if needed

            Jac_x_to_y = round(J[1,0], 3)
            Jac_y_to_x = round(J[0,1], 3)
            Jac_x_to_x = round(J[0,0], 3)
            Jac_y_to_y = round(J[1,1], 3)

            feedback_gain = round(
                (J[0,1] * J[1,0]) / (J[0,0] * J[1,1]),
                3
            )

            #Extract NN jacobian values for feedback gain calculation
            J_xx_nn = result_XY["J_own"]
            J_xy_nn = result_XY["J_cross"]
            J_yy_nn = result_YX["J_own"]
            J_yx_nn = result_YX["J_cross"]

            # NN feedback gain
            if J_xx_nn != 0 and J_yy_nn != 0:
                feedback_gain_nn = (J_xy_nn * J_yx_nn) / (J_xx_nn * J_yy_nn)
            else:
                feedback_gain_nn = np.nan
                
            row = {
                "param_set": i,
                "lag_in": lag_in,
                "Jac_x_to_x": Jac_x_to_x,
                "Jac_y_to_y": Jac_y_to_y,
                "Jac_x_to_y": Jac_x_to_y,
                "Jac_y_to_x": Jac_y_to_x,
                "feedback_gain": feedback_gain,
                "x_to_y_NN": int(result_XY["granger"]),
                "y_to_x_NN": int(result_YX["granger"]),
                "J_xx_NN": J_xx_nn,
                "J_xy_NN": J_xy_nn,
                "J_yx_NN": J_yx_nn,
                "J_yy_NN": J_yy_nn,
                "feedback_gain_NN": feedback_gain_nn
            }

            # Add parameter grid info
            for col in param_grid.columns:
                row[col] = param_grid.loc[i-1, col]

            results.append(row)

    df = pd.DataFrame(results)

    return df


#############
#Actual run
#############

#############
#Linear
#############

# df_L = NL_Granger_test(
#     "synthetic_dataset_multiple_paramsets_L.csv",
#     "param_grid_L.csv",
#     "J_star_list_L_long.csv"
# )
# df_L.to_csv("output_NL_Granger_L.csv", index=False)


#############
#Multiplicative
#############
# df_M = NL_Granger_test(
#     "synthetic_dataset_multiple_paramsets_M.csv",
#     "param_grid_M.csv",
#     "J_star_list_M_long.csv"
# )

# df_M.to_csv("output_NL_Granger_M.csv", index=False)

#############
#Density dependent
#############
# df_D = NL_Granger_test(
#     "synthetic_dataset_multiple_paramsets_D.csv",
#     "param_grid_D.csv",
#     "J_star_list_D_long.csv"
# )
# df_D.to_csv("output_NL_Granger_D.csv", index=False)



#ELSA TEST

df_L_Elsa = NL_Granger_test(
    "synthetic_dataset_multiple_paramsets_L_testELSA.csv",
    "param_grid_L_testELSA.csv",
    "J_star_list_L_long_testELSA.csv"
)
df_L_Elsa.to_csv("output_NL_Granger_L_testELSA.csv", index=False)