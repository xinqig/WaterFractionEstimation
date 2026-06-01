import os
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray
from sklearn.metrics import mean_squared_error, r2_score
# --- Configuration ---
landsat_dir = 'Landsat_reflectance'
modis_dir = 'MODIS_reflectance'
water_body_mask = 'large_water_bodies/large_water_bodies.shp'
validation_date = '2016-10-25'
bands = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "mNDWI1", "mNDWI2", "NDVI"]
crs_wgs84 = 'EPSG:4326'

# --- Load Water Mask ---
water_bodies = gpd.read_file(water_body_mask)

# --- Utilities ---
def assign_crs(dataset):
    if not dataset.rio.crs:
        dataset.rio.write_crs(crs_wgs84, inplace=True)
    return dataset

def mask_water_bodies(dataset, shapefile):
    dataset = assign_crs(dataset)
    return dataset.rio.clip(shapefile.geometry, shapefile.crs, drop=False, invert=True, all_touched=True)

import tensorflow as tf
import gpflow
from gpflow.utilities import set_trainable

def gpflow_sgpr_calibration(modis_vals, landsat_vals, modis_validation, num_inducing=100, return_model=False):
    valid_mask = (~np.isnan(modis_vals)) & (~np.isnan(landsat_vals))
    X_train = modis_vals[valid_mask].reshape(-1, 1).astype(np.float64)
    Y_train = landsat_vals[valid_mask].reshape(-1, 1).astype(np.float64)

    if len(X_train) < 100:
        preds = np.full(modis_validation.shape, np.nan)
        return (preds, None) if return_model else preds

    # Select inducing points
    inducing_points = X_train[np.random.choice(len(X_train), size=min(num_inducing, len(X_train)), replace=False)]

    # Define kernel and model
    kernel = gpflow.kernels.SquaredExponential()
    likelihood = gpflow.likelihoods.Gaussian()

    model = gpflow.models.SVGP(kernel=kernel,
                               likelihood=likelihood,
                               inducing_variable=inducing_points,
                               num_latent_gps=1)

    # Optimization
    training_data = (X_train, Y_train)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss_closure(training_data),
                 variables=model.trainable_variables,
                 options=dict(maxiter=500), method="L-BFGS-B", compile=True)

    # Prediction
    X_pred = modis_validation.reshape(-1, 1).astype(np.float64)
    mean, _ = model.predict_f(X_pred)

    return (mean.numpy().flatten(), model) if return_model else mean.numpy().flatten()


def compute_index(b1, b2):
    with np.errstate(divide='ignore', invalid='ignore'):
        idx = (b1 - b2) / (b1 + b2)
        idx[(b1 + b2) == 0] = np.nan
    return idx

# --- Load and Mask Datasets ---
landsat_datasets = {}
modis_datasets = {}

landsat_files = sorted([f for f in os.listdir(landsat_dir) if f.endswith(".nc")])
modis_files = sorted([f for f in os.listdir(modis_dir) if f.endswith(".nc")])

for l_file, m_file in zip(landsat_files, modis_files):
    date = l_file.split("_")[-1].split(".nc")[0]

    l_ds = xr.open_dataset(os.path.join(landsat_dir, l_file))
    m_ds = xr.open_dataset(os.path.join(modis_dir, m_file))

    landsat_datasets[date] = mask_water_bodies(l_ds, water_bodies)
    modis_datasets[date] = mask_water_bodies(m_ds, water_bodies)

# --- Calibration ---
print("\n--- GPflow Sparse GP Calibration ---")
gpflow_predictions = {}

for band in bands:
    modis_values, landsat_values = [], []

    for date in landsat_datasets:
        if date == validation_date:
            continue
        l_vals = landsat_datasets[date][band].values.flatten()
        m_vals = modis_datasets[date][band].values.flatten()
        mask = (~np.isnan(l_vals)) & (~np.isnan(m_vals))
        landsat_values.extend(l_vals[mask])
        modis_values.extend(m_vals[mask])

    landsat_values = np.array(landsat_values)
    modis_values = np.array(modis_values)
    modis_val = modis_datasets[validation_date][band].values.flatten()
    val_mask = ~np.isnan(modis_val)

    corrected = np.full(modis_val.shape, np.nan)
    predicted, sgpr_model = gpflow_sgpr_calibration(modis_values, landsat_values, modis_val[val_mask], return_model=True)
    corrected[val_mask] = predicted
    gpflow_predictions[band] = corrected.reshape(modis_datasets[validation_date][band].shape)

    # Evaluate model on training data
    train_pred, _ = sgpr_model.predict_f(modis_values.reshape(-1, 1).astype(np.float64))
    rmse = np.sqrt(mean_squared_error(landsat_values, train_pred.numpy().flatten()))
    r2 = r2_score(landsat_values, train_pred.numpy().flatten())
    print(f"[{band}] GPflow SGPR RMSE: {rmse:.4f}, R²: {r2:.4f}")

# --- Compute Indices ---
gpflow_predictions["mNDWI1"] = compute_index(gpflow_predictions["Green"], gpflow_predictions["SWIR1"])
gpflow_predictions["mNDWI2"] = compute_index(gpflow_predictions["Green"], gpflow_predictions["SWIR2"])
gpflow_predictions["NDVI"]   = compute_index(gpflow_predictions["NIR"], gpflow_predictions["Red"])


# --- Save Output ---
np.savez(f"gp_calibrated_modis_{validation_date}.npz", **gpflow_predictions)
