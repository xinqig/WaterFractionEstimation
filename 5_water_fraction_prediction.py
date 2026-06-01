import os
import xarray as xr
import matplotlib.pyplot as plt
import re
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from datetime import datetime
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import geopandas as gpd
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import pickle
warnings.filterwarnings("ignore")

validation_index = 0  # Change this index to select different validation sets (0 to 5)

with open(f"svgp_100_pts_val_idx_{validation_index}.pkl", "rb") as f:
    model = pickle.load(f)

data_dates = ['2016-10-25','2022-09-24', '2022-10-10', '2022-11-11', '2022-12-13', '2022-12-29']
loaded = np.load(f"gp_calibrated_modis_{data_dates[validation_index]}.npz")
predictions = {key: loaded[key] for key in loaded.files}

variables_to_use = ["SWIR1", "SWIR2", "Blue", "Red", "mNDWI1", "mNDWI2", "NDVI"]
validation_feature_list = []
for band in variables_to_use:
    # Flatten each band (assuming each prediction is in a 2D array)
    band_data = predictions[band].astype(np.float64).flatten()
    validation_feature_list.append(band_data)

# Stack them column-wise: each column is one band, shape becomes (n_pixels, n_bands)
X_validation_index_multi = np.column_stack(validation_feature_list)

# Apply the same validation mask (assuming it applies to the flattened arrays)
X_validation_index_valid = X_validation_index_multi[validation_mask]
    
# Apply the same validation mask (assuming it applies to the flattened arrays)
y_validation_filtered = y_validation_data.reshape(-1,1).astype(np.float64)

# Predict the mean and variance 
y_mean, y_var = model.predict_y(X_validation_index_valid)

# Clip predictions: values > 1 set to 1, values < 0 set to 0
y_pred = tf.clip_by_value(y_mean, 0.0, 1.0)

ypred_np = y_pred.numpy()
# ypred_1d = ypred_np.flatten()
mask_val = ~np.isnan(y_validation_filtered)
mask_pred = ~np.isnan(ypred_np)
notnan_mask = mask_val & mask_pred
y_val_filtered = y_validation_filtered[notnan_mask]
y_pred_filtered = y_pred[notnan_mask]

rmse = root_mean_squared_error(y_val_filtered, y_pred_filtered)
mae = mean_absolute_error(y_val_filtered, y_pred_filtered)
r2 = r2_score(y_val_filtered, y_pred_filtered)
r, p_value = pearsonr(y_val_filtered, y_pred_filtered)

print(f"RMSE: {rmse:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"R²: {r2:.4f}")
print(f"Pearson's r: {r:.4f}")

land_masked_ypred = ypred_np.copy()

# Create a final output array for validation predictions filled with NaNs
final_pred = np.full(validation_mask.shape, np.nan)
final_pred[validation_mask]=np.squeeze(land_masked_ypred)

valid_mask = np.squeeze(~np.isnan(final_pred))

# For the pixels where prediction is missing (i.e. validation_mask is False),
# fill in with the previous day's values
final_pred[~valid_mask] = np.squeeze(y_validation_raw[~valid_mask])

# Flatten and filter out NaNs for both actual and predicted values
y_validation_flat = y_validation_raw.flatten()
y_pred_combined_flat = final_pred.flatten()

# Mask to filter out NaNs in both arrays
mask_combined = ~np.isnan(y_validation_flat) & ~np.isnan(y_pred_combined_flat)
y_val_final = y_validation_flat[mask_combined]
y_pred_final = y_pred_combined_flat[mask_combined]

rmse_final = root_mean_squared_error(y_val_final, y_pred_final)
mae_final = mean_absolute_error(y_val_final, y_pred_final)
r2_final = r2_score(y_val_final, y_pred_final)

comb_r, p_value = pearsonr(y_val_final, y_pred_final)

print("Combined Metrics:")
print(f"Combined RMSE: {rmse_final:.4f}")
print(f"Combined Pearson's r: {comb_r:.4f}")

import matplotlib.colors as mcolors

cmap_flood = mcolors.LinearSegmentedColormap.from_list(
    "flood_cmap", 
    [(0.0, "#EAF1FA"),    # Light grey (dry)
     (0.3, "#62A2DD"),    # Light blue (emergent water)
     (1.0, "#053AAC")]    # Dark blue (deep/full water)
)

# Use lat/lon coordinates from the dataset directly
lat = MODIS_mNDWI['Blue'].coords['y']
lon = MODIS_mNDWI['Blue'].coords['x']

# Reshape the predicted array to match the spatial shape (y, x)
y_pred_reshaped = y_pred_combined_flat.reshape(len(lat), len(lon))

# Wrap in a DataArray with geocoordinates
predicted_water_da = xr.DataArray(
    y_pred_reshaped,
    dims=("y", "x"),
    coords={"y": lat, "x": lon},
    # name="Predicted Water Fraction"
    name="water_fraction"
)

predicted_water_da.to_netcdf(f'water_fraction_{data_dates[validation_index]}.nc')


