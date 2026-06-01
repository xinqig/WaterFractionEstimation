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
# Paths
modis_indices_folder = 'Landsat_wofs'  # Folder containing all indices

wo_folder = 'WOs_Bourke_clipped_bounds'
# wo_folder = 'Bourke_two_pairs_WOs'
date_pattern = r'\d{4}-\d{2}-\d{2}'  # Pattern to extract date from filenames
# Select the validation pair (0-5)
validation_index = 4

water_body_mask = 'large_water_bodies/large_water_bodies.shp'
crs_wgs84 = 'EPSG:4326'

def assign_crs(dataset):
    if not dataset.rio.crs:
        dataset.rio.write_crs(crs_wgs84, inplace=True)
    return dataset
def mask_water_bodies(dataset, shapefile):
    dataset = assign_crs(dataset)
    return dataset.rio.clip(shapefile.geometry, shapefile.crs, drop=False, invert=True, all_touched=True)
water_bodies = gpd.read_file(water_body_mask)

# Get file lists
modis_files = os.listdir(modis_indices_folder)
wo_files = os.listdir(wo_folder)

# Function to get date from filename
def get_date_from_filename(filename):
    date_match = re.search(date_pattern, filename)
    return datetime.strptime(date_match.group(0), '%Y-%m-%d') if date_match else None

data_pairs = []
for wo_file in wo_files:
    wo_date = get_date_from_filename(wo_file)
    if not wo_date:
        continue  # Skip if date not found in WO file

    # Find MODIS file with date within one day
    modis_file = next((f for f in modis_files if abs((get_date_from_filename(f) - wo_date).days) <= 1), None)
    if not modis_file:
        continue  # Skip if no matching MODIS file found

    data_pairs.append((wo_file, modis_file, wo_date))

# Initialize lists to store data
X_train_data = []
y_train_data = []

# Initialize lists for validation data
X_validation_data = []
y_validation_data = []

# classified_data = xr.open_dataset('multiyear_water_summary_resampled.nc')
variables_to_use = ["SWIR1", "SWIR2", "Blue", "Red", "mNDWI1", "mNDWI2", "NDVI"]
# variables_to_use = [ "mNDWI1"]

# Loop through each pair of WO and MODIS files
for i, (wo_file, modis_file, wo_date) in enumerate(data_pairs):
    # Load WO file (target variable)
    resampled_wos = xr.open_dataset(os.path.join(wo_folder, wo_file))
    wo_values = resampled_wos['__xarray_dataarray_variable__'].values.flatten()

    # Load MODIS indices file (7 features)
    MODIS_mNDWI = xr.open_dataset(os.path.join(modis_indices_folder, modis_file))
    MODIS_mNDWI_masked = mask_water_bodies(MODIS_mNDWI, water_bodies)
    # Flatten each MODIS data variable and stack them into a 2D array
    modis_features = np.column_stack([MODIS_mNDWI_masked[var].values.flatten() for var in variables_to_use])

    # Update the mask: Only include mixed areas (class == 1)
    nonan_mask = (~np.isnan(modis_features).any(axis=1)) & (~np.isnan(wo_values))
    wo_mask = (wo_values > 0)
    mask = nonan_mask & wo_mask
    
    if i == validation_index - 1:
        prev_day_mask = mask.copy()
        prev_day_y = wo_values
    # Check if this is the validation pair
    if i == validation_index:
        # Store the validation data separately
        modis_val = xr.open_dataset(os.path.join('MODIS_wofs', f'modis_wofs_{wo_date.strftime("%Y-%m-%d")}.nc'))
        # Flatten each MODIS data variable and stack them into a 2D array
        modis_val_features = np.column_stack([modis_val[var].values.flatten() for var in variables_to_use])
        validation_mask = nonan_mask & (~np.isnan(modis_val_features).any(axis=1))
        X_validation_data = modis_val_features[validation_mask]
        y_validation_data = wo_values[validation_mask]
        y_validation_array = resampled_wos.copy()
        y_validation_raw = wo_values        
        continue

    # Collect data for overall dataset (for training)
    X_train_data.append(modis_features[mask])
    y_train_data.append(wo_values[mask])

# Convert lists to numpy arrays
X_train = np.concatenate(X_train_data)
y_train = np.concatenate(y_train_data)

X_validation = np.array(X_validation_data)
y_validation = np.array(y_validation_data)

X_train_index = X_train.astype(np.float64)
y_train_reshaped = y_train.reshape(-1,1).astype(np.float64)

# Define the main kernel
main_kernel = gpflow.kernels.Matern32()

combined_kernel = main_kernel

likelihood=gpflow.likelihoods.StudentT()

from sklearn.cluster import KMeans

M = 100
kmeans = KMeans(n_clusters=M).fit(X_train_index)
Z = kmeans.cluster_centers_

# Define the SVGP model
model = gpflow.models.SVGP(
    kernel=main_kernel,
    likelihood=likelihood,
    inducing_variable=Z,
    num_latent_gps=1  
)

data=(X_train_index, y_train_reshaped)

# Minimize using Scipy
optimizer = gpflow.optimizers.Scipy()

optimizer.minimize(
    lambda: - model.elbo(data), 
    model.trainable_variables,
)

model_filename = f'svgp_{M}_pts_all.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
