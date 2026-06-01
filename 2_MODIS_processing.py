import os
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np
import warnings
import rasterio

import open_func
import quality_ctrl
import index_cal
# import wofs
warnings.filterwarnings("ignore")

folder_name = 'MODIS_folder' # Choose file list in the directory
# Bourke_boundary_path = os.path.join("Bourke", "Zoomed_study_area.shp") # Open file boundary
Bourke_boundary_path = 'clipped_boundary/clipped_boundary.shp'
# UTM_projection = "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext" # Sinusoidal projection
WGS84_projection = "EPSG:4326"

# Open all .hdf file in the directory
modis_bands_Terra = open_func.open_raw(folder_name)
modis_bands_info_Terra = open_func.open_info(folder_name)
file_list_Terra = os.listdir(folder_name)

# Pre-processing for scaled reflectance
modis_reflectance_Terra = quality_ctrl.scaled_reflectance(modis_bands_Terra, file_list_Terra)

# Apply cloud masking         
cloud_masked_data_Terra = quality_ctrl.cloud_removal(modis_reflectance_Terra, modis_bands_info_Terra)

# Apply ideal quality
ideal_data_Terra = quality_ctrl.ideal_quality(cloud_masked_data_Terra)

grouped_datasets = {}
for dataset in ideal_data_Terra:
    equator_crossing_date = dataset.attrs.get('EQUATORCROSSINGDATE.2') or dataset.attrs.get('EQUATORCROSSINGDATE.1')

    if equator_crossing_date:
        if equator_crossing_date not in grouped_datasets:
            grouped_datasets[equator_crossing_date] = []
        grouped_datasets[equator_crossing_date].append(dataset)
# Merge datasets within the same group
merged_datasets = []
for date, datasets in grouped_datasets.items():
    if len(datasets) > 1:
        merged_dataset = xr.concat(datasets, dim='y')
        merged_datasets.append(merged_dataset)

# Reproject to WGS84
reprojected_data, reprojected_data_info = quality_ctrl.reprojection(merged_datasets, 
                                                                    modis_bands_info_Terra, 
                                                                    merged_datasets, WGS84_projection)

def clip(file_boundary_path, modis_bands):
    # Load the boundary file
    file_boundary = gpd.read_file(file_boundary_path)

    # Ensure consistent CRS across all MODIS bands
    target_crs = modis_bands[0].rio.crs  # Assume all MODIS bands share the same CRS
    if file_boundary.crs != target_crs:
        file_boundary = file_boundary.to_crs(crs=target_crs)
    
    # Clean boundary geometry to avoid issues
    file_boundary.geometry = file_boundary.geometry.buffer(0)

    # Clip each MODIS band
    modis_pre_clip_geom = []
    for band in modis_bands:
        clipped_band = band.rio.clip(
            file_boundary.geometry.apply(mapping),
            crs=file_boundary.crs,
            all_touched=False,  # Set to False for precise boundary clipping
            from_disk=True
        ).squeeze()
        modis_pre_clip_geom.append(clipped_band)
    
    return modis_pre_clip_geom

# Clip to study area
clipped_ideal_data_Terra = clip(Bourke_boundary_path,reprojected_data)
clipped_dataset = []
max_value = clipped_ideal_data_Terra[0]['sur_refl_b07_1'].max().item()
for i in range(len(clipped_ideal_data_Terra)):
    clipped_ideal_data_Terra[i] = clipped_ideal_data_Terra[i].where(clipped_ideal_data_Terra[i] != max_value, np.nan)

filtered_data = []
for da in clipped_ideal_data_Terra:
    valid_ratio = np.isfinite(da['sur_refl_b07_1']).sum() / da['sur_refl_b07_1'].size
    if valid_ratio >= 0.9:
        filtered_data.append(da)

print(f"Kept {len(filtered_data)} out of {len(clipped_ideal_data_Terra)} images (≥90% valid pixels)")

for data_array in filtered_data:
    # Extract the date attribute
    date = data_array.attrs['EQUATORCROSSINGDATE.2'] or data_array.attrs['EQUATORCROSSINGDATE.1']

    # Create a filename for the NetCDF file
    filename = f"modis_bourke_{date}.nc"
    
    # Save each DataArray to a separate NetCDF file
    data_array.to_netcdf(filename)
    
open_func.plot_false_new(filtered_data)

fig = open_func.plot_false(clipped_ideal_data_Terra)
fig.savefig("modis_false_colour_plots.png", dpi=300, bbox_inches='tight')