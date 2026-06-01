import numpy as np
import os
import geopandas as gpd
from shapely.geometry import mapping
import scipy.ndimage

desired_bands = ["sur_refl_b01_1",
                "sur_refl_b02_1",
                "sur_refl_b03_1",
                "sur_refl_b04_1",
                "sur_refl_b05_1",
                "sur_refl_b06_1",
                "sur_refl_b07_1",
                "QC_500m_1"]

def scaled_reflectance(modis_bands,file_list):
    modis_reflectance = []
    all_bands = desired_bands[0:-1]
    for i, filename in enumerate(file_list):
        # Create a new dataset to store the masked and calibrated data
        reflectance = modis_bands[i].copy()

        for band_name in all_bands:
            # Extract the scale factor and add_offset for the current band
            scale_factor = modis_bands[i][band_name].scale_factor
            add_offset = modis_bands[i][band_name].add_offset

            # Apply the scale factor and add_offset to calibrate the data
            calibrated_data = modis_bands[i][band_name].values * scale_factor + add_offset

            # Update the reflectance dataset with the calibrated data
            reflectance[band_name].values = calibrated_data

        # Append the calibrated reflectance dataset to the list
        modis_reflectance.append(reflectance)  
    return modis_reflectance

def clip(file_boundary_path,modis_bands):

    file_boundary = gpd.read_file(file_boundary_path)

    modis_pre_clip_geom = []
    # Check the CRS of the study area extent
    for i in range(len(modis_bands)):
        if not file_boundary.crs == modis_bands[i].rio.crs:
            # If the CRS is not equal, reproject the data
            file_boundary = file_boundary.to_crs(crs=modis_bands[i].rio.crs)
            
    # Clip the data with .rio.clip
    for i in range(len(modis_bands)):
        modis_pre_clip_geom.append(modis_bands[i].rio.clip(file_boundary.geometry.apply(mapping),
                                                                            crs=file_boundary.crs,
                                                                            # Include all pixels even partial pixels
                                                                            all_touched=True,
                                                                            from_disk=True).squeeze())
    return modis_pre_clip_geom

# Reproject data to WGS1984
def reprojection(data, data_info, file_list, projection):
    
    reprojected_data = []
    reprojected_data_info = []

    for i in range(len(file_list)):
        # for band_name in desired_bands:
            # reprojected_data.append(modis_bands[i].copy())
        reprojected_data.append(data[i].rio.reproject(projection))
        reprojected_data_info.append(data_info[i].rio.reproject(projection))
    return reprojected_data, reprojected_data_info


def ideal_quality(modis_bands):
    num_plots = len(modis_bands)
    masked_dataset = []

    # Iterate over the data and apply cloud removal
    for i in range(num_plots):
        # Extract the QC band values
        qc_values = modis_bands[i]['QC_500m_1'].values.astype(np.uint32)

        # Create a mask for ideal quality pixels (bits 0-1 = 00)
        ideal_quality_mask = np.bitwise_and(qc_values, 0b11) == 0b00

        # Create a new dataset to store the masked data
        masked_dataset.append(modis_bands[i].copy())

        for band_name in desired_bands:
                # Apply the mask to the original data to extract ideal quality pixels
                masked_dataset[i][band_name] = modis_bands[i][band_name].where(ideal_quality_mask)

    return masked_dataset

def cloud_removal(data_500, data_info):
    cloud_masked_data = []
    num_plots = len(data_500)
    masked_info = []
    # Iterate over the data and apply cloud removal
    for i in range(num_plots):
        
        qc_trial = data_info[i]['state_1km_1'].values.astype(np.uint32)

        # Create a mask for cloud pixels (bits 10-11 = 10 or 11)
        # cloud_mask = np.bitwise_and(qc_trial, 1 << 10) == 0
        # Extract cloud state bits (bits 0-1)
        # cloud_mask = np.bitwise_and(qc_trial, 3) == 0
        # cloud_mask = np.bitwise_and(qc_trial, 0b111) != 0b001
        cloud_mask = np.bitwise_and(qc_trial, 0b11) != 0b01
        # Resample the 1km cloud mask to match the 500m resolution
        resampled_cloud_mask = scipy.ndimage.zoom(cloud_mask, 2, order=0)

        # Adjust the resampled cloud mask to match data_500 dimensions
        target_shape = data_500[i]['sur_refl_b01_1'].shape

        # Adjust x-axis (rows)
        if resampled_cloud_mask.shape[0] != target_shape[0]:
            resampled_cloud_mask = resampled_cloud_mask[:target_shape[0], :]

        # Adjust y-axis (columns)
        if resampled_cloud_mask.shape[1] != target_shape[1]:
            resampled_cloud_mask = resampled_cloud_mask[:, :target_shape[1]]

        masked_info.append(data_info[i]['num_observations_1km'].where(cloud_mask))
        
        # resampled_info.append(scipy.ndimage.zoom(masked_info[i], 2, order=0))
        
        masked_data = data_500[i].copy()

        for band_name in desired_bands:
            if band_name != 'state_1km_1':
                # Apply the resampled cloud mask to the original data to extract cloud-free pixels
                masked_data[band_name] = data_500[i][band_name].where(resampled_cloud_mask)

        cloud_masked_data.append(masked_data)
    return cloud_masked_data