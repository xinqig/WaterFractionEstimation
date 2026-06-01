import os
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray as rxr
import geopandas as gpd
import earthpy.plot as ep
from shapely.geometry import mapping, box
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import numpy as np

def open_info(folder_name):
    file_list = os.listdir(folder_name) 
    size = len(file_list)
    modis = []
    path = []
    desired_bands = ["num_observations_1km",
                     "granule_pnt_1",
                     "state_1km_1"]
    # Open MODIS files
    for i in range(size):
        path.append(os.path.join(folder_name,file_list[i]))
        modis.append(rxr.open_rasterio(path[i],
                                    masked=True,
                                    variable=desired_bands).squeeze())
    return modis

def open_raw(folder_name):
    
    file_list = os.listdir(folder_name) 
    size = len(file_list)
    # modis = []
    modis_bands = []
    path = []

    desired_bands = ["sur_refl_b01_1",
                    "sur_refl_b02_1",
                    "sur_refl_b03_1",
                    "sur_refl_b04_1",
                    "sur_refl_b05_1",
                    "sur_refl_b06_1",
                    "sur_refl_b07_1",
                    "QC_500m_1"]
    
    # Open MODIS files
    for i in range(size):
        path.append(os.path.join(folder_name,file_list[i]))
        # modis.append(rxr.open_rasterio(path[i], masked=True))
        modis_bands.append(rxr.open_rasterio(path[i],
                                             masked=True,
                                             variable=desired_bands).squeeze())

    return modis_bands

def plot_false(modis_no):
    modis_rgb_xr = []

    rgb_bands = ['sur_refl_b05_1',
                'sur_refl_b02_1',
                'sur_refl_b04_1']

    # Determine the number of plots (can be any value)
    num_plots = len(modis_no)

    # Calculate the number of rows and columns for the subplots grid
    num_rows = int(num_plots**0.5)  # Square root rounded down
    num_cols = (num_plots + num_rows - 1) // num_rows

    # Create the subplots dynamically
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20,15))

    # Flatten the axes if necessary
    axes = axes.flatten()

    # Iterate over the data and plot each subplot
    for i, ax in enumerate(axes[:num_plots]):
        modis_rgb_xr.append(modis_no[i][rgb_bands].to_array())
        if 'EQUATORCROSSINGDATE.2' in modis_no[i].attrs:
            ep.plot_rgb(modis_rgb_xr[i].values,
                        rgb=[0, 1, 2],
                        ax=ax,
                        title=modis_no[i].attrs['EQUATORCROSSINGDATE.2'])
        else:
            ep.plot_rgb(modis_rgb_xr[i].values,
                        rgb=[0, 1, 2],
                        ax=ax,
                        title=modis_no[i].attrs['EQUATORCROSSINGDATE.1'])

    # Remove any empty subplots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    return plt.show()


def plot_false_new(modis_no):
    modis_rgb_xr = []

    rgb_bands = ['sur_refl_b05_1',
                 'sur_refl_b02_1',
                 'sur_refl_b04_1']

    num_plots = len(modis_no)
    num_rows = int(num_plots ** 0.5)
    num_cols = (num_plots + num_rows - 1) // num_rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))
    axes = axes.flatten()

    for i, ax in enumerate(axes[:num_plots]):
        modis_rgb_xr.append(modis_no[i][rgb_bands].to_array())
        img = modis_rgb_xr[i].values

        # Handle all-NaN or constant images
        if np.isnan(img).all():
            img = np.zeros_like(img)  # blank but safe
        elif np.nanmax(img) == np.nanmin(img):
            img = np.zeros_like(img)  # also safe for flat constant arrays

        # Get appropriate title
        title = modis_no[i].attrs.get('EQUATORCROSSINGDATE.2') or modis_no[i].attrs.get('EQUATORCROSSINGDATE.1') or f"Image {i}"

        # Plot the RGB image
        try:
            ep.plot_rgb(img, rgb=[0, 1, 2], ax=ax, title=title, stretch=True)
        except Exception as e:
            ax.set_title(f"Failed to plot\n{title}", fontsize=10)
            ax.axis('off')
            print(f"Plot failed at index {i}: {e}")

    # Remove extra axes
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig  # return the figure so you can save it externally

def plot_one_false(data):
    
    fig, ax = plt.subplots(figsize=(10, 10))
    swir_nir_g = ['sur_refl_b05_1', 'sur_refl_b02_1', 'sur_refl_b04_1']
    false_fig = (data[swir_nir_g].to_array())

    ep.plot_rgb(false_fig.values, rgb=[0, 1, 2], title=false_fig[0].attrs['EQUATORCROSSINGDATE.2'], ax=ax)

    return plt.show()

def plot_ndwi(modis_bands):
    ndwi_bands = ['sur_refl_b02_1', 'sur_refl_b04_1']
    modis_pre_ndwi = []
    modis_ndwi = []

    # Determine the number of plots (can be any value)
    num_plots = len(modis_bands)

    # Calculate the number of rows and columns for the subplots grid
    num_rows = int(num_plots**0.5)  # Square root rounded down
    num_cols = (num_plots + num_rows - 1) // num_rows

    # Create the figure and define gridspec
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(num_rows, num_cols + 1, width_ratios=[1] * num_cols + [0.05])

    # Iterate over the data and plot each subplot
    for i in range(num_plots):
        row = i // num_cols
        col = i % num_cols

        ax = fig.add_subplot(gs[row, col])
        modis_pre_ndwi.append(modis_bands[i][ndwi_bands].to_array())
        modis_ndwi.append((modis_pre_ndwi[i][1] - modis_pre_ndwi[i][0]) / (modis_pre_ndwi[i][1] + modis_pre_ndwi[i][0]))
        # Plot the NDWI image
        im = ax.imshow(modis_ndwi[i], cmap='coolwarm', vmin=-1, vmax=1)
        ax.axis('off')
        if 'EQUATORCROSSINGDATE.2' in modis_bands[i].attrs:
            ax.set_title(modis_bands[i].attrs['EQUATORCROSSINGDATE.2'], fontsize=20)
        else:
            ax.set_title(modis_bands[i].attrs['EQUATORCROSSINGDATE.1'], fontsize=20)

    # Add colorbar
    cax = fig.add_subplot(gs[:, -1])  # Position the colorbar to span all rows
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('NDWI', fontsize=18)  # Set colorbar label fontsize
    cbar.ax.tick_params(labelsize=16)  # Set colorbar tick fontsize

    # Adjust the layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the rect values as needed to create space for the colorbar

    return plt.show()
