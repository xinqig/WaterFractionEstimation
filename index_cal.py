import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def calculate_mndwi(data):
    ndwi_bands = ['sur_refl_b07_1', 'sur_refl_b04_1']
    modis_pre_mndwi = []
    modis_mndwi = []

    # Determine the number of plots (can be any value)
    num_plots = len(data)
    # Iterate over the data and plot each subplot
    for i in range(num_plots):
        modis_pre_mndwi.append(data[i][ndwi_bands].to_array())
        modis_mndwi.append((modis_pre_mndwi[i][1] - modis_pre_mndwi[i][0]) / (modis_pre_mndwi[i][1] + modis_pre_mndwi[i][0]))
    return modis_mndwi


def calculate_tcw(data):
    ndwi_bands = ['sur_refl_b03_1', 'sur_refl_b04_1',
                  'sur_refl_b01_1', 'sur_refl_b02_1',
                  'sur_refl_b06_1', 'sur_refl_b07_1']
    modis_pre_tcw = []
    modis_tcw = []

    # Determine the number of plots (can be any value)
    num_plots = len(data)
    # Iterate over the data and plot each subplot
    for i in range(num_plots):
        modis_pre_tcw.append(data[i][ndwi_bands].to_array())
        modis_tcw.append(0.0315*modis_pre_tcw[i][0] + 0.2021*modis_pre_tcw[i][1]
                          +0.3102*modis_pre_tcw[i][2] + 0.1594*modis_pre_tcw[i][3]
                          -0.6806*modis_pre_tcw[i][4] -0.6109*modis_pre_tcw[i][5])
    return modis_tcw



def calculate_fwi(data):
    ndwi_bands = ['sur_refl_b04_1', 'sur_refl_b01_1',
                  'sur_refl_b02_1', 'sur_refl_b06_1',
                  'sur_refl_b07_1']
    modis_pre_fwi = []
    modis_fwi = []
    # Determine the number of plots (can be any value)
    num_plots = len(data)
    # Iterate over the data and plot each subplot
    for i in range(num_plots):
        modis_pre_fwi.append(data[i][ndwi_bands].to_array())
        modis_fwi.append(1.7204+171*modis_pre_fwi[i][0]+3*modis_pre_fwi[i][1]
                         -70*modis_pre_fwi[i][2]-45*modis_pre_fwi[i][3]
                         -71*modis_pre_fwi[i][4])
    return modis_fwi


def calculate_ndwi(data):
    ndwi_bands = ['sur_refl_b05_1', 'sur_refl_b02_1']
    modis_pre_ndwi = []
    modis_ndwi = []

    # Determine the number of plots (can be any value)
    num_plots = len(data)
    # Iterate over the data and plot each subplot
    for i in range(num_plots):
        modis_pre_ndwi.append(data[i][ndwi_bands].to_array())
        modis_ndwi.append((modis_pre_ndwi[i][1] - modis_pre_ndwi[i][0]) / (modis_pre_ndwi[i][1] + modis_pre_ndwi[i][0]))
    return modis_ndwi



def calculate_ndvi(data):
    ndwi_bands = ['sur_refl_b01_1', 'sur_refl_b02_1']
    modis_pre_ndvi = []
    modis_ndvi = []

    # Determine the number of plots (can be any value)
    num_plots = len(data)
    # Iterate over the data and plot each subplot
    for i in range(num_plots):
        modis_pre_ndvi.append(data[i][ndwi_bands].to_array())
        modis_ndvi.append((modis_pre_ndvi[i][1] - modis_pre_ndvi[i][0]) / (modis_pre_ndvi[i][1] + modis_pre_ndvi[i][0]))
    return modis_ndvi


def plot_index(data, label):
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111)
    im = ax.imshow(data, cmap='coolwarm', vmax=0)
    cbar = fig.colorbar(im)
    cbar.set_label(label, fontsize=18)  # Set colorbar label fontsize
    cbar.ax.tick_params(labelsize=16)  # Set colorbar tick fontsize


def calculate_AWEI_nsh(data):
    AWEI_nsh_bands = ['sur_refl_b04_1', 'sur_refl_b06_1',
                      'sur_refl_b02_1', 'sur_refl_b07_1']
    bands = []
    AWEI_nsh = []

    # Determine the number of plots (can be any value)
    num_plots = len(data)
    # Iterate over the data and plot each subplot
    for i in range(num_plots):
        bands.append(data[i][AWEI_nsh_bands].to_array())
        AWEI_nsh.append(4*(bands[i][0] - bands[i][1]) - (0.25*bands[i][2] + 2.75*bands[i][3]))
    return AWEI_nsh


def calculate_AWEI_sh(data):
    AWEI_sh_bands = ['sur_refl_b03_1', 'sur_refl_b04_1',
                      'sur_refl_b02_1', 'sur_refl_b06_1', 'sur_refl_b07_1']
    bands = []
    AWEI_sh = []

    # Determine the number of plots (can be any value)
    num_plots = len(data)
    # Iterate over the data and plot each subplot
    for i in range(num_plots):
        bands.append(data[i][AWEI_sh_bands].to_array())
        AWEI_sh.append(bands[i][0] + 2.5 * bands[i][1] 
                        - 1.5*(bands[i][2] + bands[i][3]) 
                        - 0.25 * bands[i][4])
    return AWEI_sh