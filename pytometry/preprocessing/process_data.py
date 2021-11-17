"""
Author:     Thomas Ryborz
ICB         HelmholtzZentrum münchen
Date:       15.01.2020

Module to create a compansation matrix from spillover data.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import rcParams
import getpass
import os.path
from tkinter import *
from tkinter import filedialog
import FlowCytometryTools as fct
import anndata as ann
import math


def create_spillover_mat(fcsdata, key = '$SPILLOVER'):
    """
    Creates a spillover matrix from meta data of an .fcs file
    :param fcsdata: Meta data from .fcs file
    :return: Spillover matrix as panda dataframe.
    """
    spillover = fcsdata.meta[key].split(",")
    num_col = int(spillover[0])
    channel_names = spillover[1:(int(spillover[0]) + 1)]
    channel_data = fcsdata.meta['_channels_']

    if '$PnS' in channel_data:
        channel_renames = [str(channel_data['$PnS'][channel_data['$PnN'] == name][0]) for name in channel_names]
    else:
        channel_renames = channel_names

    spill_values = np.reshape([float(inp) for inp in spillover[(int(spillover[0]) + 1):]], [num_col, num_col])
    spill_df = pd.DataFrame(spill_values, columns=channel_renames)
    return spill_df


def create_comp_mat(spillmat, relevant_data=''):
    """
    Creates a compensation matrix from a spillover matrix.
    :param spillmat: Spillover matrix as panda dataframe.
    :param relevant_data: A list of channels for customized selection.
    :return: Compensation matrix as panda dataframe.
    """
    if relevant_data == '':
        comp_mat = np.linalg.inv(spillmat)
        compens = pd.DataFrame(comp_mat, columns=list(spillmat.columns))
    else:
        comp_mat = np.linalg.inv(spillmat)
        compens = pd.DataFrame(comp_mat, columns=relevant_data)

    return compens


def find_indexes(adata):
    """
    Finds channels of interest for computing bleedthrough.
    :param adata: anndata object
    :return: Array of indexes.
    """
    index = adata.var.index
    index_array = []

    for item in index:
        if item.endswith('-A') and not item.count('SC-'):
            index_array.append('area')
        elif item.endswith('-H') and not item.count('SC-'):
            index_array.append('height')
        else:
            index_array.append('other')

    adata.var['signal_type'] = pd.Categorical(index_array)
    return adata


def compute_bleedthr(adata):
    """
    Computes bleedthrough for data channels.
    :param adata: AnnData object to be processed
    :return: AnnData object with calculated bleedthrough.
    """
    compens = adata.uns['comp_mat']
    # save original data as layer
    if 'original' not in adata.layers:
        adata.layers['original'] = adata.X

    # Ignore channels 'FSC-H', 'FSC-A', 'SSC-H', 'SSC-A', 'FSC-Width', 'Time'
    if 'signal_type' not in adata.var_keys():
        adata = find_indexes(adata)
    #select non other indices
    indexes = np.invert(adata.var['signal_type'] == 'other')
    
    bleedthrough = np.dot(adata.X[:, indexes], 
                          compens)
    adata.X[:, indexes] = bleedthrough
    return adata


def split_area(adata, option='area'):
    """
    Methode to filter out height or area data.
    :param adata: AnnData object containing data.
    :param option: Switch for choosing 'area' or 'height'.
    :return: AnnData object containing area or height data
    """

    option_key = option
    if option_key not in ['area', 'height', 'other']:
        print(f"{option_key} is not a valid category. Return all.")
        return adata
    #Check if indices for area and height have been computed
    if 'signal_type' not in adata.var_keys():
        adata = find_indexes(adata)
    
    index = adata.var['signal_type']==option_key
    non_idx = np.flatnonzero(np.invert(index))
    #merge non-idx entries in data matrix with obs
    non_cols = adata.var_names[non_idx].values
    for idx, colname in enumerate(non_cols):
        adata.obs[colname] = adata.X[:,non_idx[idx]].copy()
    
    #create new anndata object
    adataN = ann.AnnData(X = adata.X[:, np.flatnonzero(index)], 
                                obs = adata.obs, 
                            
                                uns = adata.uns)
    adataN.var_names = adata.var_names[index].values 
    return adataN

# TODO: adapt index choice and move function to plotting module
# Plot data. Choose between Area, Height both(default)
def plotdata(adata, option=''):
    """
    Creating scatterplot from Anndata object.
    :param adata: AnnData object containing data.
    :param option: Switch to choose directly between area and height data.
    """
    if option == 'area':
        index = find_indexes(adata, option='area')
        datax = adata.X[:, index]

    elif option == 'height':
        index = find_indexes(adata, option='height')
        datax = adata.X[:, index]
    else:
        index = find_indexes(adata, option='')
        datax = adata.X[:, index]

    rcParams['figure.figsize'] = (15, 6)

    names = adata.var_names[index]
    number = len(names)

    columns = 3
    rows = math.ceil(number / columns)

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.6)

    for index in range(0, number, 1):
        ax = fig.add_subplot(rows, columns, index + 1)
        sb.distplot(np.arcsinh(datax[:, names == names[index]] / 10),
                    kde=False, norm_hist=False, bins=400, ax=ax, axlabel=names[index])

    plt.show()


def load_mult_files():
    """
    Methode for loading multiple files at once.
    :return: A list of the loaded files.
    """
    username = getpass.getuser()  # current username

    file_dialog = Tk()
    file_dialog.withdraw()

    elements = []
    file_names = filedialog.askopenfilenames(initialdir="/home/%s/SampleData/" % username, title="Select file",
                                             filetypes=(("all files", "*.*"), ("fcs files", "*.fcs"),
                                                        ("h5ad files", ".h5ad")))

    for file_name in file_names:
        filename, file_extension = os.path.splitext(file_name)

        if file_extension == '.fcs':
            elements.append(fct.FCMeasurement(ID='FCS-file', datafile=file_name))
        elif file_extension == '.h5ad':
            elements.append(ann.read_h5ad(file_name))
        else:
            print('File ' + file_name + ' can not be loaded!')

    return elements
