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

from ..tools import normalize_arcsinh


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


def find_indexes(adata, key_added = 'signal_type'):
    """
    Finds channels of interest for computing bleedthrough.
    :param adata: anndata object
    :param key_added: key, where result vector is added to the adata.var
    :return: a categorical vector in adata.var[f'{key_added}'] 
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


def compute_bleedthr(adata, key = 'signal_type'):
    """
    Computes bleedthrough for data channels.
    :param adata: AnnData object to be processed
    :return: AnnData object with calculated bleedthrough.
    """
    key_in = key
    compens = adata.uns['comp_mat']
    # save original data as layer
    if 'original' not in adata.layers:
        adata.layers['original'] = adata.X

    # Ignore channels 'FSC-H', 'FSC-A', 'SSC-H', 'SSC-A', 'FSC-Width', 'Time'
    if key_in not in adata.var_keys():
        adata = find_indexes(adata)
    #select non other indices
    indexes = np.invert(adata.var[key_in] == 'other')
    
    bleedthrough = np.dot(adata.X[:, indexes], 
                          compens)
    adata.X[:, indexes] = bleedthrough
    return adata


def split_area(adata, key='signal_type', option='area'):
    """
    Methode to filter out height or area data.
    :param adata: AnnData object containing data.
    :param key: key for adata.var where the variable type is stored
    :param option: Switch for choosing 'area' or 'height'.
    :return: AnnData object containing area or height data
    """

    option_key = option
    key_in = key
    
    if option_key not in ['area', 'height', 'other']:
        print(f"{option_key} is not a valid category. Return all.")
        return adata
    #Check if indices for area and height have been computed
    if key_in not in adata.var_keys():
        adata = find_indexes(adata)
    
    index = adata.var[key_in] == option_key
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

# TODO: move function to plotting module
# Plot data. Choose between Area, Height both(default)
def plotdata(adata, 
             key = 'signal_type', 
             normalize = True,
             cofactor = 10, 
             option='',
             save = '',
             **kwargs
             ):
    """
    Creating histogram plot from Anndata object.
    :param adata: AnnData object containing data.
    :param cofactor: float value to normalize with in arcsinh-transform
    :param option: Switch to choose directly between area and height data.
    :param save: Filename to save the shown figure 
    :param kwargs: Passed to :func:`matplotlib.pyplot.savefig`
    """
    option_key = option
    key_in = key
    
    #Check if indices for area and height have been computed
    if key_in not in adata.var_keys():
        adata = find_indexes(adata)
    
    if normalize:
        adata = normalize_arcsinh(adata, cofactor)
    
    if option_key not in ['area', 'height', 'other']:
        print(f"Option {option_key} is not a valid category. Return all.")
        datax = adata.X
        var_names = adata.var_names.values
    else:
        index = adata.var[key_in] == option_key
        datax = adata.X[:, index]
        var_names = adata.var_names[index].values 
    
    if len(var_names)== 0:
        print(f"Option {option_key} led to the selection of 0 variables.\
               Nothing to plot.")
        return
        

    rcParams['figure.figsize'] = (15, 6)

    names = var_names
    number = len(names)

    columns = 3
    rows = math.ceil(number / columns)

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.6)

    for idx in range(number):
        ax = fig.add_subplot(rows, columns, idx + 1)
        sb.distplot(datax[:, names == names[idx]],
                    kde=False, norm_hist=False, 
                    bins=400, ax=ax, 
                    axlabel=names[idx])
    if save !='':
        plt.savefig(save, bbox_inches = 'tight', **kwargs)
    plt.show()
    
    return


def load_mult_files():
    """
    Methode for loading multiple files at once.
    :return: A list of the loaded files.
    """
    username = getpass.getuser()  # current username

    file_dialog = Tk()
    file_dialog.withdraw()

    elements = []
    file_names = filedialog.askopenfilenames(initialdir="/home/%s/SampleData/" % username, 
                                             title="Select file",
                                             filetypes=(("all files", "*.*"), 
                                                        ("fcs files", "*.fcs"),
                                                        ("h5ad files", ".h5ad")))

    for file_name in file_names:
        filename, file_extension = os.path.splitext(file_name)

        if file_extension == '.fcs':
            elements.append(fct.FCMeasurement(ID='FCS-file', 
                                              datafile=file_name))
        elif file_extension == '.h5ad':
            elements.append(ann.read_h5ad(file_name))
        else:
            print('File ' + file_name + ' can not be loaded!')

    return elements
