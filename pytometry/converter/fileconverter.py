"""
Author:     Thomas Ryborz
ICB         HelmholtzZentrum münchen
Date:       15.01.2020

Fileconverter for .fcs -> .h5ad and .h5ad -> .fcs
"""


import getpass
import os.path
from pathlib import Path
from tkinter import *
from tkinter import filedialog
import FlowCytometryTools as fct
import anndata
import numpy as np
from pytometry.Preprocessing import process_data as proc
from pytometry.converter import fcswriter


def __toanndata(filenamefcs, fcsfile):
    """
    Converts .fcs file to .h5ad file.
    :param filenamefcs: filename without extension
    :param fcsfile: path to .fcs file
    :return: Anndata object with additional .uns entries
    """
    fcsdata = fct.FCMeasurement(ID='FCS-file', datafile=fcsfile)
    adata = anndata.AnnData(X=fcsdata.data[:].values)
    adata.var_names = fcsdata.channel_names
    adata.uns['meta'] = fcsdata.read_meta()

    if '$SPILLOVER' in fcsdata.meta:
        adata.uns['spill_mat'] = proc.create_spillover_mat(fcsdata)
        adata.uns['comp_mat'] = proc.create_comp_mat(adata.uns['spill_mat'])

    adata.write_h5ad(Path(filenamefcs + '_converted' + '.h5ad'))
    return adata


def __tofcs(filenameh5ad, anndatafile):
    """
    Converts .h5ad file to .fcs file.
    :param filenameh5ad: filename without extension
    :param anndatafile: path to .h5ad file
    :return: Metadata of the created .fcs file.
    """
    # String to avoid duplicate keywords
    clear_dupl = ['__header__', '_channels_', '_channel_names_',
                  '$BEGINANALYSIS', '$ENDANALYSIS', '$BEGINSTEXT', '$ENDSTEXT',
                  '$BEGINDATA', '$ENDDATA', '$BYTEORD', '$DATATYPE',
                  '$MODE', '$NEXTDATA', '$TOT', '$PAR', '$fcswrite version']

    adata = anndata.read_h5ad(anndatafile)
    dictionary = adata.uns['meta']
    ch_shortnames = dictionary['_channels_'][:, 0]

    # Include long channel names in seperate Key
    count = 1

    for name in dictionary['_channel_names_']:
        dictionary['$P' + str(count) + 'S'] = name
        count = count + 1

    for i in clear_dupl:
        dictionary.pop(i, None)

    fcswriter.write_fcs(Path(filenameh5ad + '_converted' + '.fcs'), ch_shortnames, np.array(adata.var_names).tolist(),
                        adata.X, dictionary, 'big', False)
    return fct.FCMeasurement('FCS-file', filenameh5ad + '_converted' + '.fcs')


def readandconvert(datafile=''):
    """
    Loads files and converts them according to their extension.
    :rtype: A list of loaded files.
    """
    elementlist = []

    # Path to file
    if datafile != '':
        file_names = [datafile]
    else:
        username = getpass.getuser()  # current username

        file_dialog = Tk()
        file_dialog.withdraw()

        file_names = filedialog.askopenfilenames(initialdir="/home/%s/SampleData/" % username, title="Select file",
                                                 filetypes=(("all files", "*.*"), ("fcs files", "*.fcs"),
                                                            ("h5ad files", ".h5ad")))

    for file_name in file_names:

        file_path = file_name
        filename, file_extension = os.path.splitext(file_path)

        if file_extension == '.fcs':
            elementlist.append(__toanndata(filename, file_path))
        elif file_extension == '.h5ad':
            elementlist.append(__tofcs(filename, file_path))
        else:
            print('File ' + file_name + ' can not be converted!')

    return elementlist
