
import os.path
from pathlib import Path
import anndata as anndata
import numpy as np
import pandas as pd
import tools.FlowCytometryTools as fct
from converter import fcswriter


# create a spillover matrix
def __create_spillover_mat(fcsdata):
    spillover = fcsdata.meta['$SPILLOVER'].split(",")
    num_col = int(spillover[0])
    channel_names = spillover[1:(int(spillover[0]) + 1)]
    channel_data = fcsdata.meta['_channels_']
    channel_renames = [str(channel_data['$PnS'][channel_data['$PnN'] == name].values[0]) for name in channel_names]
    spill_values = np.reshape([float(inp) for inp in spillover[(int(spillover[0]) + 1):]], [num_col, num_col])
    spill_df = pd.DataFrame(spill_values, columns=channel_renames)
    return spill_df


# Conversion fcs to AnnData
def __toanndata(filenamefcs, fcsfile):
    fcsdata = fct.FCMeasurement(ID='FCS-file', datafile=fcsfile)
    adata = anndata.AnnData(X=fcsdata.data[:].values)
    adata.var_names = fcsdata.channel_names
    adata.uns = fcsdata.read_meta()
    adata.uns['meta'] = fcsdata.read_meta()

    if '$SPILLOVER' in fcsdata.meta:
        adata.uns['comp_mat'] = __create_spillover_mat(fcsdata)

    adata.write_h5ad(Path(filenamefcs + '_converted' + '.h5ad'))
    return adata


# Conversion AnnData to fcs
def __tofcs(filenameh5ad, anndatafile):
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
        # print(dictionary['_channels_'][6][0])

    for i in clear_dupl:
        dictionary.pop(i, None)
    fcswriter.write_fcs(Path(filenameh5ad + '_converted' + '.fcs'), ch_shortnames, np.array(adata.var_names).tolist(),
                        adata.X, dictionary, 'big', False)


# main function
def read_convert(datafile):
    # Path to file
    file_name = datafile

    # perform conversion
    file_path = file_name
    filename, file_extension = os.path.splitext(file_path)

    if file_extension == '.fcs':
        __toanndata(filename, file_path)
    elif file_extension == '.h5ad':
        __tofcs(filename, file_path)
    else:
        print('No appropriate file selected!')
