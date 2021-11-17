#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 18:19:35 2021

@author: buettnerm
"""

import scanpy as sc
import anndata as ann
import numpy as np

def normalize_arcsinh(adata, cofactor):
    """
    :param adata: anndata object
    :param cofactor: all values are divided by this 
                     factor before arcsinh transformation
                     recommended values for cyTOF data: 5
                     and for flow data: 150 
    :return: normalised adata object
    """
    
    adata.X = np.arcsinh(adata.X/cofactor)
    return adata