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

def normalize_logicle(adata, t = 262144, m = 4.5, w = 0.5, a = 0):
    """
    Logicle transformation, implemented as defined in the
    GatingML 2.0 specification, adapted from FlowKit and Flowutils Python packages:
        
    logicle(x, T, W, M, A) = root(B(y, T, W, M, A) − x)
    
    where B is a modified bi-exponential function defined as:
        
    B(y, T, W, M, A) = ae^(by) − ce^(−dy) − f
    
    The Logicle transformation was originally defined in the publication:
        
    Moore WA and Parks DR. Update for the logicle data scale including operational
    code implementations. Cytometry A., 2012:81A(4):273–277.
    
    :param adata: anndata object 
    :param t: parameter for the top of the linear scale (e.g. 262144)
    :param m: parameter for the number of decades the true logarithmic scale
        approaches at the high end of the scale
    :param w: parameter for the approximate number of decades in the linear region
    :param a: parameter for the additional number of negative decades
    """
    
    #initialise precision
    taylor_length = 16 
    #initialise parameter dictionary
    p = dict()
    
    T = t
    M = m
    W = w
    A = a

	#actual parameters
	#formulas from bi-exponential paper
    p['w'] = W / (M + A)
    p['x2'] = A / (M + A)
    p['x1'] = p['x2'] + p['w']
    p['x0'] = p['x2'] + 2 * p['w']
    p['b'] = (M + A) * np.log(10)
    p['d'] = solve(p['b'], p['w'])
    c_a = np.exp(p['x0'] * (p['b'] + p['d']))
    mf_a = np.exp(p['b'] * p['x1']) - c_a / np.exp(p['d'] * p['x1'])
    p['a'] = T / ((np.exp(p['b']) - mf_a) - c_a / np.exp(p['d']))
    p['c'] = c_a * p['a']
    p['f'] = -mf_a * p['a']

    #use Taylor series near x1, i.e., data zero to
    #avoid round off problems of formal definition
    p['xTaylor'] = p['x1'] + p['w'] / 4

    #compute coefficients of the Taylor series
    posCoef = p['a'] * np.exp(p['b'] * p['x1'])
    negCoef = -p['c'] / np.exp(p['d'] * p['x1'])

	#16 is enough for full precision of typical scales
    p['taylor'] = np.zeros(taylor_length)

    for i in range(0, taylor_length):
        posCoef *= p['b'] / (i + 1)
        negCoef *= -p['d'] / (i + 1)
        p['taylor'][i] = posCoef + negCoef
	
    p['taylor'][1] = 0 # exact result of Logicle condition

	#end original initialize method
    for i in range(0, adata.n_vars):
        for j in range(0,adata.n_obs): 
            adata.X[j,i] = scale(adata.X[j,i], p)
    
    return adata


def scale(value, p):
    
    DBL_EPSILON = 1e-9 #from C++, defined as the smallest difference between 1 
    # and the next larger number
    #handle true zero separately
    if (value == 0):
        return p['x1']

	#reflect negative values
    negative = value < 0 
    if (negative):
        value = - value

	#initial guess at solution
	
    if (value < p['f']):
		#use linear approximation in the quasi linear region
        x = p['x1'] + value / p['taylor'][0]
    else:
		#otherwise use ordinary logarithm
        x = np.log(value / p['a']) / p['b']

	#try for double precision unless in extended range
    tolerance = 3 * DBL_EPSILON
    if (x > 1):
        tolerance = 3 * x * DBL_EPSILON

    for i in range(0, 40):
        #compute the function and its first two derivatives
        ae2bx = p['a'] * np.exp(p['b'] * x)
        ce2mdx = p['c'] / np.exp(p['d'] * x)

        if (x < p['xTaylor']):
			#near zero use the Taylor series
            y = seriesBiexponential(p, x) - value
        else:
			# this formulation has better round-off behavior
            y = (ae2bx + p['f']) - (ce2mdx + value)
        abe2bx = p['b'] * ae2bx
        cde2mdx = p['d'] * ce2mdx
        dy = abe2bx + cde2mdx
        ddy = p['b'] * abe2bx - p['d'] * cde2mdx

		# this is Halley's method with cubic convergence
        delta = y / (dy * (1 - y * ddy / (2 * dy * dy)))
        x -= delta

		# if we've reached the desired precision we're done
        if (abs(delta) < tolerance):
			# handle negative arguments
            if (negative):
                return 2 * p['x1'] - x
            else:
                return x
		
	# if we get here, scale did not converge
    return -1
    

    

def solve(b, w): 
    
    
    DBL_EPSILON = 1e-9 #from C++, defined as the smallest difference between 1 
    # and the next larger number
    


	# w == 0 means its really arcsinh
    if (w == 0):
        return b

	#precision is the same as that of b
    tolerance = 2 * b * DBL_EPSILON

	# based on RTSAFE from Numerical Recipes 1st Edition
	# bracket the root
    d_lo = 0
    d_hi = b

	#bisection first step
    d = (d_lo + d_hi) / 2
    last_delta = d_hi - d_lo

	# evaluate the f(w,b) = 2 * (ln(d) - ln(b)) + w * (b + d)
	# and its derivative
    f_b = -2 * np.log(b) + w * b
    f = 2 * np.log(d) + w * d + f_b
    last_f = np.nan

    for i in range(1, 40):
        #compute the derivative
        df = 2 / d + w

		# if Newton's method would step outside the bracket
		# or if it isn't converging quickly enough
        if (((d - d_hi) * df - f) * ((d - d_lo) * df - f) >= 0 or
			abs(1.9 * f) > abs(last_delta * df)):
		
            # take a bisection step
            delta = (d_hi - d_lo) / 2
            d = d_lo + delta
            if (d == d_lo):
                return d # nothing changed, we're done
		
        else:
		
			# otherwise take a Newton's method step
            delta = f / df
            t = d
            d -= delta
            if (d == t):
                return d #nothing changed, we're done
		
		# if we've reached the desired precision we're done
        if (abs(delta) < tolerance):
            return d
        last_delta = delta

		# recompute the function
        f = 2 * np.log(d) + w * d + f_b
        if (f == 0 or f == last_f):
            return d # found the root or are not going to get any closer
        last_f = f

		# update the bracketing interval
        if (f < 0):
            d_lo = d
        else:
            d_hi = d
	

    return -1


def seriesBiexponential(p, value):
    
    #initialise precision
    taylor_length = 16
    #Taylor series is around x1
    x = value - p['x1']
	# note that taylor[1] should be identically zero according
	# to the Logicle condition so skip it here
    sum1 = p['taylor'][taylor_length - 1] * x
    for i in range(taylor_length - 2, 1, -1):
        sum1 = (sum1 + p['taylor'][i]) * x
	
    return (sum1 * x + p['taylor'][0]) * x
    
     
