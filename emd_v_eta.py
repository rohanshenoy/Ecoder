import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import emd
from utils.metrics import hexMetric

import scipy

from scipy import stats, optimize, interpolate

import os

import pandas as pd

def load_data(inputFile):
    data=pd.read_csv(inputFile,usecols=[*range(0,48)])
    data_values=data.values
            
    return data_values

def load_phy(inputFile):
    data=pd.read_csv(inputFile,usecols=[*range(48,49)])
    data_values=data.values
            
    return data_values


def plot_eta(current_dir,models,phys):
    
    fig,ax=plt.subplots()
    plt.figure(figsize=(6,4))
    model_list=models.split(',')
    model_0=model_list[0]
    
    #Calculate offset for better presentation
    indices_00 = range(0,(len(phys)))
    
    eta_00=[]
                  
    for i in indices_00:
        eta_00=np.append(eta_00,phys[i][0])
    
    offset=0.75*((np.max(eta_00)-np.min(eta_00))/10)/(len(model_list))
                       
    
    for im,model in enumerate(models.split(',')):
        if model==model_0:continue
        input_dir=os.path.join(current_dir,model,'verify_input_calQ.csv')
        output_dir=os.path.join(current_dir,model,'verify_decoded_calQ.csv')
        
        input_Q=load_data(input_dir)
        output_Q=load_data(output_dir)
        
        indices = range(0,(len(input_Q)))
        
        emd_values = np.array([emd(input_Q[i],output_Q[j]) for i, j in zip(indices,indices)])
    
        eta=[]
        for i in indices:
            eta=np.append(eta,phys[i][0])
            
        x=eta
        y=emd_values
        
        nbins=10
        lims=None
        stats=True
        if lims==None: lims = (x.min(),x.max())
        median_result = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5))
        lo_result     = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5-0.68/2))
        hi_result     = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5+0.68/2))
        median = np.nan_to_num(median_result.statistic)
        hi = np.nan_to_num(hi_result.statistic)
        lo = np.nan_to_num(lo_result.statistic)
        hie = hi-median
        loe = median-lo
        bin_edges = median_result.bin_edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.

        off = offset*im
        plt.errorbar(x=bin_centers+off, y=median, yerr=[loe,hie], label=model)
            
    #Plot model_0 so that x axis has correct labels without offset
    
    input_dir_0=os.path.join(current_dir,model_0,'verify_input_calQ.csv')
    output_dir_0=os.path.join(current_dir,model_0,'verify_decoded_calQ.csv')
        
    input_Q_0=load_data(input_dir_0)
    output_Q_0=load_data(output_dir_0)
    
    indices_0 = range(0,(len(input_Q_0)))
        
    emd_values_0 = np.array([emd(input_Q_0[i],output_Q_0[j]) for i, j in zip(indices_0,indices_0)])
    
    eta_0=[]
    for i in indices_0:
        eta_0=np.append(eta_0,phys[i][0])
        
    x=eta_0
    y=emd_values_0
        
    nbins=10
    lims=None
    stats=True
    if lims==None: lims = (x.min(),x.max())
    median_result = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5))
    lo_result     = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5-0.68/2))
    hi_result     = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5+0.68/2))
    median = np.nan_to_num(median_result.statistic)
    hi = np.nan_to_num(hi_result.statistic)
    lo = np.nan_to_num(lo_result.statistic)
    hie = hi-median
    loe = median-lo
    bin_edges = median_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.

    plt.errorbar(x=bin_centers, y=median, yerr=[loe,hie], label=model_0)
    
    plt.legend(loc='upper right')
    plt.xlabel(r'$\eta$')
    plt.ylabel('EMD')
    plt.savefig(current_dir+'ae_comp_eta_EMD',dpi=600)   
 
