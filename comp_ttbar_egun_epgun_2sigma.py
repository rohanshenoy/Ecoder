import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import emd
from utils.metrics import hexMetric

import scipy

from scipy import stats, optimize, interpolate

import os

import pandas as pd

def load_data(inputFile):
    data=pd.read_csv(inputFile,usecols=[*range(0,48)],nrows=1000)
    data_values=data.values
            
    return data_values

def load_phy(inputFile):
    data=pd.read_csv(inputFile,usecols=[*range(48,49)],nrows=1000)
    data_values=data.values
            
    return data_values

phys_ttbar = load_phy('/ecoderemdvol/ttbar/AE/22AE/aps_plots/8x8_c8_S2_tele/verify_input_calQ.csv')
phys_epgun  = load_phy('/ecoderemdvol/EleGun/EPGun-PU200/AE/comp_ae_mse_full_eta_tele_2/8x8_c8_S2_tele/verify_input_calQ.csv')
phys_egun  = load_phy('/ecoderemdvol/EleGun/EGun-PU200/AE/comp_3/8x8_c8_S2_ae_mse/verify_input_calQ.csv')

ttbar_emd_input_Q   = load_data('/ecoderemdvol/ttbar/AE/22AE/aps_plots/8x8_c8_S2_ae_mse/verify_input_calQ.csv')

ttbar_emd_output_Q  = load_data('/ecoderemdvol/ttbar/AE/22AE/aps_plots/8x8_c8_S2_ae_mse/verify_decoded_calQ.csv')

egun_emd_input_Q   = load_data('/ecoderemdvol/EleGun/EGun-PU200/AE/comp_3/8x8_c8_S2_ae_mse/verify_input_calQ.csv')

egun_emd_output_Q  = load_data('/ecoderemdvol/EleGun/EGun-PU200/AE/comp_3/8x8_c8_S2_ae_mse/verify_decoded_calQ.csv')

epgun_emd_input_Q    = load_data('/ecoderemdvol/EleGun/EPGun-PU200/AE/comp_ae_mse_full_eta_tele_2/8x8_c8_S2_ae_mse_full_eta/verify_input_calQ.csv')

epgun_emd_output_Q   = load_data('/ecoderemdvol/EleGun/EPGun-PU200/AE/comp_ae_mse_full_eta_tele_2/8x8_c8_S2_ae_mse_full_eta/verify_decoded_calQ.csv')
    
#Calculate offset for better presentation
    
indices_ttbar = range(0,(len(ttbar_emd_input_Q)))
indices_egun = range(0,(len(egun_emd_input_Q)))
indices_epgun = range(0,(len(epgun_emd_input_Q)))
eta_ttbar=[]
eta_epgun=[]
eta_egun=[]
for i in indices_ttbar:
    eta_ttbar=np.append(eta_ttbar,phys_ttbar[i][0])
    
for i in indices_egun:
    eta_egun=np.append(eta_egun,phys_egun[i][0])
    
for i in indices_epgun:
    eta_epgun=np.append(eta_epgun,phys_epgun[i][0])

offset=0.75*((np.max(eta_ttbar)-np.min(eta_ttbar))/10)/2
 
emd_values_ttbar_emd  = np.array([emd(ttbar_emd_input_Q[i],ttbar_emd_output_Q[j]) for i, j in zip(indices_ttbar,indices_ttbar)]) 
emd_values_egun_emd  = np.array([emd(egun_emd_input_Q[i],egun_emd_output_Q[j]) for i, j in zip(indices_egun,indices_egun)])     
emd_values_epgun_emd   = np.array([emd(epgun_emd_input_Q[i],epgun_emd_output_Q[j]) for i, j in zip(indices_epgun,indices_epgun)])

#emd_values_arr=[emd_values_ttbar_tele,emd_values_ttbar_emd,emd_values_egun_tele,emd_values_egun_emd]
emd_values_arr=[emd_values_ttbar_emd,emd_values_egun_emd,emd_values_epgun_emd]
eta_arr=[eta_ttbar,eta_egun,eta_epgun]
labels = ['ttbar','eta_-1.5_1.5','epgun_1.5_3']
colors1=['tab:blue','tab:orange','tab:green']
colors2=['blue','red','tab:olive']

fig,ax=plt.subplots()
plt.figure(figsize=(6,4))

for iteration,emd in enumerate(emd_values_arr):
    
    x=eta_arr[iteration]
    y = emd
    
    nbins=10
    stats=True
    lims = (1.6,3.0)
    median_result = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5))
    one_sigma_lo_result     = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5-0.68/2))
    one_sigma_hi_result     = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5+0.68/2))
    
    two_sigma_lo_result     = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5-0.95/2))
    two_sigma_hi_result     = scipy.stats.binned_statistic(x, y, bins=nbins, range=lims, statistic=lambda x: np.quantile(x,0.5+0.95/2))
    median = np.nan_to_num(median_result.statistic)
    
    one_sigma_hi = np.nan_to_num(one_sigma_hi_result.statistic)
    one_sigma_lo = np.nan_to_num(one_sigma_lo_result.statistic)
    
    two_sigma_hi = np.nan_to_num(two_sigma_hi_result.statistic)
    two_sigma_lo = np.nan_to_num(two_sigma_lo_result.statistic)
    
    
    one_sigma_hie = one_sigma_hi-median
    one_sigma_loe = median-one_sigma_lo
    
    two_sigma_hie = two_sigma_hi-median
    two_sigma_loe = median-two_sigma_lo
    
    
    bin_edges = median_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    
    off = 0
    if iteration != 1:
        off = offset*iteration
    plt.errorbar(x=bin_centers+off, y=median, yerr=[one_sigma_loe,one_sigma_hie],label=labels[iteration],color=colors1[iteration])
    plt.errorbar(x=bin_centers+off, y=median, yerr=[two_sigma_loe,two_sigma_hie],fmt='--',alpha=0.5,color=colors2[iteration])
    
plt.legend(loc='upper right')
plt.xlabel(r'$\eta$')
plt.ylabel('EMD')
plt.savefig('/ecoderemdvol/ttbar_egun_epgun_ae_comp_eta_EMD_2sigma.pdf',dpi=600)
    
    

