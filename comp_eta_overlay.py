import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import emd
from utils.metrics import hexMetric

import scipy

from scipy import stats, optimize, interpolate

import os

import pandas as pd

def load_data(inputFile):
    data=pd.read_csv(inputFile,usecols=[*range(0,48)],nrows=800000)
    data_values=data.values
            
    return data_values

def load_phy(inputFile):
    data=pd.read_csv(inputFile,usecols=[*range(48,49)],nrows=800000)
    data_values=data.values
            
    return data_values
"""

ttbar_tele_input  = '/ecoderemdvol/22AE/aps_plots/8x8_c8_S2_tele/verify_input_calQ.csv'

ttbar_tele_output = '/ecoderemdvol/22AE/aps_plots/8x8_c8_S2_tele/verify_decoded_calQ.csv'

ttbar_emd_input   = '/ecoderemdvol/22AE/aps_plots/8x8_c8_S2_ae_mse/verify_input_calQ.csv'

ttbar_emd_output  = '/ecoderemdvol/22AE/aps_plots/8x8_c8_S2_ae_mse/verify_decoded_calQ.csv'


egun_tele_input   = '/ecoderemdvol/EleGun/AE/comp_3/8x8_c8_S2_tele/verify_input_calQ.csv'

egun_tele_output  = '/ecoderemdvol/EleGun/AE/comp_3/8x8_c8_S2_tele/verify_decoded_calQ.csv'

egun_emd_input    = '/ecoderemdvol/EleGun/AE/comp_3/8x8_c8_S2_ae_mse/verify_input_calQ.csv'

egun_emd_output   = '/ecoderemdvol/EleGun/AE/comp_3/8x8_c8_S2_ae_mse/verify_decoded_calQ.csv'

"""

phys_ttbar = load_phy('/ecoderemdvol/22AE/aps_plots/8x8_c8_S2_tele/verify_input_calQ.csv')
phys_egun  = load_phy('/ecoderemdvol/EleGun/AE/comp_3/8x8_c8_S2_tele/verify_input_calQ.csv')

ttbar_tele_input_Q  = load_data('/ecoderemdvol/22AE/aps_plots/8x8_c8_S2_tele/verify_input_calQ.csv')

ttbar_tele_output_Q = load_data('/ecoderemdvol/22AE/aps_plots/8x8_c8_S2_tele/verify_decoded_calQ.csv')

ttbar_emd_input_Q   = load_data('/ecoderemdvol/22AE/aps_plots/8x8_c8_S2_ae_mse/verify_input_calQ.csv')

ttbar_emd_output_Q  = load_data('/ecoderemdvol/22AE/aps_plots/8x8_c8_S2_ae_mse/verify_decoded_calQ.csv')


egun_tele_input_Q   = load_data('/ecoderemdvol/EleGun/AE/comp_3/8x8_c8_S2_tele/verify_input_calQ.csv')

egun_tele_output_Q  = load_data('/ecoderemdvol/EleGun/AE/comp_3/8x8_c8_S2_tele/verify_decoded_calQ.csv')

egun_emd_input_Q    = load_data('/ecoderemdvol/EleGun/AE/comp_3/8x8_c8_S2_ae_mse/verify_input_calQ.csv')

egun_emd_output_Q   = load_data('/ecoderemdvol/EleGun/AE/comp_3/8x8_c8_S2_ae_mse/verify_decoded_calQ.csv')
    
#Calculate offset for better presentation
indices_ttbar = range(0,(len(phys_ttbar)))
    
indices_ttbar = range(0,(len(ttbar_tele_input_Q)))

eta_ttbar=[]
                  
for i in indices_ttbar:
    eta_ttbar=np.append(eta_ttbar,phys_ttbar[i][0])
    
offset=0.75*((np.max(eta_ttbar)-np.min(eta_ttbar))/10)/4
        
indices_egun = range(0,(len(egun_tele_input_Q)))
    
eta_egun=[]
                  
for i in indices_egun:
    eta_egun=np.append(eta_egun,phys_egun[i][0])
    
emd_values_ttbar_tele = np.array([emd(ttbar_tele_input_Q[i],ttbar_tele_output_Q[j]) for i, j in zip(indices_ttbar,indices_ttbar)]) 
emd_values_ttbar_emd  = np.array([emd(ttbar_emd_input_Q[i],ttbar_emd_output_Q[j]) for i, j in zip(indices_ttbar,indices_ttbar)]) 
emd_values_egun_tele  = np.array([emd(egun_tele_input_Q[i],egun_tele_output_Q[j]) for i, j in zip(indices_egun,indices_egun)])     
emd_values_egun_emd   = np.array([emd(egun_emd_input_Q[i],egun_emd_output_Q[j]) for i, j in zip(indices_egun,indices_egun)])

emd_values_arr=[emd_values_ttbar_tele,emd_values_ttbar_emd,emd_values_egun_tele,emd_values_egun_emd]
labels = ['ttbar_tele','ttbar_emd','egun_tele','egun_emd']

fig,ax=plt.subplots()
plt.figure(figsize=(6,4))

for iteration,emd in enumerate(emd_values_arr):
    
    x=[]
    if iteration == 0 or iteration == 1:
        x = eta_ttbar
    else:
        x = eta_egun
    
    y = emd
    
    nbins=10
    stats=True
    lims = (1.6,3.0)
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
    
    off = 0
    if iteration != 3:
        off = offset*iteration
    plt.errorbar(x=bin_centers+off, y=median, yerr=[loe,hie], label=labels[iteration])
    
plt.legend(loc='upper right')
plt.xlabel(r'$\eta$')
plt.ylabel('EMD')
plt.savefig('/ecoderemdvol/ttbar_egun_ae_comp_eta_EMD.pdf',dpi=600)
    
    


