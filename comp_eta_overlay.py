import numpy as np
import matplotlib.pyplot as plt

from utils.metrics import emd
from utils.metrics import hexMetric

import scipy

from scipy import stats, optimize, interpolate

import os

import pandas as pd

def load_data(inputFile):
    data=pd.read_csv(inputFile,usecols=[*range(0,48)],nrows=800)
    data_values=data.values
            
    return data_values

def load_phy(inputFile):
    data=pd.read_csv(inputFile,usecols=[*range(48,49)],nrows=80)
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

eta_1_phys = load_phy('/ecoderemdvol/EleGun/EPGun-PU200/AE/comp_ae_mse_eta_1_tele/8x8_c8_S2_ae_mse_eta_1/verify_input_calQ.csv')
eta_2_phys = load_phy('/ecoderemdvol/EleGun/EPGun-PU200/AE/comp_ae_mse_eta_2_tele/8x8_c8_S2_ae_mse_eta_2/verify_input_calQ.csv')
full_eta_phys = load_phy('/ecoderemdvol/EleGun/EPGun-PU200/AE/comp_ae_mse_full_eta_tele_2/8x8_c8_S2_ae_mse_full_eta/verify_input_calQ.csv')


eta_1_input_Q = load_data('/ecoderemdvol/EleGun/EPGun-PU200/AE/comp_ae_mse_eta_1_tele/8x8_c8_S2_ae_mse_eta_1/verify_input_calQ.csv')
eta_2_input_Q = load_data('/ecoderemdvol/EleGun/EPGun-PU200/AE/comp_ae_mse_eta_2_tele/8x8_c8_S2_ae_mse_eta_2/verify_input_calQ.csv')
full_eta_input_Q = load_data('/ecoderemdvol/EleGun/EPGun-PU200/AE/comp_ae_mse_full_eta_tele_2/8x8_c8_S2_ae_mse_full_eta/verify_input_calQ.csv')

eta_1_output_Q = load_data('/ecoderemdvol/EleGun/EPGun-PU200/AE/comp_ae_mse_eta_1_tele/8x8_c8_S2_ae_mse_eta_1/verify_decoded_calQ.csv')
eta_2_output_Q = load_data('/ecoderemdvol/EleGun/EPGun-PU200/AE/comp_ae_mse_eta_2_tele/8x8_c8_S2_ae_mse_eta_2/verify_decoded_calQ.csv')
full_eta_output_Q = load_data('/ecoderemdvol/EleGun/EPGun-PU200/AE/comp_ae_mse_full_eta_tele_2/8x8_c8_S2_ae_mse_full_eta/verify_decoded_calQ.csv')

    
#Calculate offset for better presentation
indices_full = range(0,(len(full_eta_phys)))

eta_full=[]
                  
for i in indices_full:
    eta_full=np.append(eta_full,full_eta_phys[i][0])
    
offset=0.75*((np.max(eta_full)-np.min(eta_full))/10)/3
        
indices_eta_1 = range(0,(len(eta_1_phys)))
    
eta_1=[]
                  
for i in indices_eta_1:
    eta_1=np.append(eta_1,eta_1_phys[i][0])
    
indices_eta_2 = range(0,(len(eta_2_phys)))
    
eta_2=[]
                  
for i in indices_eta_2:
    eta_2=np.append(eta_2,eta_2_phys[i][0])
    
emd_values_full = np.array([emd(full_eta_input_Q[i],full_eta_output_Q[j]) for i, j in zip(indices_full,indices_full)])

emd_values_1 = np.array([emd(eta_1_input_Q[i],eta_1_output_Q[j]) for i, j in zip(indices_eta_1,indices_eta_1)])

emd_values_2 = np.array([emd(eta_2_input_Q[i],eta_2_output_Q[j]) for i, j in zip(indices_eta_2,indices_eta_2)])

emd_values_arr=[emd_values_full,emd_values_1,emd_values_2]
eta_arr= [eta_full,eta_1,eta_2]
labels = ['full_eta','eta_1','eta_2']
offset = [0,-0.04,0.04]

fig,ax=plt.subplots()
plt.figure(figsize=(6,4))

for iteration,emd in enumerate(emd_values_arr):
    
    x=eta_arr[iteration]
    
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
    
    off = offset[iteration]
    plt.errorbar(x=bin_centers+off, y=median, yerr=[loe,hie], label=labels[iteration])
    
plt.legend(loc='upper right')
plt.xlabel(r'$\eta$')
plt.ylabel('EMD')
plt.savefig('/ecoderemdvol/comp_split_eta_EMD.pdf',dpi=300)
    
    


