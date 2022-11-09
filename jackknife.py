import os
from glob import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression
import pickle
import seaborn as sns
from datetime import date

tele = pd.read_csv('/ecoderemdvol/paper_plots/physics/tele.csv')
emd  = pd.read_csv('/ecoderemdvol/paper_plots/physics/emd.csv')

binetasize = 0.1
binptsize = 5

tele['genpart_bineta'] = ((tele.genpart_abseta - 1.6)/binetasize).astype('int32')
emd['genpart_bineta'] = ((emd.genpart_abseta - 1.6)/binetasize).astype('int32')

def effrms(df, c=0.68):
    """Compute half-width of the shortest interval
    containing a fraction 'c' of items in a 1D array.
    """
    out = {}
    for col in df:
        x = df[col]
        x = np.sort(x, kind="mergesort")
        m = int(c * len(x)) + 1
        out[col] = [np.min(x[m:] - x[:-m]) / 2.0]
    return pd.DataFrame(out).iloc[0]

eta_binned = {} 
pt_mean_binned = {} 
pt_rms_eff_binned={}

#For telescope_loss first

select = tele.query('genpart_pid==1 and genpart_pt>10')
eta_binned['AEtele'] = (select.groupby('genpart_bineta').mean())['genpart_abseta']
pt_mean_binned['AEtele'] = (select.groupby('genpart_bineta').mean())['corr_eta_over_gen_pt']
pt_rms_eff_binned['AEtele'] = (select.groupby('genpart_bineta').apply(effrms))['corr_eta_over_gen_pt']

#emd next

select = emd.query('genpart_pid==1 and genpart_pt>10')
eta_binned['AElphe'] = (select.groupby('genpart_bineta').mean())['genpart_abseta']
pt_mean_binned['AElphe'] = (select.groupby('genpart_bineta').mean())['corr_eta_over_gen_pt']
pt_rms_eff_binned['AElphe'] = (select.groupby('genpart_bineta').apply(effrms))['corr_eta_over_gen_pt']

eta = {} 
pt = {} 

#For telescope_loss first

select = tele.query('genpart_pid==1 and genpart_pt>10')
eta['AEtele'] = select['genpart_abseta'].values
pt['AEtele'] = select['corr_eta_over_gen_pt'].values

#emd next

select = emd.query('genpart_pid==1 and genpart_pt>10')
eta['AElphe'] = select['genpart_abseta'].values
pt['AElphe'] = select['corr_eta_over_gen_pt'].values

#emd loss jacknife

n = len(eta['AElphe'])

full_jk = np.empty(shape=(n,13))

for i in range(n):
    
    jk_eta = np.delete(eta['AElphe'],i)
    jk_pt = np.delete(pt['AElphe'],i)
    
    eta_binned = {} 
    pt_mean_binned = {} 
    pt_rms_eff_binned={}
    
    jk_bin_eta = ((jk_eta - 1.6)/0.1).astype('int32')
    
    df = pd.DataFrame(columns=['eta','bin_eta','pt'])
    
    df['eta'] = jk_eta
    df['bin_eta'] = jk_bin_eta
    df['pt'] = jk_pt

    eta_binned['AElphe'] = (df.groupby('bin_eta').mean())['eta']
    pt_mean_binned['AElphe'] = (df.groupby('bin_eta').mean())['pt']
    pt_rms_eff_binned['AElphe'] = (df.groupby('bin_eta').apply(effrms))['pt']
    
    physics_res = pt_rms_eff_binned['AElphe']/pt_mean_binned['AElphe']
    
    full_jk[i] = np.asarray(physics_res)
        
df_emd = pd.DataFrame(full_jk)

df_emd.to_csv('/ecoderemdvol/paper_plots/physics/emd_jk_2.csv')

#telescope loss jacknife

#emd loss jacknife

n = len(eta['AEtele'])

full_jk = np.empty(shape=(n,13))

for i in range(n):
    
    jk_eta = np.delete(eta['AEtele'],i)
    jk_pt = np.delete(pt['AEtele'],i)
    
    eta_binned = {} 
    pt_mean_binned = {} 
    pt_rms_eff_binned={}
    
    jk_bin_eta = ((jk_eta - 1.6)/0.1).astype('int32')
    
    df = pd.DataFrame(columns=['eta','bin_eta','pt'])
    
    df['eta'] = jk_eta
    df['bin_eta'] = jk_bin_eta
    df['pt'] = jk_pt

    eta_binned['AEtele'] = (df.groupby('bin_eta').mean())['eta']
    pt_mean_binned['AEtele'] = (df.groupby('bin_eta').mean())['pt']
    pt_rms_eff_binned['AEtele'] = (df.groupby('bin_eta').apply(effrms))['pt']
    
    physics_res = pt_rms_eff_binned['AEtele']/pt_mean_binned['AEtele']
    
    full_jk[i] = np.asarray(physics_res)
        
df_tele = pd.DataFrame(full_jk)

df_emd.to_csv('/ecoderemdvol/paper_plots/physics/emd_tele.csv')


        



