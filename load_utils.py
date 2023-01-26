import numpy as np
import pandas as pd
import argparse
import json
import pickle
import os
import numba
import tensorflow as tf
from tensorflow.keras import losses

from tensorflow.keras import callbacks
from tensorflow import keras as kr
import tensorflow.keras.optimizers as opt

#import model layers
from tensorflow.keras.layers import Layer,Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Conv2DTranspose, Reshape, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1_l2      
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#import QKeras as qkr
from qkeras import QDense, QConv2D, QActivation
import json

# for sinkhorn metric
#import ot_tf
import ot

import matplotlib.pyplot as plt


import uproot
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, BaseSchema

def load_data(infile,nrows):
    
    # charge data headers of 48 Input Trigger Cells (TC) 
    CALQ_COLS = ['CALQ_%i'%c for c in range(0, 48)]
    
    #Keep track of phys data

    data = pd.read_csv(infile, dtype=np.float64, header=0, nrows = nrows, usecols= CALQ_COLS)
    eta = pd.read_csv(infile, dtype=np.float64, header=0, nrows = nrows).tc_eta
    phi = pd.read_csv(infile, dtype=np.float64, header=0, nrows = nrows).tc_phi
    
    # mask rows where occupancy is zero
    mask_occupancy = (data[CALQ_COLS].astype('float64').sum(axis=1) != 0)
    data = data[mask_occupancy]
    eta = eta[mask_occupancy]
            
    data = data[CALQ_COLS].astype('float64')
    data_values = data.values
    eta_values = eta.values
    print(data.shape)
    data.describe()

    return (data_values,eta_values)

@numba.jit
def normalize(data, sumlog2=True):
    maxes =[]
    sums =[]
    sums_log2=[]
    for i in range(len(data)):
        maxes.append( data[i].max() )
        sums.append( data[i].sum() )
        sums_log2.append( 2**(np.floor(np.log2(data[i].sum()))) )
        if sumlog2:
            data[i] = 1.*data[i]/(sums_log2[-1] if sums_log2[-1] else 1.)
        else:
            data[i] = 1.*data[i]/(data[i].sum() if data[i].sum() else 1.)
    if sumlog2:
        return  data,np.array(maxes),np.array(sums_log2)
    else:
        return data,np.array(maxes),np.array(sums)

@numba.jit
def unnormalize(norm_data,maxvals, sumlog2=True):
    for i in range(len(norm_data)):
        if sumlog2:
            sumlog2 = 2**(np.floor(np.log2(norm_data[i].sum())))
            norm_data[i] =  norm_data[i] * maxvals[i] / (sumlog2 if sumlog2 else 1.)
        else:
            norm_data[i] =  norm_data[i] * maxvals[i] / (norm_data[i].sum() if norm_data[i].sum() else 1.)
    return norm_data



def loadTrainingData(inputRoot,
                     rootFileTDirectory='FloatingpointThreshold0DummyHistomaxGenmatchGenclustersntuple',
                     gen_pt_min = 0.1,
                     gen_pt_max = 35,
                     abs_eta_min=2.1,
                     abs_eta_max=2.7,
                     useADC=False):
    
    eventFile = uproot.open(inputRoot)

    events = NanoEventsFactory.from_root(
    eventFile,
    treepath=rootFileTDirectory+'/HGCalTriggerNtuple',
    schemaclass=BaseSchema).events()

    gen_pt = events.gen_pt
    gen_eta = events.gen_eta

    #make gen_pt and gen_eta cuts, gen eta cuts on electrons in positive eta, msking on both e +- event

    pt_min_mask = (gen_pt >=gen_pt_min)[:,0]
    pt_max_mask = (gen_pt <=gen_pt_max)[:,0]
    eta_min_mask = (gen_eta[:,0] >=abs_eta_min)
    eta_max_mask = (gen_eta[:,0] <= abs_eta_max)
    
    mask = pt_min_mask * pt_max_mask * eta_min_mask * eta_max_mask

    #test if all events were discared, if all discarded skip this root file

    test = (mask == [False]*len(mask))

    if ak.all(test):
        return 0

    tc_mask_data = ak.Array(
        {
            'tc_zside': events['tc_zside'][mask],
            'tc_layer': events['tc_layer'][mask],
            'tc_waferu': events['tc_waferu'][mask],
            'tc_waferv': events['tc_waferv'][mask],
            'tc_cellu': events['tc_cellu'][mask],
            'tc_cellv': events['tc_cellv'][mask],
            'tc_data': events['tc_data'][mask],
            #'tc_simenergy': events['tc_simenergy'][mask],
            'tc_eta': events['tc_eta'][mask]
        }
    )

    df = ak.to_pandas(tc_mask_data)

    dfRemap = pd.read_csv('/ecoderemdvol/tcRemap.csv')
    df = df.reset_index().merge(dfRemap)

    df['ADCT'] = (df.tc_data* ((1./np.cosh(df.tc_eta))*2**12).astype(int)/2**12).astype(int)

    #create new unique index (can't get pivot working on multi-indexing, but this is essentially the same)
    df['WaferEntryIdx'] = (df.entry*1000000 + df.tc_layer*10000 + df.tc_waferu*100 + df.tc_waferv)*df.tc_zside

    val = 'ADCT'
    if useADC:
        val='tc_data'
    dfTrainData = df.pivot_table(index='WaferEntryIdx',columns='tc_cell_train',values=val).fillna(0).astype(int)
    dfTrainData.columns = [f'CALQ_{i}' for i in range(48)]

    dfTrainData[['entry','zside','layer','waferu','waferv']] = df.groupby(['WaferEntryIdx'])[['entry','tc_zside','tc_layer','tc_waferu','tc_waferv']].mean()
    
    #dfTrainData['simenergy'] = df.groupby(['WaferEntryIdx'])[['tc_simenergy']].sum()

    #Mapping wafer_u,v to physical coordinates
    dfEtaPhi=pd.read_csv('/ecoderemdvol/WaferEtaPhiMap.csv')
    dfTrainData=dfTrainData.merge(dfEtaPhi, on=['layer','waferu','waferv'])
    dfTrainData.reset_index(drop=True,inplace=True)
    
    return dfTrainData





