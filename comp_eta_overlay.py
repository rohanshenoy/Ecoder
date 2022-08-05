import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import emd
from utils.metrics import hexMetric
import scipy
from scipy import stats, optimize, interpolate
import os
import pandas as pd

import argparse
from utils.logger import _logger

def load_data(inputFile):
    data=pd.read_csv(inputFile,usecols=[*range(0,48)])
    data_values=data.values
            
    return data_values

def load_eta(inputFile):
    data=pd.read_csv(inputFile,usecols=[48])
    data_values=data.to_numpy().flatten()
            
    return data_values

parser = argparse.ArgumentParser()
parser.add_argument('-i',"--inputDirs", type=str, default='/ecoderemdvol/QK/comp_ttbar/', dest="inputDirs",
                    help="previous AE trainings to compare")
parser.add_argument('-l',"--inputLegends", type=str, default='a,b,c', dest="inputLegends",
                    help="previous AE trainings to compare")
parser.add_argument('-o',"--outputDir", type=str, default='/ecoderemdvol/QK/comp/', dest="outputDir")

def main(args):
    
    inputDirs = args.inputDirs.split(",")
    inputLegends = args.inputLegends.split(",")
    offsets = [-0.04,0,0.04]

    lengths = len(inputDirs)
    etas = []
    inputs = []
    outputs = []
    emds=[]
    
    for inputDir in inputDirs:
        from utils.metrics import emd
        etas.append(load_eta(inputDir+'verify_input_calQ.csv'))
        input_Q = load_data(inputDir+'verify_input_calQ.csv')
        inputs.append(input_Q)
        decoded_Q = load_data(inputDir+'verify_decoded_calQ.csv')
        outputs.append(decoded_Q)

        indices = range(0,len(input_Q))
        emd_app = np.array([emd(input_Q[i],decoded_Q[j]) for i, j in zip(indices,indices)])
        emds.append(emd_app)
        
    column_names = []
    
    for i in range(lengths):
        column_names.append(inputLegends[i]+'_eta')
        column_names.append(inputLegends[i]+'_emd')
    df = pd.DataFrame(columns=column_names)
    
    for i in range(lengths):
        df[inputLegends[i]+'_eta'] = etas[i]
        df[inputLegends[i]+'_emd'] = emds[i]
    df.to_csv(args.outputDir+'comp.csv')

    for iteration,emd in enumerate(emds):

        x=etas[iteration]
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

        off = offsets[iteration]
        plt.errorbar(x=bin_centers+off, y=median, yerr=[one_sigma_loe,one_sigma_hie], label=inputLegends[iteration])

    plt.legend(loc='upper right')
    plt.xlabel(r'$|\eta|$')
    plt.ylabel('EMD')
    plt.savefig(args.outputDir+'overlay.pdf',dpi=600)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    
    

