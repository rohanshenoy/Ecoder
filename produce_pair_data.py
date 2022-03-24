import numpy as np
import pandas as pd
import itertools
from utils.metrics import emd
from utils.metrics import hexMetric

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c',type=int, default=0, dest="cut",help="split jobs")

def main(args):
    
    cut_int=args.cut
    
    data_dir='/ecoderemdvol/HGCal22Data_signal_driven_ttbar_v11/nElinks_5/5Elinks_data.csv'
    CALQ_COLS = ['CALQ_%i'%c for c in range(0, 48)]

    data =pd.read_csv(data_dir, dtype=np.float64, header=0, usecols=[*range(0,48)], names=CALQ_COLS, nrows=10000)
    data = data[CALQ_COLS].astype('float64')
    
    calQ = data.values
    start_ind = int((cut_int/100)*len(calQ))
    end_ind   = int(((cut_int+1)/100)*len(calQ))
    test_indices = range(start_ind, end_ind)

    idx1_test = np.array([i for i,j in itertools.product(test_indices,test_indices)])
    idx2_test = np.array([j for i,j in itertools.product(test_indices,test_indices)])

    thresh_i=[]
    thresh_j=[]

    for i, j in zip(idx1_test, idx2_test):
        emd_value = emd(calQ[i],calQ[j])
        if(emd_value<4 and emd_value>0):
            thresh_i=np.append(thresh_i,i)
            thresh_j=np.append(thresh_j,j)
    thresh_data=pd.DataFrame([thresh_i,thresh_j])
    thresh_data=thresh_data.T
    thresh_data.to_csv('/ecoderemdvol/HGCal22Data_signal_driven_ttbar_v11/pair/thresh_pair_'+str(cut_int)+'_indices.csv',index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
