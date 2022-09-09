import os
import uproot
import numpy as np
import pandas as pd
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, BaseSchema

def loadTrainingData(inputRoot,
                     rootFileTDirectory='FloatingpointThreshold0DummyHistomaxGenmatchGenclustersntuple',
                     outputFileName='CALQ.csv',
                     N_eLinks=5,
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

    pt_mask = (gen_pt <=gen_pt_max)[:,0]
    eta_min_mask = (gen_eta[:,0] >=abs_eta_min)
    eta_max_mask = (gen_eta[:,0] <= abs_eta_max)
    mask = ak.Array(np.logical_and(np.asarray(pt_mask),np.asarray(eta_min_mask),np.asarray(eta_max_mask)))

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
            'tc_simenergy': events['tc_simenergy'][mask],
            'tc_eta': events['tc_eta'][mask]
        }
    )

    select_eLinks = {5 : (tc_mask_data[b'tc_layer']==9),
                     4 : (tc_mask_data[b'tc_layer']==7) | (tc_mask_data[b'tc_layer']==11),
                     3 : (tc_mask_data[b'tc_layer']==13),
                     2 : (tc_mask_data[b'tc_layer']<7) | (tc_mask_data[b'tc_layer']>13),
                     -1 : (tc_mask_data[b'tc_layer']>0)}

    assert N_eLinks in select_eLinks

    df = ak.to_pandas(tc_mask_data[select_eLinks[N_eLinks]])

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

    #Mapping wafer_u,v to physical coordinates
    dfEtaPhi=pd.read_csv('/ecoderemdvol/WaferEtaPhiMap.csv')
    dfTrainData=dfTrainData.merge(dfEtaPhi, on=['layer','waferu','waferv'])
    dfTrainData.reset_index(drop=True,inplace=True)
    dfTrainData.to_csv(outputFileName,index=False)
    
    return 1

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',dest='inputRoot', default='ntuple.root', help="TPG Ntuple directory/file to process")
    parser.add_argument('-d','--dir',dest='rootFileTDirectory', default='FloatingpointThreshold0DummyHistomaxGenmatchGenclustersntuple', help="Directory within input root file to find HGCalTriggerNtuple TTree")
    parser.add_argument('-N',dest='N_eLinks',type=int,default=5,help='Number of eRx to select')
    parser.add_argument('--ADC',dest='useADC',default=False,action='store_true',help='Use ADC rather than transverse ADC')
    parser.add_argument('-o','--output',dest='outputFileName',default='CALQ.csv',help='Output file name (either a .csv or .pkl file name)')
    parser.add_argument('--pt_max',  dest='gen_pt_max',  type=float, default=200, help='maximum gen_pt')
    parser.add_argument('--eta_min', dest='abs_eta_min', type=float, default=1.6, help='minimum gen_eta')
    parser.add_argument('--eta_max', dest='abs_eta_max', type=float, default=3.0, help='maximum gen_eta')
    
    args = parser.parse_args()

    out = loadTrainingData(inputRoot = args.inputRoot,
                     rootFileTDirectory = args.rootFileTDirectory,
                     outputFileName = args.outputFileName,
                     N_eLinks = args.N_eLinks,
                     gen_pt_max = args.gen_pt_max,
                     abs_eta_min = args.abs_eta_min,
                     abs_eta_max = args.abs_eta_max,
                     useADC = args.useADC)
