## Load Training Data

Loads data from root ntuples into a format usable for training.

python3 produceTrainingData.py -i /ecoderemdvol/HGCal22Data_signal_driven_ttbar_v11/ -o ./testmerge.csv

### Input data

Input data comes from the output of the HGCAL TPG ntuplizer: https://twiki.cern.ch/twiki/bin/viewauth/CMS/HGCALTriggerPrimitivesSimulation

Also found here: 

```
/eos/uscms/store/user/lpchgcal/ConcentratorNtuples/L1THGCal_Ntuples/TTbar_v11/
```

The ntuplizer should be run with a threshold sum algorithm, with the threshold set to 0, such that all trigger cells are saved in the resulting root file.

```
python3 produceTrainingData.py -i /TTbar_v11/ -o ./5Elinks_Layer9_test0.csv
```
