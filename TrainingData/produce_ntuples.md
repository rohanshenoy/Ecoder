## Setup at cmslpc 
For producing raw event data and submitting crab jobs.

Step 1: Setting up the CMS software and producing MC electron gun samples:
```
export SCRAM_ARCH=slc7_amd64_gcc820
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsrel CMSSW_11_0_0_patch1
cd CMSSW_11_0_0_patch1/src/
cmsenv
scram b -j 10
```

Get configuration files for crab jobs:
```
cp -r /uscms/home/cmantill/nobackup/hgcal/CMSSW_11_0_0_patch1/src/Configuration/ .
cp /uscms/home/cmantill/nobackup/hgcal/CMSSW_11_0_0_patch1/src/EGM-Phase2HLTDRWinter20GSDIGI-NoPU_cfg.py .
cp /uscms/home/cmantill/nobackup/hgcal/CMSSW_11_0_0_patch1/src/EGM-Phase2HLTDRWinter20GSDIGI-PU200-FlatEta_cfg.py .
cp /uscms/home/cmantill/nobackup/hgcal/CMSSW_11_0_0_patch1/src/crab_submit.py .
mkdir crab/
```
Setup crab and grid certificate and submit
```
source /cvmfs/cms.cern.ch/common/crab-setup.sh
voms-proxy-init --voms cms --valid 168:00
python crab_submit.py
```

The output of these MC simulations will be used to produce the HGCal ntuples. For eg: 
```
/EGM-Phase2HLTTDRWinter20GS-200PU-flat_etaphi/rshenoy-crab_EGM-Phase2HLTTDRWinter20GS-00012-200PU-May28-578888d15031768497f573fa59e0a252/USER
```

Step 2: producing HGCal ntuple data.

Setup:
```
export SCRAM_ARCH=slc7_amd64_gcc900 
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsrel CMSSW_12_0_0_pre3
cd CMSSW_12_0_0_pre3/src/
cmsenv
git cms-init
git cms-merge-topic -u dnoonan08:AE-evalOnADC
scram b -j4
```

Get the configuration files
```
cd CMSSW_12_0_0_pre3/src/L1Trigger/L1THGCalUtilities/test
# copy fragment file
#cp /uscms/home/cmantill/nobackup/hgcal/CMSSW_12_0_0_pre3/src/L1Trigger/L1THGCalUtilities/test/produce_ntuple_fromelectrongun_genmatch_econdata_v11_cfg.py .
#cp /uscms/home/cmantill/nobackup/hgcal/CMSSW_12_0_0_pre3/src/L1Trigger/L1THGCalUtilities/test/produce_ntuple_fromelectrongun_genmatch_threshold0_v11_cfg.py .
# test that it works locally
#cmsRun produce_ntuple_fromelectrongun_genmatch_econdata_v11_cfg.py
# copy crab config and modify so that you can run on your new dataset
#cp /uscms/home/cmantill/nobackup/hgcal/CMSSW_12_0_0_pre3/src/L1Trigger/L1THGCalUtilities/test/eleGunCrabConfig.py .
# open eleGunCrabConfig.py and modify config.Data.inputDataset,config.Data.outLFNDirBase,config.General.requestName
# setup crab and grid certificate
source /cvmfs/cms.cern.ch/common/crab-setup.sh
voms-proxy-init --voms cms --valid 168:00
# submit crab job (once eleGunCrabConfig.py has been modified)
crab submit eleGunCrabConfig.py
```
Use these ntuples to produce training data.
