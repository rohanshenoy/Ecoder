import os
import glob
import pandas as pd


tc_data = pd.read_csv('/ecoderemdvol/hgcal_data/all_layers/1.csv')

low_eta = tc_data[tc_data.tc_eta<=2.0]
med_eta = tc_data[tc_data.tc_eta>2.0][tc_data.tc_eta<=2.4]
high_eta = tc_data[tc_data.tc_eta>2.4]
    
lell = low_eta[low_eta.layer<5]
leml = low_eta[low_eta.layer>=5][low_eta.layer<=11]
lehl = low_eta[low_eta.layer>11]
    
mell = med_eta[med_eta.layer<5]
meml = med_eta[med_eta.layer>=5][med_eta.layer<=11]
mehl = med_eta[med_eta.layer>11]
    
hell = high_eta[high_eta.layer<5]
heml = high_eta[high_eta.layer>=5][high_eta.layer<=11]
hehl = high_eta[high_eta.layer>11]

for i in range(2,11):
    file = '/ecoderemdvol/hgcal_data/all_layers/{}.csv'.format(i)
    
    tc_data = pd.read_csv(file)
    
    low_eta = tc_data[tc_data.tc_eta<=2.0]
    med_eta = tc_data[tc_data.tc_eta>2.0][tc_data.tc_eta<=2.4]
    high_eta = tc_data[tc_data.tc_eta>2.4]
    
    lell_i = low_eta[low_eta.layer<5]
    leml_i = low_eta[low_eta.layer>=5][low_eta.layer<=11]
    lehl_i = low_eta[low_eta.layer>11]
    
    mell_i = med_eta[med_eta.layer<5]
    meml_i = med_eta[med_eta.layer>=5][med_eta.layer<=11]
    mehl_i = med_eta[med_eta.layer>11]
    
    hell_i = high_eta[high_eta.layer<5]
    heml_i = high_eta[high_eta.layer>=5][high_eta.layer<=11]
    hehl_i = high_eta[high_eta.layer>11]
    
    lell = pd.concat([lell,lell_i])
    leml = pd.concat([leml,leml_i])
    lehl = pd.concat([lehl,lehl_i])
    
    mell = pd.concat([mell,mell_i])
    meml = pd.concat([meml,leml_i])
    mehl = pd.concat([mehl,mehl_i])
    
    hell = pd.concat([hell,hell_i])
    heml = pd.concat([heml,heml_i])
    hehl = pd.concat([hehl,hehl_i])
    
    
lell.to_csv('/ecoderemdvol/hgcal_data/all_layers/lell.csv')
leml.to_csv('/ecoderemdvol/hgcal_data/all_layers/leml.csv')
lehl.to_csv('/ecoderemdvol/hgcal_data/all_layers/lehl.csv')

mell.to_csv('/ecoderemdvol/hgcal_data/all_layers/mell.csv')
meml.to_csv('/ecoderemdvol/hgcal_data/all_layers/meml.csv')
mehl.to_csv('/ecoderemdvol/hgcal_data/all_layers/mehl.csv')

hell.to_csv('/ecoderemdvol/hgcal_data/all_layers/hell.csv')
heml.to_csv('/ecoderemdvol/hgcal_data/all_layers/heml.csv')
hehl.to_csv('/ecoderemdvol/hgcal_data/all_layers/hehl.csv')

