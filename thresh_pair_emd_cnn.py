"""
Created on Thu Apr 15 08:02:19 2021

@author: Javier Duarte, Rohan Shenoy, UCSD
"""

import numpy as np
import mplhep as hep
import pickle

import os
import pandas as pd

import itertools
import sys
sys.path.insert(0, "../")

import matplotlib
import matplotlib.pyplot as plt

from utils.wafer import plot_wafer as plotWafer

from utils.metrics import emd
from utils.metrics import hexMetric

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, BatchNormalization, Activation, Average, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
        
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class thresh_pair_EMD_CNN:
    
    X1_train=[]
    X2_train=[]
    
    def ittrain(calQ_directory,num_filt, kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d, num_epochs,Loss):
        
        def load_data_2(dir,indices):
            # charge data headers of 48 Input Trigger Cells (TC) 
            CALQ_COLS = ['CALQ_%i'%c for c in range(0, 48)]

            full_data =pd.read_csv(dir, dtype=np.float64, header=0, usecols=[*range(0,48)], names=CALQ_COLS, nrows=10000)
            full_data = full_data[CALQ_COLS].astype('float64')


            #Allow repeated wafers this way

            thresh_data=np.empty([len(indices),48])

            for i in range(len(indices)):
                thresh_data[i]=full_data.iloc[indices[i]]

            data_values = thresh_data

            return data_values
        
        #Truth values
        
        truth_data=pd.read_csv('/ecoderemdvol/HGCal22Data_signal_driven_ttbar_v11/thresh_pair_data.csv')
        
        ind_1=np.asarray(truth_data.iloc[0],dtype='int')
        ind_2=np.asarray(truth_data.iloc[1],dtype='int')
        emd_values=np.asarray(truth_data.iloc[2])
        
        current_directory='/ecoderemdvol/22EMD/thresh_pair/'
                
        #Get calQ_data, splitting into desired indices
        
        Q1_data=load_data_2(calQ_directory,ind_1)
        Q2_data=load_data_2(calQ_directory,ind_2)
        
        #Arranging the hexagon
        arrange443 = np.array([0,16, 32,
                               1,17, 33,
                               2,18, 34,
                               3,19, 35,
                               4,20, 36,
                               5,21, 37,
                               6,22, 38,
                               7,23, 39,
                               8,24, 40,
                               9,25, 41,
                               10,26, 42,
                               11,27, 43,
                               12,28, 44,
                               13,29, 45,
                               14,30, 46,
                               15,31, 47])
        fig=plt.figure()
        fig=plt.hist(emd_values, alpha=1, bins=np.arange(0, 4,0.05), label='TrueEMD')
        fig=plt.xlabel('EMD [GeV]')
        fig=plt.ylabel('Samples')
        fig=plt.legend()
        plt.savefig(os.path.join(current_directory,'TrueEMD.png'))
        plt.show()
        
        calQ1     = Q1_data
        sumQ1     = calQ1.sum(axis=1)
        calQ1     = calQ1[sumQ1>0]
        sumQ1     = sumQ1[sumQ1>0]

        calQ1_443 = (calQ1/np.expand_dims(sumQ1,-1))[:,arrange443].reshape(-1,4,4,3)
        
        calQ2     = Q2_data
        sumQ2     = calQ2.sum(axis=1)
        calQ2     = calQ2[sumQ2>0]
        sumQ2     = sumQ2[sumQ2>0]

        calQ2_443 = (calQ2/np.expand_dims(sumQ2,-1))[:,arrange443].reshape(-1,4,4,3)
        
        #Split 70-30 to match pair_emd, these indices are now linearly arranged

        train_indices = range(0, int(0.7*len(calQ1)))
        val_indices = range(int(0.7*len(calQ1)), len(calQ1))

        train_index=int(0.7*len(calQ1))

        idx1_train = np.array([i for i in train_indices])
        idx2_train = np.array([j for j in train_indices])

        X1 = calQ1_443
        X2 = calQ2_443

        X1_train = X1[0:train_index]
        X2_train = X2[0:train_index]

        y_train = np.array([emd(calQ1[i],calQ2[j]) for i, j in zip(train_indices,train_indices)])

        X1_val = X1[train_index:]
        X2_val = X2[train_index:]
        y_val  = np.array([emd(calQ1[i],calQ2[j]) for i, j in zip(val_indices, val_indices)])

        print(X1_train.shape)
        print(X2_train.shape)
        print(y_train.shape)

        print(X1_val.shape)
        print(X2_val.shape)
        print(y_val.shape) 
        
        #Building CNN
        
        # make a convolutional model as a more advanced PoC
        input1 = Input(shape=(4, 4, 3,), name='input_1')
        input2 = Input(shape=(4, 4, 3,), name='input_2')
        x = Concatenate(name='concat')([input1, input2])

        #Number of Conv2D Layers
        for i in range(1,num_conv_2d+1):
            ind=str(i)
            x = Conv2D(num_filt, kernel_size, strides=(1, 1), name='conv2d_'+ind, padding='same', kernel_regularizer=l1_l2(l1=0,l2=1e-4))(x)
            x = BatchNormalization(name='batchnorm_'+ind)(x)
            x = Activation('relu', name='relu_'+ind)(x)

        x = Flatten(name='flatten')(x)

        #Number of Dense Layers
        for i in range(1,num_dens_layers+1):
            ind=str(i)
            jind=str(i+num_conv_2d)
            x = Dense(num_dens_neurons, name='dense_'+ind, kernel_regularizer=l1_l2(l1=0,l2=1e-4))(x)
            x = BatchNormalization(name='batchnorm'+jind)(x)
            x = Activation('relu', name='relu_'+jind)(x)

        output = Dense(1, name='output')(x)
        model = Model(inputs=[input1, input2], outputs=output, name='base_model')
        model.summary()
        
        # make a model that enforces the symmetry of the EMD function by averging the outputs for swapped inputs
        output = Average(name='average')([model((input1, input2)), model((input2, input1))])
        sym_model = Model(inputs=[input1, input2], outputs=output, name='sym_model')
        sym_model.summary()
        
        final_directory=os.path.join(current_directory,r'thresh_pair_emd_models')
        if not os.path.exists(final_directory):
                os.makedirs(final_directory)
        callbacks = [ModelCheckpoint('/ecoderemdvol/22EMD/thresh_pair/thresh_pair_emd_models/'+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+'best.h5', monitor='val_loss', verbose=1, save_best_only=True),
                     ModelCheckpoint('/ecoderemdvol/22EMD/thresh_pair/thresh_pair_emd_models/'+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+'last.h5', monitor='val_loss', verbose=1, save_last_only=True),
                    ]
        
        #opt = tf.keras.optimizers.Adam(learning_rate=4e-3)

        sym_model.compile(optimizer='adam', loss=Loss, metrics=['mse', 'mae', 'mape', 'msle'])
        history = sym_model.fit((X1_train, X2_train), y_train, 
                            validation_data=((X1_val, X2_val), y_val),
                            epochs=num_epochs, verbose=1, batch_size=32, callbacks=callbacks)
        
        #Making directory for graphs

        img_directory=os.path.join(current_directory,r'Threshold Pair EMD Plots')
        if not os.path.exists(img_directory):
            os.makedirs(img_directory)

        #Plot Validation loss and training loss

        plt.close()
        fig=plt.plot(history.history['loss'], label='Train')
        fig=plt.plot(history.history['val_loss'], label='Val.')
        fig=plt.xlabel('Epoch')
        fig=plt.ylabel(Loss+''+'loss')
        fig=plt.legend()
        plt.savefig(img_directory+"/"+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+"Loss.png")
        plt.show()
        plt.close()

        #Plots True EMD and Pred Emd Histogram

        plt.close()
        y_val_preds = model.predict((X1_val, X2_val))
        fig=plt.figure()
        fig=plt.hist(y_val, alpha=0.5, bins=np.arange(0, 4,0.05), label='TrueEMD')
        fig=plt.hist(y_val_preds, alpha=0.5, bins=np.arange(0, 4,0.05), label='EMDCNN')
        fig=plt.xlabel('EMD [GeV]')
        fig=plt.ylabel('Samples')
        fig=plt.legend()
        fig=plt.savefig(img_directory+"/"+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+"Hist.png")
        plt.show()
        plt.close()

        #Plot Relative Difference

        plt.close()
        rel_diff = (y_val_preds[y_val>0].flatten()-y_val[y_val>0].flatten())/y_val[y_val>0].flatten()
        fig=plt.figure()
        fig=plt.hist(rel_diff, bins=np.arange(-1, 1, 0.01), color='green', label = 'mean = {:.3f}, std. = {:.3f}'.format(np.mean(rel_diff), np.std(rel_diff)))
        fig=plt.xlabel('EMD rel. diff.')
        fig=plt.ylabel('Samples')
        fig=plt.legend()
        fig=plt.savefig(img_directory+"/"+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+"RelD.png")
        plt.show()
        plt.close()

        #Plot True EMD vs Pred Emd Graphic

        plt.close()
        fig, ax = plt.subplots(figsize =(5, 5)) 
        x_bins = np.arange(0, 4,0.05)
        y_bins = np.arange(0, 4,0.05)
        plt.hist2d(y_val.flatten(), y_val_preds.flatten(), bins=[x_bins,y_bins])
        plt.plot([0, 15], [0, 15], color='gray', alpha=0.5)
        ax.set_xlabel('True EMD [GeV]')
        ax.set_ylabel('Pred. EMD [GeV]')
        fig=plt.savefig(img_directory+"/"+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+"Graphic.png")
        plt.show()
        plt.close()
        
        return(np.mean(rel_diff),np.std(rel_diff))
    