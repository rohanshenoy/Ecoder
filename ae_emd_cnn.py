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

class ae_EMD_CNN:
    
    X1_train=[]
    X2_train=[]
    
    def ittrain(test_ae_directory,num_filt, kernel_size, num_dens_neurons, num_dens_layers, num_conv_2d, num_epochs,Loss):
        
        def load_data(inputFile):
            
            data=pd.read_csv(inputFile,usecols=[*range(0,48)])
            data_values=data.values
            
            return data_values
        
        current_directory='/ecoderemdvol/22EMD/ae'
        
        #Take dataset PASSED from previous Autoencoder Training
        csv_directory=os.path.join(test_ae_directory,'8x8_c8_S2_tele')
        input_loc=os.path.join(csv_directory,'verify_input_calQ.csv')

        q_input_data=load_data(input_loc)

        print(q_input_data.shape)

        remap_8x8 = [4, 12, 20, 28,  5, 13, 21, 29,  6, 14, 22, 30,  7, 15, 23, 31, 
                     24, 25, 26, 27, 16, 17, 18, 19,  8,  9, 10, 11,  0,  1,  2,  3, 
                     59, 51, 43, 35, 58, 50, 42, 34, 57, 49, 41, 33, 56, 48, 40, 32]

        output_loc=os.path.join(csv_directory,'verify_decoded_calQ.csv')
        ae_input_data=load_data(output_loc)
        print(ae_input_data.shape)
        
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
        
        #Get True EMD Values
        indices = range(0, len(q_input_data))
        emd_values = np.array([emd(q_input_data[i],ae_input_data[j]) for i, j in zip(indices,indices)])
        
        #Plot True EMD for input vs AE(input)

        fig=plt.figure()
        fig=plt.hist(emd_values, alpha=1, bins=np.arange(0, 2.5,0.01), label='TrueEMD')
        fig=plt.xlabel('EMD [GeV]')
        fig=plt.ylabel('Samples')
        fig=plt.legend()
        plt.savefig(os.path.join(current_directory,'TrueEMD.png'))
        plt.show()
        
        calQ     = q_input_data
        sumQ     = calQ.sum(axis=1)
        calQ     = calQ[sumQ>0]
        sumQ     = sumQ[sumQ>0]

        calQ_443 = (calQ/np.expand_dims(sumQ,-1))[:,arrange443].reshape(-1,4,4,3)

        calA     = ae_input_data
        sumA     = tf.math.reduce_sum(calA, axis=1)
        calA     = calA[sumA>0]
        sumA     = sumA[sumA>0]

        calA_443 = (calA/tf.expand_dims(sumA,-1))
        r = tf.gather(calA_443, arrange443, axis=1)
        r = tf.reshape(r, (-1, 4, 4, 3))
        calA_443=r

        train_indices = range(0, int(0.6*len(calQ)))
        val_indices = range(int(0.6*len(calQ)), len(calQ))

        train_index=int(0.6*len(calQ))

        idx1_train = np.array([i for i in train_indices])
        idx2_train = np.array([j for j in train_indices])

        X1 = calQ_443
        X2 = calA_443

        X1_train = X1[0:train_index]
        X2_train = X2[0:train_index]

        y_train = np.array([emd(calQ[i],calA[j]) for i, j in zip(train_indices,train_indices)])

        X1_val = X1[train_index:]
        X2_val = X2[train_index:]
        y_val = np.array([emd(calQ[i],calA[j]) for i, j in zip(val_indices, val_indices)])

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
        
        final_directory=os.path.join(current_directory,r'ae_emd_models')
        if not os.path.exists(final_directory):
                os.makedirs(final_directory)
        callbacks = [ModelCheckpoint('/ecoderemdvol/22EMD/ae/ae_emd_models/'+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+'best.h5', monitor='val_loss', verbose=1, save_best_only=True),
                     ModelCheckpoint('/ecoderemdvol/22EMD/ae/ae_emd_models/'+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+'last.h5', monitor='val_loss', verbose=1, save_last_only=True),
                    ]

        sym_model.compile(optimizer='adam', loss=Loss, metrics=['mse', 'mae', 'mape', 'msle'])
        history = sym_model.fit((X1_train, X2_train), y_train, 
                            validation_data=((X1_val, X2_val), y_val),
                            epochs=num_epochs, verbose=1, batch_size=128, callbacks=callbacks)
        
        #Making directory for graphs

        img_directory=os.path.join(current_directory,r'AE EMD Plots')
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
        fig=plt.hist(y_val, alpha=0.5, bins=np.arange(0, 2.5,0.01), label='TrueEMD')
        fig=plt.hist(y_val_preds, alpha=0.5, bins=np.arange(0, 2.5,0.01), label='EMDCNN')
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
        x_bins = np.arange(0, 2.5, 0.01)
        y_bins = np.arange(0, 2.5, 0.01)
        plt.hist2d(y_val.flatten(), y_val_preds.flatten(), bins=[x_bins,y_bins])
        plt.plot([0, 15], [0, 15], color='gray', alpha=0.5)
        ax.set_xlabel('True EMD [GeV]')
        ax.set_ylabel('Pred. EMD [GeV]')
        fig=plt.savefig(img_directory+"/"+str(num_filt)+str(kernel_size)+str(num_dens_neurons)+str(num_dens_layers)+str(num_conv_2d)+str(num_epochs)+Loss+"Graphic.png")
        plt.show()
        plt.close()
        
        return(np.mean(rel_diff),np.std(rel_diff))
    
