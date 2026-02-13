#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:37:48 2023

@author: ffiguere
"""
import dolfin as df
import numpy as np
import fetricks as ft
import tensorflow as tf
from hyperNNics.core.hyperelastic_isotropic_material import HyperelasticIsotropicMaterial
tf.keras.backend.set_floatx('float64')

class NNMaterial(HyperelasticIsotropicMaterial):
    def param_parser(self, param):
        self.model = tf.keras.models.load_model(param['NNmodel_file'])
        super().param_parser(param)
        
    # full C
    def get_dpsi(self, C):
        I1, I2, I3, J = ft.get_invariants_iso_np(C)
                
        i1 = tf.cast(I1 - 3 , dtype=tf.float64)
        i2 = tf.cast(I2 - 3 , dtype=tf.float64)
        i3 = tf.cast(I3 - 1 , dtype=tf.float64)
        with tf.GradientTape(persistent = True) as t1:
            t1.watch((i1,i2,i3))
            tensor_data=tf.reshape(tf.cast([i1,i2,i3],dtype=tf.float64),(1,3))
            pred = self.model(tensor_data)
            
        dpsidi1 = t1.gradient(pred, i1).numpy()
        dpsidi2 = t1.gradient(pred, i2).numpy()
        dpsidi3 = t1.gradient(pred, i3).numpy()
        
        # print([dpsidi1, dpsidi2])
        return dpsidi1, dpsidi2, dpsidi3

    def get_dpsi_batch(self, C):
        # I1, I2, I3, J = ft.get_invariants_iso_np(C)
        
        I1 = tf.linalg.trace(C)
        I2 = 0.5*(tf.linalg.trace(C)**2 - tf.linalg.trace(C@C))
        I3 = tf.linalg.det(C)
        J = I3**(0.5)
        
        i1 = tf.cast(I1 - 3 , dtype=tf.float64)
        i2 = tf.cast(I2 - 3 , dtype=tf.float64)
        i3 = tf.cast(I3 - 1 , dtype=tf.float64)
        
        with tf.GradientTape(persistent = True) as t1:
            t1.watch((i1,i2,i3))
            tensor_data=tf.transpose(tf.cast([i1,i2],dtype=tf.float64))
            pred = self.model(tensor_data)
            
        dpsidi1 = t1.gradient(pred, i1).numpy()
        dpsidi2 = t1.gradient(pred, i2).numpy()
        dpsidi3 = t1.gradient(pred, i3).numpy()
        # print([dpsidi1, dpsidi2])
        return dpsidi1, dpsidi2
    # full C
    def get_d2psi(self, C):
        I1, I2, I3, J = ft.get_invariants_iso_np(C)
        
        i1 = tf.cast(I1 - 3 , dtype=tf.float64)
        i2 = tf.cast(I2 - 3 , dtype=tf.float64)
        i3 = tf.cast(I3 - 1 , dtype=tf.float64)
        
        with tf.GradientTape(persistent = True) as t2:
            t2.watch((i1,i2,i3))
            with tf.GradientTape(persistent = True) as t1:
                t1.watch((i1,i2,i3))
                tensor_data=tf.reshape(tf.cast([i1,i2,i3],dtype=tf.float64),(1,3))
                pred = self.model(tensor_data)
                
            dpsidi1 = t1.gradient(pred, i1)
            dpsidi2 = t1.gradient(pred, i2)

        d2psidi1i1 = t2.gradient(dpsidi1, i1).numpy() # Higher order derivative of psi/i1
        d2psidi2i2 = t2.gradient(dpsidi2, i2).numpy() # Higher order derivative of psi/i2
        d2psidi1di2 = t2.gradient(dpsidi1, i2).numpy() # Mixed derivative of psi/i1/i2   
        
        return d2psidi1i1, d2psidi2i2, d2psidi1di2  
    
    def get_d2psi_batch(self, C):
        # I1, I2, I3, J = ft.get_invariants_iso_np(C)
        
        I1 = tf.linalg.trace(C)
        I2 = 0.5*(tf.linalg.trace(C)**2 - tf.linalg.trace(C@C))
        I3 = tf.linalg.det(C)
        # J = I3**(0.5)
        
        i1 = tf.cast(I1 - 3 , dtype=tf.float64)
        i2 = tf.cast(I2 - 3 , dtype=tf.float64)
        i3 = tf.cast(I3 - 1 , dtype=tf.float64)
        
        with tf.GradientTape(persistent = True) as t2:
            t2.watch((i1,i2,i3))
            with tf.GradientTape(persistent = True) as t1:
                t1.watch((i1,i2,i3))
                tensor_data=tf.transpose(tf.cast([i1,i2,i3],dtype=tf.float64))
                pred = self.model(tensor_data)
                
            dpsidi1 = t1.gradient(pred, i1)
            dpsidi2 = t1.gradient(pred, i2)
            dpsidi3 = t1.gradient(pred, i3)

        d2psidi1i1 = t2.gradient(dpsidi1, i1).numpy() # Higher order derivative of psi/i1
        d2psidi2i2 = t2.gradient(dpsidi2, i2).numpy() # Higher order derivative of psi/i2
        d2psidi1di2 = t2.gradient(dpsidi1, i2).numpy() # Mixed derivative of psi/i1/i2   
        d2psidi1di3 = t2.gradient(dpsidi1, i3).numpy()
        d2psidi2di3 = t2.gradient(dpsidi2, i3).numpy()
        d2psidi3di3 = t2.gradient(dpsidi3, i3).numpy()
        return d2psidi1i1, d2psidi2i2, d2psidi1di2, d2psidi1di3, d2psidi2di3, d2psidi3di3
    
    def get_d2psi_batch_2(self, C):
        # I1, I2, I3, J = ft.get_invariants_iso_np(C)
        
        I1 = tf.linalg.trace(C)
        I2 = 0.5*(tf.linalg.trace(C)**2 - tf.linalg.trace(C@C))
        I3 = tf.linalg.det(C)
        # J = I3**(0.5)
        
        i1 = tf.cast(I1 - 3 , dtype=tf.float64)
        i2 = tf.cast(I2 - 3 , dtype=tf.float64)
        i3 = tf.cast(I3 - 1 , dtype=tf.float64)
        
        with tf.GradientTape(persistent = True) as t2:
            t2.watch((i1,i2,i3))
            with tf.GradientTape(persistent = True) as t1:
                t1.watch((i1,i2,i3))
                tensor_data=tf.transpose(tf.cast([i1,i2,i3],dtype=tf.float64))
                pred = self.model(tensor_data)
                
            dpsidi1 = t1.gradient(pred, i1)
            dpsidi2 = t1.gradient(pred, i2)
            dpsidi3 = t1.gradient(pred, i3)
            
        d2psidi1i1 = t2.gradient(dpsidi1, i1).numpy() # Higher order derivative of psi/i1
        d2psidi2i2 = t2.gradient(dpsidi2, i2).numpy() # Higher order derivative of psi/i2
        d2psidi1di2 = t2.gradient(dpsidi1, i2).numpy() # Mixed derivative of psi/i1/i2   
        d2psidi1di3 = t2.gradient(dpsidi1, i3).numpy()
        d2psidi2di3 = t2.gradient(dpsidi2, i3).numpy()
        d2psidi3di3 = t2.gradient(dpsidi3, i3).numpy()
        return dpsidi1.numpy(), dpsidi2.numpy(), dpsidi3.numpy(),d2psidi1i1, d2psidi2i2, d2psidi1di2,d2psidi1di3, d2psidi2di3, d2psidi3di3 
    
    