# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 18:37:04 2018

@author: dranaju
"""

# Importando bibliotecas
import numpy as np
import random
import os

# Criando a rede neural

class Network():
    def __init__(self, input_size, nb_action):
        self.inputLayerSize = input_size
        self.hiddenLayerSize = 30
        self.outputLayerSize = nb_action
        self.W_hid = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.b_hid = np.random.randn(1,self.hiddenLayerSize)
        self.W_out = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        self.b_out = np.random.randn(1,self.outputLayerSize)
        self.eta = 0.001
        
    def forward(self, X):
        # Propagates the inputs through network
        self.s_hid = np.dot(X, self.W_hid) + self.b_hid #150x8
        self.z_hid = self.sigmoid(self.s_hid) #150x8
        self.s_out = np.dot(self.z_hid,self.W_out) + self.b_out #150x3
        self.z_out = self.sigmoid(self.s_out) #150x3
        self.z_out = self.softmax(self.z_out*10)       
        #yHat = self.oneHot(self.z_out)
        yHat = self.z_out
        return yHat
        
    def calculateDelta_out(self, y):
        self.delta_out = (self.z_out - y)*self.d_sigmoid(self.z_out) #150x3
    
    def calculateDelta_hid(self):
        self.delta_hid = np.dot(self.delta_out, self.W_out.T)*self.d_sigmoid(self.z_hid) #150x8
        
    def updateW_and_b(self, X):
        self.W_out += - self.eta*np.dot(self.z_hid.T, self.delta_out)
        self.b_out += - self.eta*(self.delta_out)
        self.W_hid += - self.eta*np.dot(X.T, self.delta_hid)
        self.b_hid += - self.eta*(self.delta_hid)
    
    def sigmoid(self, z):
        # Apply the sigmoid fuction to a scalar, vector or matrix
        return 1/(1+np.exp(-z))
    
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))
    
    def d_sigmoid(self, z):
        return z*(1-z)
    
    def oneHot(self, x):
        y = np.array([np.zeros(self.outputLayerSize)])
        imax = np.argmax(x[0])
        for i in range(len(x[0])):
            if (i == imax):
                y[0][i] = 1.
            else:
                y[0][i] = 0.
        return y
    
    def train(self, inputs, outputs):
        Loss = 0.0
        for i in range(inputs.shape[0]):
            x = np.reshape(inputs[i,:],(1,30))
            y = np.reshape(outputs[i,:],(1,2))
            self.forward(x)
            #print self.z_out
            #Loss += np.sum((self.z_out - y)**2)*0.5
            Loss += -np.sum(np.dot(y,np.log(self.z_out).T))
            #print Loss
            self.calculateDelta_out(y)
            self.calculateDelta_hid()
            self.updateW_and_b(x)
        return Loss
        
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.last_state = 0.
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = self.model.forward(state)
        action = np.argmax(probs)
        return action
        
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model.forward(batch_state)
        next_outputs = self.model.forward(batch_next_state)
        target = self.gamma*next_outputs + batch_reward
        td_loss = 0.5*((target - outputs)**2)
        
    def update(self, reward, new_signal):
        #print(reward)
        self.memory.push()
        action = self.select_action(new_state)
        
        