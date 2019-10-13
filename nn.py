#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
#from Box2D.b2 import (vec2, world, circleShape, polygonShape, staticBody, dynamicBody, fixtureDef, pi, contactListener)
from Box2D.b2 import contactListener
from parameters import *



class nnContactListener(contactListener):
    def __init__(self):
        contactListener.__init__(self)
        self.sensors = dict()
    
    def BeginContact(self, contact):
        f1, f2 = contact.fixtureA, contact.fixtureB
        if "ground" in (f1.userData, f2.userData):
            if isinstance(f1.userData, tuple):	
                # This fixture is an Animatronic sensor
                self.sensors[f1.userData[0]][f1.userData[1]] = 1.0
            if isinstance(f2.userData, tuple):
                # This fixture is an Animatronic sensor
                self.sensors[f2.userData[0]][f2.userData[1]] = 1.0
    
    def EndContact(self, contact):
        f1, f2 = contact.fixtureA, contact.fixtureB
        if "ground" in (f1.userData, f2.userData):
            if isinstance(f1.userData, tuple):
                # This fixture is an Animatronic sensor
                self.sensors[f1.userData[0]][f1.userData[1]] = 0.0
            if isinstance(f2.userData, tuple):
                # This fixture is an Animatronic sensor
                self.sensors[f2.userData[0]][f2.userData[1]] = 0.0
    
    def registerSensors(self, id, n):
        """
            Args:
                id: Animatronic unique identifier
                n: number of sensor to register
        """
        self.sensors[id] = [0.0]*n
    
    def unregisterSensors(self, id):
        del self.sensors[id]


def breed(creatures):
    # This function is weird...
    if len(creatures) < 2:
        return []
    offspring = []
    p1 = creatures[0]
    for p2 in creatures[1:]:
        offspring.append(p1.breed(p2))
    return offspring + breed(creatures[1:])


def cross(array1, array2):
    assert(array1.shape == array2.shape)
    new_list = []
    a1, a2 = array1.flat, array2.flat
    for i in range(array1.size):
        r = np.random.randint(2)
        if r == 0:
            # inherit from first parent
            new_list.append(a1[i])
        if r == 1:
            # inherit from second parent
            new_list.append(a2[i])
    return np.array(new_list).reshape(array1.shape)


def cross2(array1, array2):
    """ Cross function with whole genes instead of single nucleotides """
    assert(array1.shape == array2.shape)
    new_array = np.zeros_like(array1)
    #a1, a2 = array1.flat, array2.flat
    for i in range(array1.shape[1]):
        r = np.random.randint(2)
        if r == 0:
            # inherit from first parent
            new_array[:,i] = array1[:,i].copy()
        if r == 1:
            # inherit from second parent
            new_array[:,i] = array2[:,i].copy()    
    return new_array



class NeuralNetwork:
    def __init__(self, layers, activation="tanh"):
        activations = { "tanh": self.tanh,
                        "sigmoid": self.sigmoid,
                        "sigmoid_derivative": self.sigmoid_derivative,
                        "relu": self.relu}
        
        # Hyper-parameters
        self.layers = layers
        self.total_neurons = sum(layers)
        self.total_synapses = sum(layers[i]*layers[i+1] for i in range(len(layers)-1))
        self.activation = activation.lower()
        if activation in activations:
            self.activation_f = activations[activation]
        
        # Weights
        self.weights = []
        for i in range(len(self.layers)-1):
            self.weights.append(np.random.uniform(size=(self.layers[i]+1, self.layers[i+1]), low=-1, high=1))
    
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))
    
    def tanh(self, x):
        # Better than sigmoid for our purpose
        return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))
    
    def relu(self, x):
        return np.maximum(x, np.zeros_like(x))
    
    def sigmoid_derivative(self, x):
        return x*(1-x)
    
    def feedforward(self, x):
        self.output = np.array(x+[1.0])   # Add the bias unit
        for i in range(0, len(self.weights)-1):
            self.output = self.activation_f(np.dot(self.output, self.weights[i]))
            self.output = np.append(self.output, 1.0)   # Add the bias unit
        self.output = self.activation_f(np.dot(self.output, self.weights[-1]))
    
    def copy(self):
        new_nn = NeuralNetwork(self.layers)
        new_nn.weights = []
        for w in self.weights:
            new_nn.weights.append(w.copy())
        return new_nn

