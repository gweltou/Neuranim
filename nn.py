#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from Box2D.b2 import (vec2, world, circleShape, polygonShape, staticBody, dynamicBody, fixtureDef, pi, contactListener)
from parameters import *


class nnContactListener(contactListener):
    def __init__(self):
        contactListener.__init__(self)
        self.sensors = dict()
    
    def BeginContact(self, contact):
        f1, f2 = contact.fixtureA, contact.fixtureB
        if "ground" in (f1.userData, f2.userData):
            if isinstance(f1.userData, tuple):	# This fixture is an Animatronic sensor
                self.sensors[f1.userData[0]][f1.userData[1]] = 1.0
            if isinstance(f2.userData, tuple):	# This fixture is an Animatronic sensor
                self.sensors[f2.userData[0]][f2.userData[1]] = 1.0
    
    def EndContact(self, contact):
        f1, f2 = contact.fixtureA, contact.fixtureB
        if "ground" in (f1.userData, f2.userData):
            if isinstance(f1.userData, tuple):	# This fixture is an Animatronic sensor
                self.sensors[f1.userData[0]][f1.userData[1]] = 0.0
            if isinstance(f2.userData, tuple):	# This fixture is an Animatronic sensor
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


def save_creatures(filename, creatures, history="", generation=0):
    with open(filename, 'w') as f:
        lines = []
        lines.append('history: {}\n'.format(history))
        for (score, c) in creatures:
            lines.append('####\n')
            lines.append('score: {}\n'.format(score))
            for weight in c.nn.weights:
                lines.append(str(weight.tolist()))
                lines.append('\n')
            lines.append('\n')
        header = []
        header.append('layers: {}\n'.format(creatures[0][1].nn.layers))
        header.append('neurons: {}\n'.format(creatures[0][1].nn.total_neurons))
        header.append('synapses: {}\n'.format(creatures[0][1].nn.total_synapses))
        header.append('generation: {}\n'.format(generation))
        header.append('\n\n')
        f.writelines(header + lines)


def import_creatures(filename, world, startpos):
    history = ""
    creatures = []
    c = None
    layers = NEURON_LAYERS
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        for l in lines:
            if l.startswith('history:'):
                batch_history = l[8:].strip()
            elif l.startswith('layers:'):
                layers = eval(l[7:].strip())
            elif l == '####':
                # New creature definition
                if c:
                    creatures.append(c)
                c = Animatronic(world, position=startpos, layers=layers)
                c.nn.weights = []
            elif l.startswith('[['):
                # Weight array
                c.nn.weights.append(np.array(eval(l)))
        creatures.append(c)
    return creatures, history



class NeuralNetwork:
    def __init__(self, layers):
        # Hyper-parameters
        self.layers = layers
        self.total_neurons = sum(layers)
        self.total_synapses = sum(layers[i]*layers[i+1] for i in range(len(layers)-1))
        
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
        self.output = np.array(x+[1])   # Add the bias unit
        for i in range(0, len(self.weights)-1):
            self.output = self.tanh(np.dot(self.output, self.weights[i]))
            self.output = np.append(self.output, 1.0)   # Add the bias unit
        self.output = self.tanh(np.dot(self.output, self.weights[-1]))
    
    def copy(self):
        new_nn = NeuralNetwork(self.layers)
        new_nn.weights = []
        for w in self.weights:
            new_nn.weights.append(w.copy())
        return new_nn


class Animatronic(object):
    def __init__(self, world, position=(0, 0), layers=NEURON_LAYERS):
        self.world = world
        self.position = vec2(position)
        self.target = vec2(TARGET)
        self.nn = NeuralNetwork(layers)
        self.keeper = False
        self.id = "Animatronic"
    
    def init_body(self):
        """
            Order of defining bodies, joints and sensors is important
            self.joints must be symetrical so it can be reversed for mirror mode
            
            Sensors number (x):
                      (0)-----x-----(1) [[[ BODY ]]] (2)-----x-----(4)
        """
        
        self.bodies = []
        
        self.body = self.world.CreateDynamicBody(position=self.position)
        self.body.CreatePolygonFixture(box=(0.5, 0.5), density=1, friction=0.3,
                                       userData = "body_trunc")
        # Ground/Body sensors
        self.body.CreateCircleFixture(pos=(-0.5, 0.5), radius=0.15,
                                       density=0,
                                       isSensor=True, userData = (self.id, 1))
        self.body.CreateCircleFixture(pos=(0.5, 0.5), radius=0.15,
                                       density=0,
                                       isSensor=True, userData = (self.id, 2))
        self.bodies.append(self.body)
        
        # Legs
        self.lleg = self.world.CreateDynamicBody(position=self.position)
        fixture = self.lleg.CreatePolygonFixture(box=(0.3, 0.15), density=1, friction=0.3,
                                                 userData = "lleg")
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.lleg)
        
        self.rleg = self.world.CreateDynamicBody(position=self.position)
        fixture = self.rleg.CreatePolygonFixture(box=(0.3, 0.15), density=1, friction=0.3)
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.rleg)
        
        # Feet
        self.lfoot = self.world.CreateDynamicBody(position=self.position)
        fixture = self.lfoot.CreatePolygonFixture(box=(0.36, 0.08), density=1, friction=0.3,
                                                  userData = "lfoot")
        fixture.filterData.groupIndex = -1
        
        ## Ground/Foot sensor
        self.lfoot.CreateCircleFixture(pos=(-0.36, 0), radius=0.15,
                                       density=1, friction=1.0, restitution=0.0,
                                       userData = (self.id, 0), groupIndex=-1)
        
        self.rfoot = self.world.CreateDynamicBody(position=self.position)
        fixture = self.rfoot.CreatePolygonFixture(box=(0.36, 0.08), density=1, friction=0.3,
                                                  userData = "rfoot")
        fixture.filterData.groupIndex = -1
        
        ## Ground/Foot sensor
        self.rfoot.CreateCircleFixture(pos=(0.36, 0), radius=0.15,
                                       density=1, friction=1.0, restitution=0.0,
                                       userData = (self.id, 3), groupIndex=-1)
        self.bodies.append(self.lfoot)
        self.bodies.append(self.rfoot)
        
        self.n_sensors = 4
        self.world.contactListener.registerSensors(self.id, self.n_sensors)
        
        self.joints = []
        self.joints.append(
            self.world.CreateRevoluteJoint(
                bodyA = self.lleg,
                bodyB = self.lfoot,
                collideConnected = False,
                localAnchorA = (-0.3, 0),
                localAnchorB = (0.36, 0),
                referenceAngle = pi,
                enableMotor = True,
                motorSpeed = 0.0,
                maxMotorTorque = 10.0
            ))
        self.joints.append(
            self.world.CreateRevoluteJoint(
                bodyA = self.body,
                bodyB = self.lleg,
                collideConnected = False,
                localAnchorA = (-0.4, -0.45),
                localAnchorB = (0.3, 0),
                referenceAngle = pi,
                enableMotor = True,
                motorSpeed = 0.0,
                maxMotorTorque = 20.0
            ))
        self.joints.append(
            self.world.CreateRevoluteJoint(
                bodyA = self.body,
                bodyB = self.rleg,
                collideConnected = False,
                localAnchorA = (0.4, -0.45),
                localAnchorB = (-0.3, 0),
                referenceAngle = pi,
                enableMotor = True,
                motorSpeed = 0.0,
                maxMotorTorque = 20.0
            ))
        self.joints.append(
            self.world.CreateRevoluteJoint(
                bodyA = self.rleg,
                bodyB = self.rfoot,
                collideConnected = False,
                localAnchorA = (0.3, 0),
                localAnchorB = (-0.36, 0),
                referenceAngle = pi,
                enableMotor = True,
                motorSpeed = 0.0,
                maxMotorTorque = 10.0
            ))
    
    def set_target(self, pos):
        self.target = vec2(pos)
    
    def update(self, list_sensors, mirror=False):
        dpos = self.target - self.body.position
        #dpos = self.target
        if dpos.length > 1:
            dpos.Normalize()
        joint_angles = [(j.angle%(2*pi))/pi - 1 for j in self.joints]
        # Make the limbs angle list symmetric (second half *= -1)
        for i in range(len(joint_angles)//2 + len(joint_angles)%2, len(joint_angles)):
            joint_angles[i] *= -1
        
        to_nn = [dpos.x, dpos.y] + joint_angles + list_sensors
        if dpos.x < 0 or mirror:
            # Mirror mode
            joint_angles = joint_angles[::-1]
            to_nn = [-dpos.x, dpos.y] + joint_angles + list_sensors[::-1]
        
        self.nn.feedforward(to_nn)
        if dpos.x < 0 or mirror:
            # Mirror mode
            self.joints[0].motorSpeed = -self.nn.output[3]*20
            self.joints[1].motorSpeed = -self.nn.output[2]*20
            self.joints[2].motorSpeed = -self.nn.output[1]*20
            self.joints[3].motorSpeed = -self.nn.output[0]*20
        else:
            for i in range(len(self.joints)):
                self.joints[i].motorSpeed = self.nn.output[i]*20
    
    def breed(self, other):
        nn = NeuralNetwork(NEURON_LAYERS)
        nn.weights = []
        for w1, w2 in zip(self.nn.weights, other.nn.weights):
            nn.weights.append(cross2(w1, w2))
        
        child = Animatronic(self.world, self.position)
        child.nn = nn
        child.mutate()
        return child
    
    def copy(self):
        duplicate = Animatronic(self.world, self.position)
        nn = self.nn.copy()
        duplicate.nn = nn
        return duplicate
    
    def mutate(self):
        for w in self.nn.weights:
            wf = w.flat
            for i in range(w.size):
                if np.random.randint(self.nn.total_synapses//2) == 0:
                    wf[i] = np.random.random()*2 - 1
    
    def destroy_body(self):
        for joint in self.joints:
            self.world.DestroyJoint(joint)
        for body in self.bodies:
            self.world.DestroyBody(body)
        self.world.contactListener.unregisterSensors(self.id)
