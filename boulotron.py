#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Box2D.b2 import (pi)
from nn import Animatronic


def import_creatures(filename, world, startpos):
    history = ""
    creatures = []
    c = None
    layers = []
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        for l in lines:
            if l.startswith('history:'):
                history = l[8:].strip()
            elif l.startswith('layers:'):
                layers = eval(l[7:].strip())
            elif l == '####':
                # New creature definition
                if c:
                    creatures.append(c)
                c = Boulotron2000(world, position=startpos, layers=layers)
                c.nn.weights = []
            elif l.startswith('[['):
                # Weight array
                c.nn.weights.append(eval(l))
        creatures.append(c)
    return creatures, history


class Boulotron2000(Animatronic):
    """"
         Neural network input layer:
             [pos.x] [pos.y] [joints × 6] [sensors × 6]
    """
    
    def __init__(self, *args, **kwargs):
        super(Boulotron2000, self).__init__(*args, **kwargs)
        self.id = "Boulotron2000"
    
    def init_body(self):
        """
            Order of defining joints and sensors is important
            self.joints must be symetrical so it can be reversed for mirror mode
            
            Sensors number (n):
                      (0)---X(1)-----x-----(2) [[[ BODY ]]] (3)-----x-----(4)X---(5)
        """
        
        self.bodies = []
        
        self.body = self.world.CreateDynamicBody(position=self.position)
        self.body.CreateCircleFixture(pos=(0.0, 0.0), radius=0.6, density=1.0,
                                      userData = "body_trunc")
        self.bodies.append(self.body)
        
        
        ## Ground/Body sensors
        self.body.CreateCircleFixture(pos=(-0.58, -0.1), radius=0.15,
                                       density=0,
                                       isSensor=True, userData = (self.id, 1))
        self.body.CreateCircleFixture(pos=(0.58, -0.1), radius=0.15,
                                       density=0,
                                       isSensor=True, userData = (self.id, 2))
        
        
        # Thighs
        self.lthigh = self.world.CreateDynamicBody(position=self.position)
        fixture = self.lthigh.CreatePolygonFixture(box=(0.3, 0.15), density=1, friction=0.3,
                                                 userData = "lthigh")
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.lthigh)
        
        self.rthigh = self.world.CreateDynamicBody(position=self.position)
        fixture = self.rthigh.CreatePolygonFixture(box=(0.3, 0.15), density=1, friction=0.3)
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.rthigh)
        
        
        # Legs
        self.lleg = self.world.CreateDynamicBody(position=self.position)
        fixture = self.lleg.CreatePolygonFixture(box=(0.36, 0.1), density=1, friction=0.3,
                                                  userData = "lthigh")
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.lleg)
        
        self.rleg = self.world.CreateDynamicBody(position=self.position)
        fixture = self.rleg.CreatePolygonFixture(box=(0.36, 0.1), density=1, friction=0.3,
                                                  userData = "rleg")
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.rleg)
        
        
        ## Heel sensors
        self.lleg.CreateCircleFixture(pos=(-0.36, 0), radius=0.15,
                                       density=1, friction=1.0, restitution=0.0,
                                       userData = (self.id, 0), groupIndex=-1)
        self.rleg.CreateCircleFixture(pos=(0.36, 0), radius=0.15,
                                       density=1, friction=1.0, restitution=0.0,
                                       userData = (self.id, 3), groupIndex=-1)        
        
        
        # Feet
        self.lfoot  = self.world.CreateDynamicBody(position=self.position)
        fixture = self.lfoot.CreatePolygonFixture(box=(0.2, 0.08), density=1, friction=0.3,
                                                  userData = "lfoot")
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.lfoot)
        
        self.rfoot = self.world.CreateDynamicBody(position=self.position)
        fixture = self.rfoot.CreatePolygonFixture(box=(0.2, 0.08), density=1, friction=0.3,
                                                  userData = "rfoot")
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.rfoot)
        
        
        ## Feet sensors
        self.lfoot.CreateCircleFixture(pos=(-0.2, 0), radius=0.15,
                                       density=1, friction=1.0, restitution=0.0,
                                       userData = (self.id, 0), groupIndex=-1)
        self.rfoot.CreateCircleFixture(pos=(0.2, 0), radius=0.15,
                                       density=1, friction=1.0, restitution=0.0,
                                       userData = (self.id, 3), groupIndex=-1)
       
        
        self.n_sensors = 6
        self.world.contactListener.registerSensors(self.id, self.n_sensors)
        
        self.joints = []
        self.joints.append(
            self.world.CreateRevoluteJoint(
                bodyA = self.lleg,
                bodyB = self.lfoot,
                collideConnected = False,
                localAnchorA = (-0.36, 0),
                localAnchorB = (0.2, 0),
                referenceAngle = pi,
                enableMotor = True,
                motorSpeed = 0.0,
                maxMotorTorque = 10.0
            ))
        self.joints.append(
            self.world.CreateRevoluteJoint(
                bodyA = self.lthigh,
                bodyB = self.lleg,
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
                bodyB = self.lthigh,
                collideConnected = False,
                localAnchorA = (0.0, -0.55),
                localAnchorB = (0.3, 0),
                referenceAngle = pi,
                enableMotor = True,
                motorSpeed = 0.0,
                maxMotorTorque = 20.0
            ))
        self.joints.append(
            self.world.CreateRevoluteJoint(
                bodyA = self.body,
                bodyB = self.rthigh,
                collideConnected = False,
                localAnchorA = (0.0, -0.55),
                localAnchorB = (-0.3, 0),
                referenceAngle = pi,
                enableMotor = True,
                motorSpeed = 0.0,
                maxMotorTorque = 20.0
            ))
        self.joints.append(
            self.world.CreateRevoluteJoint(
                bodyA = self.rthigh,
                bodyB = self.rleg,
                collideConnected = False,
                localAnchorA = (0.3, 0),
                localAnchorB = (-0.36, 0),
                referenceAngle = pi,
                enableMotor = True,
                motorSpeed = 0.0,
                maxMotorTorque = 10.0
            ))
        self.joints.append(
            self.world.CreateRevoluteJoint(
                bodyA = self.rleg,
                bodyB = self.rfoot,
                collideConnected = False,
                localAnchorA = (0.36, 0),
                localAnchorB = (-0.2, 0),
                referenceAngle = pi,
                enableMotor = True,
                motorSpeed = 0.0,
                maxMotorTorque = 10.0
            ))
    
    
    
    def update(self, list_sensors, mirror=False):
        dpos = self.target - self.body.position
        if dpos.length > 1:                          # Radius of sight
            dpos.Normalize()
        joint_angles = [(j.angle%(2*pi))/pi - 1 for j in self.joints]
        # Make the limbs angle list symmetric (second half *= -1)
        for i in range(len(joint_angles)//2 + len(joint_angles)%2, len(joint_angles)):
            joint_angles[i] *= -1
        
        to_nn = [dpos.x, dpos.y] + joint_angles + list_sensors
        if dpos.x < 0 or mirror:
            # Mirror mode
            to_nn = [-dpos.x, dpos.y] + joint_angles[::-1] + list_sensors[::-1]
        
        # Send input to neural network
        self.nn.feedforward(to_nn)
        
        # Read output from neural network
        if dpos.x < 0 or mirror:
            # Mirror mode
            self.joints[0].motorSpeed = -self.nn.output[5]*5.0
            self.joints[1].motorSpeed = -self.nn.output[4]*15.0
            self.joints[2].motorSpeed = -self.nn.output[3]*15.0
            self.joints[3].motorSpeed = -self.nn.output[2]*15.0
            self.joints[4].motorSpeed = -self.nn.output[1]*15.0
            self.joints[5].motorSpeed = -self.nn.output[0]*5.0
        else:
            self.joints[0].motorSpeed = -self.nn.output[0]*5.0
            self.joints[1].motorSpeed = -self.nn.output[1]*15.0
            self.joints[2].motorSpeed = -self.nn.output[2]*15.0
            self.joints[3].motorSpeed = -self.nn.output[3]*15.0
            self.joints[4].motorSpeed = -self.nn.output[4]*15.0
            self.joints[5].motorSpeed = -self.nn.output[5]*5.0

