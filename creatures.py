#!/usr/bin/env python
# -*- coding: utf-8 -*-


from Box2D.b2 import pi, vec2, world, circleShape, polygonShape, staticBody, dynamicBody, fixtureDef
import numpy as np
from parameters import *
from nn import *



def save_creatures(filename, creatures, history="", stats="", generation=0):
    with open(filename, 'w') as f:
        lines = []
        for (score, c) in creatures:
            lines.append('####\n')
            lines.append('score: {}\n'.format(score))
            for weight in c.nn.weights:
                lines.append(str(weight.tolist()))
                lines.append('\n')
            lines.append('\n')
        header = []
        header.append('type: {}\n'.format(creatures[0][1].id))
        header.append('layers: {}\n'.format(creatures[0][1].nn.get_layers()))
        header.append('neurons: {}\n'.format(creatures[0][1].nn.get_total_neurons()))
        header.append('synapses: {}\n'.format(creatures[0][1].nn.get_total_synapses()))
        header.append('activation: {}\n'.format(creatures[0][1].nn.activation))
        header.append('generation: {}\n'.format(generation))
        header.append('history: {}\n'.format(history))
        header.append('stats: {}\n'.format(stats))
        header.append('\n\n')
        f.writelines(header + lines)


def import_creatures(filename, world):
    creatures = []
    c = None
    layers = []
    data = dict()
    creature_type = "Animatronic"
    activation = ACTIVATION
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        for l in lines:
            if l.startswith('history:'):
                data['history'] = l[8:].strip()
            elif l.startswith('layers:'):
                layers = eval(l[7:].strip())
            elif l.startswith("activation:"):
                activation = l[11:].strip()
            elif l.startswith('generation:'):
                data['generation'] = int(l[11:].strip())
            elif l.startswith('stats:'):
                data['stats'] = eval(l[6:].strip())
            elif l.startswith('type:'):
                creature_type = l[5:].strip()
            elif l == '####':
                # New creature definition
                if c:
                    creatures.append(c)
                if creature_type == "Boulotron2000":
                    c = Boulotron2000(world, hidden=[], activation=activation)
                    c.nn.weights = []   # Clear neural network
                elif creature_type == "Animatronic" or creature_type == "Cubotron1000":
                    c = Cubotron1000(world, hidden=[], activation=activation)
                    c.nn.weights = []   # Clear neural network
                else:
                    print("Error: no type in creature definition")
                    sys.exit(1)
            elif l.startswith('[['):
                # Weight array
                c.nn.weights.append(np.array(eval(l)))
        creatures.append(c)
    data['creatures'] = creatures
    data['layers'] = layers
    return data




class Animatronic(object):
    """ Abstract class
    """
    id = "Animatronic" # This should be unique in case of many creatures in the same world
    
    def __init__(self, world):
        self.world = world
        self.score = 0
        self.keeper = False
    
    def set_start_position(self, x, y):
        self.start_position = vec2(x, y)
    
    def set_target(self, x, y):
        self.target = vec2(x, y)
    
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
        duplicate = self.__class__(self.world)
        duplicate.nn = self.nn.copy()
        return duplicate
    
    def mutate(self):
        total_synapses = self.nn.get_total_synapses()
        for w in self.nn.weights:
            wf = w.flat
            for i in range(w.size):
                if np.random.randint(total_synapses//2) == 0:
                    wf[i] = np.random.random()*2 - 1.0
    
    def destroy(self):
        for joint in self.joints:
            self.world.DestroyJoint(joint)
        for body in self.bodies:
            self.world.DestroyBody(body)
        self.world.contactListener.unregisterSensors(self.id)




class Cubotron1000(Animatronic):
    """"
         Neural network input layer:
             [pos.x] [pos.y] [joints × 4] [contact_sensors × 4]
             
             Contact sensors:
                 - lfoot
                 - lheel
                 - lbody
                 - rbody
                 - rheel
                 - rfoot
             Other sensors:
                 - body_angle
    """
    
    def __init__(self, world, hidden=[24, 24, 24], activation="tanh"):
        self.n_sensors = 4
        layers = [2+4+4] + hidden + [self.n_sensors]
        self.nn = NeuralNetwork()
        self.nn.init_weights(layers)
        self.nn.set_activation(activation)
        self.id = "Cubotron1000"
        super().__init__(world)
        
    
    def init_body(self):
        """
            Order of defining joints and sensors is important
            self.joints must be symetrical so it can be reversed for mirror mode
            
            Sensors number (n):
                      (0)-----x-----(1) [[[ BODY ]]] (2)-----x-----(3)
            
        """
        
        self.bodies = []
        
        self.body = self.world.CreateDynamicBody(position=self.start_position)
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
        self.lleg = self.world.CreateDynamicBody(position=self.start_position)
        fixture = self.lleg.CreatePolygonFixture(box=(0.3, 0.15), density=1, friction=0.3,
                                                 userData = "lleg")
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.lleg)
        
        self.rleg = self.world.CreateDynamicBody(position=self.start_position)
        fixture = self.rleg.CreatePolygonFixture(box=(0.3, 0.15), density=1, friction=0.3)
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.rleg)
        
        # Feet
        self.lfoot = self.world.CreateDynamicBody(position=self.start_position)
        fixture = self.lfoot.CreatePolygonFixture(box=(0.36, 0.08), density=1, friction=0.3,
                                                  userData = "lfoot")
        fixture.filterData.groupIndex = -1
        
        ## Ground/Foot sensor
        self.lfoot.CreateCircleFixture(pos=(-0.36, 0), radius=0.15,
                                       density=1, friction=1.0, restitution=0.0,
                                       userData = (self.id, 0), groupIndex=-1)
        
        self.rfoot = self.world.CreateDynamicBody(position=self.start_position)
        fixture = self.rfoot.CreatePolygonFixture(box=(0.36, 0.08), density=1, friction=0.3,
                                                  userData = "rfoot")
        fixture.filterData.groupIndex = -1
        
        ## Ground/Foot sensor
        self.rfoot.CreateCircleFixture(pos=(0.36, 0), radius=0.15,
                                       density=1, friction=1.0, restitution=0.0,
                                       userData = (self.id, 3), groupIndex=-1)
        self.bodies.append(self.lfoot)
        self.bodies.append(self.rfoot)
        
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
    
    
    def update(self, list_sensors, mirror=False):
        dpos = self.target - self.body.position
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





class Boulotron2000(Animatronic):
    """"
         Neural network input layer:
             [pos.x] [pos.y] [joints × 6] [contact_sensors × 6] [body_angle]
             
             Contact sensors:
                 - lfoot
                 - lheel
                 - lbody
                 - rbody
                 - rheel
                 - rfoot
             Other sensors:
                 - body_angle
    """
    
    def __init__(self, world, hidden=[30, 30, 30], activation="tanh"):
        self.n_sensors = 6
        layers = [2+6+6+1] + hidden + [self.n_sensors]
        self.nn = NeuralNetwork()
        self.nn.init_weights(layers)
        self.nn.set_activation(activation)
        self.id = "Boulotron2000"
        super().__init__(world)
    
    
    def init_body(self):
        """
            Order of defining joints and sensors is important
            self.joints must be symetrical so it can be reversed for mirror mode
            
            Sensors number (n):
                  (0)---X(1)-----x-----(2) [[[ BODY ]]] (3)-----x-----(4)X---(5)
        """
        
        self.bodies = []
        
        self.body = self.world.CreateDynamicBody(position=self.start_position)
        self.body.CreateCircleFixture(pos=(0.0, 0.0), radius=0.6, density=1.0,
                                      userData = "body_trunc")
        self.bodies.append(self.body)
        
        
        ## Ground/Body sensors
        self.body.CreateCircleFixture(pos=(-0.58, -0.1), radius=0.15,
                                       density=0,
                                       isSensor=True, userData = (self.id, 2))
        self.body.CreateCircleFixture(pos=(0.58, -0.1), radius=0.15,
                                       density=0,
                                       isSensor=True, userData = (self.id, 3))
        
        
        # Thighs
        self.lthigh = self.world.CreateDynamicBody(position=self.start_position)
        fixture = self.lthigh.CreatePolygonFixture(box=(0.3, 0.15), density=1, friction=0.3,
                                                 userData = "lthigh")
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.lthigh)
        
        self.rthigh = self.world.CreateDynamicBody(position=self.start_position)
        fixture = self.rthigh.CreatePolygonFixture(box=(0.3, 0.15), density=1, friction=0.3)
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.rthigh)
        
        
        # Legs
        self.lleg = self.world.CreateDynamicBody(position=self.start_position)
        fixture = self.lleg.CreatePolygonFixture(box=(0.36, 0.1), density=1, friction=0.3,
                                                  userData = "lthigh")
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.lleg)
        
        self.rleg = self.world.CreateDynamicBody(position=self.start_position)
        fixture = self.rleg.CreatePolygonFixture(box=(0.36, 0.1), density=1, friction=0.3,
                                                  userData = "rleg")
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.rleg)
        
        
        ## Heel sensors
        self.lleg.CreateCircleFixture(pos=(-0.36, 0), radius=0.15,
                                       density=1, friction=1.0, restitution=0.0,
                                       userData = (self.id, 1), groupIndex=-1)
        self.rleg.CreateCircleFixture(pos=(0.36, 0), radius=0.15,
                                       density=1, friction=1.0, restitution=0.0,
                                       userData = (self.id, 4), groupIndex=-1)        
        
        
        # Feet
        self.lfoot  = self.world.CreateDynamicBody(position=self.start_position)
        fixture = self.lfoot.CreatePolygonFixture(box=(0.2, 0.08), density=1, friction=0.3,
                                                  userData = "lfoot")
        fixture.filterData.groupIndex = -1
        self.bodies.append(self.lfoot)
        
        self.rfoot = self.world.CreateDynamicBody(position=self.start_position)
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
                                       userData = (self.id, 5), groupIndex=-1)
        
        # Contact sensors
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
        # Insert body angle sensor
        if self.bodies[0].angle < 0:
            body_angle = (self.bodies[0].angle%(-2*pi)) / (2*pi)
        else:
            body_angle = (self.bodies[0].angle%(2*pi)) / (2*pi)
        
        to_nn = [dpos.x, dpos.y] + joint_angles + list_sensors + [body_angle]
        if dpos.x < 0 or mirror:
            # Mirror mode
            to_nn = [-dpos.x, dpos.y] + joint_angles[::-1] + list_sensors[::-1] + [body_angle]
        
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

