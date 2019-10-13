import pygame as pg
import numpy as np
from Box2D.b2 import (world, vec2, kinematicBody)
from random import random, choice

PPM = 50.0  # pixels per meter
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480


def px_to_world(pos):
    return vec2(pos[0]/PPM, (SCREEN_HEIGHT-pos[1])/PPM)


def world_to_px(pos):
    return np.array((pos.x*PPM, SCREEN_HEIGHT - pos.y*PPM)).round().astype(int)


def rotate_point(pos, theta):
    c, s = np.cos(theta), np.sin(-theta)
    rotmat = np.array( ((c, -s), (s, c)),
                      dtype=np.float16)
    
    return np.dot(rotmat, pos)


def scale_image(surf, pc):
    w, h = surf.get_size()
    ws, hs = (w * pc/100.0, h * pc/100.0)
    return pg.transform.scale(surf, (int(round(ws)), int(round(hs))))


class Limb():
    def __init__(self, world, surface):
        self.world = world
        self.image = surface # should have pixel alpha
        self.width, self.height = self.image.get_size()
        self.pivots = None
    
    def set_center(self, pos):
        self.center_abs = np.array(pos, dtype=int)
    
    def init_physics(self):
        self.body = self.world.CreateDynamicBody(position=px_to_world(self.center_abs), angle=0)
        self.body.CreatePolygonFixture(box=vec2(self.width/PPM, self.height/PPM)/2, density=1)
        self.body.fixtures[0].filterData.maskBits = 0
    
    def update(self):
        self.center_abs = world_to_px(self.body.position)
        self.rotate(self.body.angle)
    
    def set_pivots(self, *l):
        p = np.array([ np.array((self.width, self.height)) * norm_pos for norm_pos in l ])
        self.pivots = p - (self.width // 2, self.height // 2)
    
    def get_pivot(self, i):
        x, y = self.pivots[i]
        return vec2(x/PPM, -y/PPM)        
    
    def rotate(self, angle):
        self.sprite = pg.transform.rotate(self.image, np.degrees(angle))
        w, h = self.sprite.get_size()
        center = np.array((w//2, h//2))
        self.offset = np.array(self.center_abs) - center
        if len(self.pivots) > 0:
            self.pivots_abs = np.array([rotate_point(pos, angle) for pos in self.pivots])
            self.pivots_abs = self.pivots_abs.round().astype(int)
            self.pivots_abs = self.pivots_abs + self.offset + center
    
    def draw(self, screen):
        screen.blit(self.sprite, self.offset)
        if __debug__:
            if len(self.pivots) > 0:
                for pos in self.pivots_abs:
                    pg.draw.circle(screen, (255,0,0), pos, 8, 1)
            pg.draw.circle(screen, (0,255,0), self.center_abs, 8, 1)
            pg.draw.line(screen, (0,255,0), self.center_abs + (-10, -10), self.center_abs + (10, 10))
            pg.draw.line(screen, (0,255,0), self.center_abs + (10, -10), self.center_abs + (-10, 10))


class Ragdoll(object):
    def __init__(self, world):
        self.lastpos = (0,0)
        self.world = world
        self.headdir = 'left'
        
        self.heads = [scale_image(img.convert_alpha(), 20) for img in (pg.image.load("headleft.png"),
                      pg.image.load("headleft2.png"),
                      pg.image.load("headright.png"))]
        
        self.head = Limb(world, self.heads[0])
        self.head.set_pivots((0.53, 0.93))
        
        img = pg.image.load("torso.png").convert_alpha()
        self.torso = Limb(world, scale_image(img, 20))
        self.torso.set_pivots((0.47, 0.06), (0.1, 0.18), (0.92, 0.16), (0.32, 0.9), (0.75, 0.9))
        
        img = pg.image.load("leftarm.png").convert_alpha()
        self.leftarm = Limb(world, scale_image(img, 20))
        self.leftarm.set_pivots((0.93, 0.09))
        
        img = pg.image.load("rightarm.png").convert_alpha()
        self.rightarm = Limb(world, scale_image(img, 20))
        self.rightarm.set_pivots((0.1, 0.05))
        
        img = pg.image.load("leftleg.png").convert_alpha()
        self.leftleg = Limb(world, scale_image(img, 20))
        self.leftleg.set_pivots((0.48, 0.1), (0.39, 0.93))
        
        img = pg.image.load("rightleg.png").convert_alpha()
        self.rightleg = Limb(world, scale_image(img, 20))
        self.rightleg.set_pivots((0.56, 0.1), (0.64, 0.93))
        
        img = pg.image.load("leftfoot.png").convert_alpha()
        self.leftfoot = Limb(world, scale_image(img, 20))
        self.leftfoot.set_pivots((0.59, 0.09))
        
        img = pg.image.load("rightfoot.png").convert_alpha()
        self.rightfoot = Limb(world, scale_image(img, 20))
        self.rightfoot.set_pivots((0.31, 0.1))
        
        self.limbs = [self.head, self.torso, self.leftarm, self.rightarm, self.leftleg, self.rightleg,
            self.leftfoot, self.rightfoot]
        for l in self.limbs:
            l.set_center((200, 100))
            l.init_physics()
        self.head.body.type = kinematicBody
        self.do_joints() 
    
    def do_joints(self):
        self.world.CreateRevoluteJoint(
            bodyA = self.head.body,
            bodyB = self.torso.body,
            collideConnected = False,
            localAnchorA = self.head.get_pivot(0), localAnchorB = self.torso.get_pivot(0),
            enableMotor = True,
            motorSpeed = 0.0,
            maxMotorTorque = 2.0
        )
        self.world.CreateRevoluteJoint(
            bodyA = self.torso.body,
            bodyB = self.leftarm.body,
            collideConnected = False,
            localAnchorA = self.torso.get_pivot(1), localAnchorB = self.leftarm.get_pivot(0),
            enableMotor = True,
            motorSpeed = 0.0,
            maxMotorTorque = 1.0
        )
        self.world.CreateRevoluteJoint(
            bodyA = self.torso.body,
            bodyB = self.rightarm.body,
            collideConnected = False,
            localAnchorA = self.torso.get_pivot(2), localAnchorB = self.rightarm.get_pivot(0),
            enableMotor = True,
            motorSpeed = 0.0,
            maxMotorTorque = 1.0
        )
        self.world.CreateRevoluteJoint(
            bodyA = self.torso.body,
            bodyB = self.leftleg.body,
            collideConnected = False,
            localAnchorA = self.torso.get_pivot(3), localAnchorB = self.leftleg.get_pivot(0),
            enableMotor = True,
            motorSpeed = 0.0,
            maxMotorTorque = 2.0
        )
        self.world.CreateRevoluteJoint(
            bodyA = self.torso.body,
            bodyB = self.rightleg.body,
            collideConnected = False,
            localAnchorA = self.torso.get_pivot(4), localAnchorB = self.rightleg.get_pivot(0),
            enableMotor = True,
            motorSpeed = 0.0,
            maxMotorTorque = 2.0
        )
        self.world.CreateRevoluteJoint(
            bodyA = self.leftleg.body,
            bodyB = self.leftfoot.body,
            collideConnected = False,
            localAnchorA = self.leftleg.get_pivot(1), localAnchorB = self.leftfoot.get_pivot(0),
            enableMotor = True,
            motorSpeed = 0.0,
            maxMotorTorque = 0.8
        )
        self.world.CreateRevoluteJoint(
            bodyA = self.rightleg.body,
            bodyB = self.rightfoot.body,
            collideConnected = False,
            localAnchorA = self.rightleg.get_pivot(1), localAnchorB = self.rightfoot.get_pivot(0),
            enableMotor = True,
            motorSpeed = 0.0,
            maxMotorTorque = 0.8
        )
        
    
    def update(self, mousepos):
        if random() < 0.02:
            self.head.image = choice(self.heads)
        mouseWorld = px_to_world(mousepos)
        deltaPos = (mouseWorld - self.head.body.worldCenter) * 10
        self.head.body.linearVelocity.Set(*deltaPos)
        for l in self.limbs:
            l.update()
    
    def draw(self, screen):
        self.leftleg.draw(screen)
        self.rightleg.draw(screen)
        self.head.draw(screen)
        self.torso.draw(screen)
        self.leftarm.draw(screen)
        self.rightarm.draw(screen)
        self.leftfoot.draw(screen)
        self.rightfoot.draw(screen)
    
    def draw_simple(self, screen):
        pass
