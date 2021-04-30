#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import pygame
import sys
import numpy as np
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_UP, K_DOWN, K_RIGHT, K_LEFT, K_r)
from nn import *

import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, pi, vec2)
from creatures import *
from parameters import *


PPM = 50.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480


def world_to_px(pos):
    return int(pos.x*PPM), int(SCREEN_HEIGHT - pos.y*PPM)


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
        self.clock = pygame.time.Clock()
        self.running = True
        self.world = world(contactListener=nnContactListener(), gravity=(0, -10), doSleep=True)
        self.ground = self.world.CreateStaticBody(shapes=polygonShape(box=(20, 0.1)), position=(15, 0.4))
        self.ground.fixtures[0].userData = "ground"
        self.ground.fixtures[0].friction = 1.0
    
    def draw_creature(self):
        for body in (self.creature.body, self.creature.lleg, self.creature.lfoot, 
                     self.creature.rleg, self.creature.rfoot):
            for fixture in body.fixtures:
                shape = fixture.shape
                if shape.type == 2:  # Polygon shape
                    vertices = [(body.transform * v) * PPM for v in shape.vertices]
                    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
                    pygame.draw.polygon(self.screen, (200, 200, 200, 255), vertices)
                if shape.type == 0: # Circle shape
                    color = (255, 255, 255, 255)
                    if fixture.userData in ("lsensor", "rsensor") and \
                        self.world.contactListener.sensors[fixture.userData] == True:
                        color = (0, 255, 0, 255)
                    pygame.draw.circle(self.screen, color,
                                       world_to_px(body.transform *shape.pos),
                                       int(shape.radius*PPM))
    
    def import_creature(self, filename):
        data = import_creatures(filename, self.world, vec2(5,3))
        self.creature = data["creatures"][0]
        self.creature.init_body()
        self.creature.set_target(vec2(0,0))
    
    
    def mainloop(self):
        while self.running:            
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    self.running = False
            keys = pygame.key.get_pressed()
            if keys[K_LEFT]:
                self.creature.set_target(vec2(-1, 2))
            if keys[K_RIGHT]:
                self.creature.set_target(vec2(1, 2))
            
            self.screen.fill((0, 0, 0, 0))
            
            for body in (self.ground,):  # or: world.bodies
                for fixture in body.fixtures:
                    shape = fixture.shape
                    vertices = [(body.transform * v) * PPM for v in shape.vertices]
                    vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
                    pygame.draw.polygon(self.screen, (255, 255, 255, 255), vertices)
            
            lsensor = self.world.contactListener.sensors["lsensor"]
            rsensor = self.world.contactListener.sensors["rsensor"]
            self.creature.update(lsensor, rsensor)
            self.draw_creature()
            
            pygame.display.flip()
            self.clock.tick(TARGET_FPS)
            self.world.Step(TIME_STEP, 6, 2)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        game = Game()
        game.import_creature(sys.argv[1])
        game.mainloop()
    pygame.quit()
    print('Done!')
