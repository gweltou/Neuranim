#!/usr/bin/env python
# -*- coding: utf-8 -*-

from heapq import *
import datetime
import pygame
import random
import numpy as np
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_UP, K_DOWN, K_RIGHT, K_LEFT, K_r)
from nn import *
from boulotron import Boulotron2000
from camera import Camera

import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, pi, vec2)
from parameters import *


PPM = 50.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480


screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
pygame.display.set_caption('Simple pygame example')
clock = pygame.time.Clock()
world = world(contactListener=nnContactListener(), gravity=(0, 0), doSleep=True)
camera = Camera(world, screen, 10.0, 10.0*SCREEN_HEIGHT/SCREEN_WIDTH)


# And a static body to hold the ground shape
ground_body = world.CreateStaticBody(
    shapes=polygonShape(box=(20, 0.1)),
    position=(15, 0.4),
)
ground_body.fixtures[0].userData = "ground"
ground_body.fixtures[0].friction = 1.0

# --- main game loop ---
DISPLAY = True
running = True
creature = Boulotron2000(world, position=(5, 5))
creature.init_body()


while running:    
    if DISPLAY:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
            if event.type == KEYDOWN:
                if event.key == K_r:
                    for body in creature.bodies:
                        body.angle = 0
                    for j in creature.joints:
                        j.motorSpeed = 0.0
                elif event.key == K_UP:
                    for j in creature.joints:
                        j.motorSpeed += 1
                elif event.key == K_DOWN:
                    for j in creature.joints:
                        j.motorSpeed -= 1
                elif event.key == K_LEFT:
                    creature.lleg.angle += 0.5
                    creature.rleg.angle -= 0.5
                    creature.lfoot.angle += 0.5
                    creature.rfoot.angle -= 0.5
                elif event.key == K_RIGHT:
                    creature.lleg.angle -= 0.5
                    creature.rleg.angle += 0.5
                    creature.lfoot.angle -= 0.5
                    creature.rfoot.angle += 0.5
                    
        
        camera.set_target(creature.position)
        camera.render()

        pygame.display.flip()
        clock.tick(TARGET_FPS)
    
    world.Step(TIME_STEP, 6, 2)
    joints = np.array([j.angle for j in creature.joints]) * np.array([1, 1, 1, -1, -1, -1])
    
    if pygame.time.get_ticks() % 100 == 0:
        for j in joints:
            print((j%(2*pi))/pi -1),
        print('')


pygame.quit()
print('Done!')
