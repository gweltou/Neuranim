#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from heapq import heappush, heappop
import datetime
import pygame
import random
import numpy as np
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_UP, K_DOWN, K_k, K_m)
from nn import *
from boulotron import Boulotron2000, import_creatures
from camera import Camera

# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, edgeShape, staticBody, dynamicBody, pi, vec2, queryCallback, AABB)
from parameters import *



TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480


world = world(contactListener=nnContactListener(), gravity=(0, -10), doSleep=True)
if DISPLAY:
    # --- pygame setup ---
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    pygame.display.set_caption('Neuranim Mutate')
    clock = pygame.time.Clock()
    camera = Camera(world, screen, 18.0, 18.0*SCREEN_HEIGHT/SCREEN_WIDTH)


# A static body to hold the ground shape
ground = world.CreateStaticBody()
ground_fix = ground.CreateEdgeFixture(vertices=[(-10,0), (40,0)], friction=1.0, userData='ground')



time0 = datetime.time()
running = True
generation = START_GEN
target = TARGET #vec2(random.choice(TARGETS))
steps = 0
mirror = False
pool = []
batch_history = ""

if generation > 0:
    print("Starting at generation {}".format(generation+1))
    pool, history = import_creatures('gen{}.txt'.format(generation), world, vec2(STARTPOS))
    print("{} creatures imported".format(len(pool)))
    print("layers: {}\n".format(pool[0].nn.layers))
    if history:
        print(history)
    batch_history = history
    if BREED:
        offspring = []
        for i in range(9):
            offspring.extend([c.copy() for c in pool])
        for c in offspring:
            c.mutate()
        pool.extend(offspring)

else:
    #pool = [Animatronic(world, position=STARTPOS) for i in range(STARTING_POPULATION)]
    pool = [Boulotron2000(world, position=STARTPOS) for i in range(STARTING_POPULATION)]

podium = []
creature = pool.pop()
creature.set_target(target)
creature.init_body()
score_min = 100


# ============== Simulation Main Loop ==============
while running:
    creature.update(world.contactListener.sensors[creature.id], mirror)
    if SCORE_MIN:
        score_min = min(score_min, (creature.target - creature.body.position).length)
    
    
    if DISPLAY:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
            if event.type == KEYDOWN:
                if event.key == K_k:
                    steps = MAX_STEPS - 10
                elif event.key == K_k:
                    steps = MAX_STEPS - 10
                elif event.key == K_m:
                    mirror = not mirror
                    if mirror: print('mirror')
                    if not mirror: print('not mirror')
        
        screen.fill((0, 0, 0, 0))
        
        # Set camera center on current creature
        camera.set_target(creature.body.position)
        camera.render()
        pygame.display.flip()
        clock.tick(TARGET_FPS)
    
    
    world.Step(TIME_STEP, 6, 2)
    steps += 1
    if steps >= MAX_STEPS or not creature.body.awake or creature.body.position.y<0:
        steps = 0
        score = (creature.target - creature.body.position).length
        if SCORE_MIN:
            score = score_min
        print(len(podium), score)
        heappush(podium, (score, creature))
        creature.destroy_body()
        
        if len(pool) >= 1:
            # Evaluate next creature in pool
            creature = pool.pop()
            creature.init_body()
            score_min = 100
            # choose a new target
            target = vec2(random.choice(TARGETS))
            creature.set_target(target)
        else:
            if not BREED:
                running = False
                break
            generation += 1
            gen_score = sum(zip(*podium)[0])/len(podium)
            winners = [heappop(podium) for i in range(WINNERS_PER_GENERATION)]
            podium = []
            # Save winners to file every 10 generations
            if generation == 1 or generation%10 == 0:
                save_creatures("gen{}.txt".format(generation), winners, batch_history, generation)
            print("{} creatures selected with scores {}".format(len(winners), zip(*winners)[0]))
            # Add previous generation winners to new pool
            pool = list(zip(*winners)[1])
            # Add winners offspring to new pool
            offspring = []
            for i in range(9):
                offspring.extend([c.copy() for c in pool])
            for c in offspring:
                c.mutate()
            pool += offspring
            print("end of generation {}".format(generation))
            print("generation score: {}".format(gen_score))
            print("New pool of {} creatures".format(len(pool)))
            creature = pool.pop()
            target = TARGET # vec2(random.choice(TARGETS))
            creature.set_target(target)
            creature.init_body()
            score_min = 100
            
        if generation > END_GEN:
            running = False


pygame.quit()
print('Done!')
