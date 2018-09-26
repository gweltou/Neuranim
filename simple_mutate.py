#!/usr/bin/env python
# -*- coding: utf-8 -*-


from heapq import heappush, heappop
import datetime
import pygame
import random
import numpy as np
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_k, K_m)
from nn import *

import Box2D  # The main library
# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, pi, vec2)
from parameters import *


PPM = 20.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480


def world_to_px(pos):
    return int(pos.x*PPM), int(SCREEN_HEIGHT - pos.y*PPM)


# --- pygame setup ---
if DISPLAY:
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
    pygame.display.set_caption('Neuranim Mutate')
    clock = pygame.time.Clock()
world = world(contactListener=nnContactListener(), gravity=(0, -10), doSleep=True)


# A static body to hold the ground shape
ground_body = world.CreateStaticBody(
    shapes=polygonShape(box=(20, 0.1)),
    position=(15, 0.4),
)
ground_body.fixtures[0].userData = "ground"
ground_body.fixtures[0].friction = 1.0



def draw_creature(screen, creature):
    for body in (creature.body, creature.lleg, creature.lfoot, 
                 creature.rleg, creature.rfoot):
        for fixture in body.fixtures:
            shape = fixture.shape
            if shape.type == 2:  # Polygon shape
                vertices = [(body.transform * v) * PPM for v in shape.vertices]
                vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
                pygame.draw.polygon(screen, (200, 200, 200, 255), vertices)
            if shape.type == 0: # Circle shape
                color = (255, 255, 255, 255)
                if fixture.userData in ("lsensor", "rsensor") and \
                    world.contactListener.sensors[fixture.userData] == True:
                    color = (0, 255, 0, 255)
                pygame.draw.circle(screen, color,
                                   world_to_px(body.transform *shape.pos),
                                   int(shape.radius*PPM))


# --- main game loop ---
time0 = datetime.time()
running = True
generation = START_GEN
target = vec2(random.choice(TARGETS))
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
    pool = [Animatronic(world, position=STARTPOS) for i in range(STARTING_POPULATION)]

podium = []
creature = pool.pop()
creature.set_target(target)
creature.init_body()
world.contactListener.registerSensors(creature.sensors)
score_min = 100

while running:
    #lsensor = world.contactListener.sensors["lsensor"]
    #rsensor = world.contactListener.sensors["rsensor"]
    creature.update(world.contactListener.sensors, mirror)
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
        
        for body in (ground_body,):  # or: world.bodies
            for fixture in body.fixtures:
                shape = fixture.shape
                vertices = [(body.transform * v) * PPM for v in shape.vertices]
                vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
    
                pygame.draw.polygon(screen, (255, 255, 255, 255), vertices)
    
        pygame.draw.circle(screen, (255,0,0,255), world_to_px(target), 5)
        draw_creature(screen, creature)
        pygame.display.flip()
        clock.tick(TARGET_FPS)
    
    world.Step(TIME_STEP, 6, 2)
    steps += 1
    if steps >= MAX_STEPS or not creature.body.awake or creature.body.position.y<1:
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
            target = vec2(random.choice(TARGETS))
            creature.set_target(target)
            creature.init_body()
            score_min = 100
            
        if generation > END_GEN:
            running = False


pygame.quit()
print('Done!')
