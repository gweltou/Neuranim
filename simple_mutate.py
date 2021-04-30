#!/usr/bin/env python
# -*- coding: utf-8 -*-


from heapq import heappush, heappop
import datetime
import pygame
import random
import numpy as np
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_UP, K_DOWN, K_k, K_m)
from nn import *
from creatures import *
from camera import Camera
import sys

# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, edgeShape, staticBody, dynamicBody, pi, vec2, queryCallback, AABB)
from parameters import *


TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 480


batch_history = ""



class Stats:
    def __init__(self):
        self.var_dict = {}
    
    def feed(self, *values):
        if len(self.var_dict) == 0:
            for var, v in enumerate(values):
                self.var_dict[var] = [v]
        else:
            for var, v in enumerate(values):
                self.var_dict[var].append(v)
        #print(self.var_dict)
        
    def plot(self, filename, title=''):
        plt.close()
        plt.plot(self.var_dict[0], self.var_dict[1], label='gen score')
        plt.plot(self.var_dict[0], self.var_dict[2], label='best')
        plt.plot(self.var_dict[0], self.var_dict[3], label='worst')
        plt.legend()
        plt.show()

    def savePlot(self, filename, title=''):
        plt.close()
        plt.plot(self.var_dict[0], self.var_dict[1], label='gen score')
        plt.plot(self.var_dict[0], self.var_dict[2], label='best')
        plt.plot(self.var_dict[0], self.var_dict[3], label='worst')
        plt.legend()
        plt.title(title)
        plt.savefig(filename)

    def reset(self):
        self.var_dict = {}



class Evolve:
    def __init__(self):
        self.world = world(contactListener=nnContactListener(),
                           gravity=(0, -10),
                           doSleep=True)
        self.time_init = datetime.time()
        self.pool = []
        self.stats = Stats()
        
        # A static body to hold the ground shape
        ground = self.world.CreateStaticBody()
        ground_fix = ground.CreateEdgeFixture(vertices=[(-50,0), (50,0)],
                                              friction=1.0,
                                              userData='ground')
        if DISPLAY:
            # --- pygame setup ---
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT),
                                                  0, 32)
            pygame.display.set_caption('Neuranim Mutate')
            self.clock = pygame.time.Clock()
            self.camera = Camera(self.world,
                                 self.screen,
                                 18.0,
                                 18.0*SCREEN_HEIGHT/SCREEN_WIDTH)
    
    
    def populate(self, n):
        print("Starting from generation 0")
        self.generation = 0
        self.pool = [Boulotron2000(self.world, position=STARTPOS) for i in range(n)]
        #self.pool = [Animatronic(self.world, position=STARTPOS) for i in range(n)]
    
    
    def newGeneration(self, winners):
        #print("{} drones selected with scores {}".format(len(winners),
        #                                                 zip(*winners)[0]))
        # Add previous generation winners to new pool
        new_pool = list(list(zip(*winners))[1])
        # Add winners offspring to new pool
        offspring = []
        # Make 10 copies of every winner
        for i in range(10):
            offspring.extend([d.copy() for d in new_pool])
        # Mutate the copies
        for d in offspring:
            d.mutate()
        new_pool += offspring
        
        self.generation += 1
        self.pool = new_pool
        print("New pool of {} drones".format(len(self.pool)))
    
    
    def saveCreatures(self, creatures):
        filename = "gen{}.txt".format(self.generation)
        save_creatures(filename, creatures, batch_history,
                       self.stats.var_dict, self.generation)

    
    def importCreatures(self, filename):
        data = import_creatures(filename, self.world, vec2(STARTPOS))
        batch_history = data["history"]
        self.pool = data["creatures"]
        self.generation = data["generation"]
        self.stats.reset()
        if "stats" in data:
            self.stats.var_dict = data["stats"]
        print("Starting from generation {}".format(self.generation))
        print("{} drones imported".format(len(self.pool)))
        print("layers: {}".format(self.pool[0].nn.layers))
        print("activation: {}".format(self.pool[0].nn.activation))
    
    
    def mainLoop(self):
        target = TARGET #vec2(random.choice(TARGETS))
        podium = []
        creature = self.pool.pop()
        creature.set_target(target)
        creature.init_body()
        score_min = 100
        steps = 0
        running = True
        mirror = False
        while running:
            creature.update(self.world.contactListener.sensors[creature.id], mirror)
            if SCORE_MIN:
                score_min = min(score_min, (creature.target - creature.body.position).length)
            
            if DISPLAY:
                for event in pygame.event.get():
                    if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                        running = False
                    if event.type == KEYDOWN:
                        if event.key == K_k:
                            steps = MAX_STEPS - 10
                        elif event.key == K_m:
                            mirror = not mirror
                            if mirror: print('mirror')
                            if not mirror: print('not mirror')
                
                self.screen.fill((0, 0, 0, 0))
                
                # Set camera center on current creature
                # A little bit above the subject
                #self.camera.set_target(creature.body.position+vec2(0.0,0.1)) # Ground texturing not working
                self.camera.render()
                pygame.display.flip()
                self.clock.tick(TARGET_FPS)
            
            
            self.world.Step(TIME_STEP, 6, 2)
            steps += 1
            if steps >= MAX_STEPS or not creature.body.awake or creature.body.position.y<0:
                steps = 0
                score = (creature.target - creature.body.position).length
                if SCORE_MIN:
                    score = score_min
                print(len(podium), score)
                heappush(podium, (score, creature))
                creature.destroy_body()
                
                if len(self.pool) > 0:
                    # Evaluate next creature in pool
                    creature = self.pool.pop()
                    creature.init_body()
                    score_min = 100
                    # choose a new target
                    target = TARGET ###vec2(random.choice(TARGETS))
                    creature.set_target(target)
                else:
                    # Pool is empty
                    if not BREED:
                        running = False
                        break
                    #self.generation += 1
                    gen_score = sum(zip(*podium)[0])/len(podium)
                    winners = [heappop(podium) for i in range(WINNERS_PER_GENERATION)]
                    podium = []
                    self.stats.feed(self.generation, gen_score, winners[0][0], winners[-1][0])
                    # Save winners to file every 10 generations
                    if self.generation == 1 or self.generation%10 == 0:
                        self.saveCreatures(winners)
                    
                    print("end of generation {}".format(self.generation))
                    print("generation score: {}".format(gen_score))
                    self.newGeneration(winners)
                    
                    creature = self.pool.pop()
                    target = TARGET # vec2(random.choice(TARGETS))
                    creature.set_target(target)
                    creature.init_body()
                    score_min = 100
                    
                if self.generation > END_GEN:
                    running = False



if __name__ == "__main__":
    evolve = Evolve()
    if len(sys.argv) == 2:
        evolve.importCreatures(sys.argv[1])
        evolve.newGeneration([(d.score, d) for d in evolve.pool])
    else:
        evolve.populate(START_POP)
    evolve.mainLoop()
    pygame.quit()
    print('Done!')

