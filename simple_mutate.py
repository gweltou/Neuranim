#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import os.path
import datetime
import random
import numpy as np
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_UP, K_DOWN, K_k, K_m, K_d)
from nn import *
import creatures
from camera import Camera
from stats import Stats


# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, edgeShape, staticBody, dynamicBody, pi, vec2, queryCallback, AABB)
from parameters import *


TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 480


batch_history = ""



def save_generation(filename, population, history="", stats="", generation=0):
    with open(filename, 'w') as f:
        lines = []
        for (score, c) in population:
            lines.append('####\n')
            lines.append('score: {}\n'.format(score))
            for weight in c.nn.weights:
                lines.append(str(weight.tolist()))
                lines.append('\n')
            lines.append('\n')
        header = []
        header.append('type: {}\n'.format(population[0][1].id))
        header.append('layers: {}\n'.format(population[0][1].nn.get_layers()))
        header.append('neurons: {}\n'.format(population[0][1].nn.get_total_neurons()))
        header.append('synapses: {}\n'.format(population[0][1].nn.get_total_synapses()))
        header.append('activation: {}\n'.format(population[0][1].nn.activation))
        header.append('generation: {}\n'.format(generation))
        header.append('history: {}\n'.format(history))
        header.append('stats: {}\n'.format(stats))
        header.append('\n\n')
        f.writelines(header + lines)



def import_generation(filename, world):
    population = []
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
                    population.append(c)
                if creature_type in dir(creatures):
                    creature_class = getattr(creatures, creature_type)
                    c = creature_class(world, hidden=[], activation=activation)
                    c.nn.weights = []   # Clear neural network
                else:
                    print("Error: bad type in creature definition")
                    sys.exit(1)
            elif l.startswith('[['):
                # Weight array
                c.nn.weights.append(np.array(eval(l)))
        population.append(c)
    data['population'] = population
    data['layers'] = layers
    return data



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
        creature_class = getattr(creatures, ANIMATRONIC)
        self.pool = [creature_class(self.world, HIDDEN_LAYERS, ACTIVATION) for i in range(n)]
        self.generation = 0
        print("Starting from generation 0   ({})".format(ANIMATRONIC))
    
    
    def newGeneration(self, winners):
        #print("{} drones selected with scores {}".format(len(winners),
        #                                                 zip(*winners)[0]))
        # Add previous generation winners to new pool
        new_pool = [w[1] for w in winners]
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
    
    
    def saveCreatures(self, population):
        directory = "run"
        directory = os.path.join(directory, population[0][1].id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = "gen{}.txt".format(self.generation)
        save_generation(os.path.join(directory, filename), population, batch_history,
                       self.stats.var_dict, self.generation)

    
    def importCreatures(self, filename):
        data = import_generation(filename, self.world)
        batch_history = data["history"]
        self.pool = data["population"]
        self.generation = data["generation"]
        self.stats.reset()
        if "stats" in data:
            self.stats.var_dict = data["stats"]
        print("Starting from generation {}".format(self.generation))
        print("{} drones imported".format(len(self.pool)))
        print("layers: {}".format(self.pool[0].nn.get_layers()))
        print("activation: {}".format(self.pool[0].nn.activation))
    
    
    def mainLoop(self):
        target = vec2(TARGET) #vec2(random.choice(TARGETS))
        podium = []
        creature = self.pool.pop()
        creature.set_start_position(STARTPOS[0], STARTPOS[1])
        creature.set_target(target.x, target.y)
        creature.init_body()
        score_min = 100
        steps = 0
        display_nn = False
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
                        elif event.k == K_d:
                            display_nn = not display_nn
                            print("display_nn", display_nn)
                
                self.screen.fill((0, 0, 0, 0))
                
                # Set camera center on current creature
                # A little bit above the subject
                #self.camera.set_target(creature.body.position+vec2(0.0,0.1)) # Ground texturing not working
                self.camera.render()
                pygame.display.flip()
                self.clock.tick(TARGET_FPS)
            
            
            self.world.Step(TIME_STEP, 6, 2)
            steps += 1
            if steps >= MAX_STEPS or not creature.body.awake or creature.body.position.y < 0:
                # End of trial for this creature
                steps = 0
                score = (creature.target - creature.body.position).length
                if SCORE_MIN:
                    score = score_min
                #print(len(podium), score)
                podium.append((score, creature,))
                creature.destroy()
                
                if len(self.pool) > 0:
                    # Evaluate next creature in pool
                    creature = self.pool.pop()
                    # choose a new target
                    target = vec2(TARGET) ###vec2(random.choice(TARGETS))
                    creature.set_target(target.x, target.y)
                    creature.set_start_position(STARTPOS[0], STARTPOS[1])
                    creature.init_body()
                    score_min = 100
                else:
                    # Pool is empty
                    if not BREED:
                        running = False
                        break
                    
                    gen_score = sum([l[0] for l in podium]) / len(podium)
                    podium.sort(key=lambda x: x[0])
                    winners = podium[:WINNERS_PER_GENERATION]
                    podium.clear()
                    self.stats.feed(self.generation, gen_score, winners[0][0], winners[-1][0])
                    # Save winners to file every 10 generations
                    if self.generation == 1 or (self.generation>1 and self.generation%10==0):
                        self.saveCreatures(winners)
                        if PLOT_EVOLUTION:
                            c = winners[0][1]
                            filename = "gen{}.png".format(self.generation)
                            filename = os.path.join("run", c.id, filename)
                            title = "{} {}".format(c.id, str(c.nn.get_layers()))
                            self.stats.savePlot(filename, title)
                    
                    print("end of generation {}".format(self.generation))
                    print("generation score: {}".format(gen_score))
                    self.newGeneration(winners)
                    
                    creature = self.pool.pop()
                    target = vec2(TARGET)# vec2(random.choice(TARGETS))
                    creature.set_target(target.x, target.y)
                    creature.set_start_position(STARTPOS[0], STARTPOS[1])
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

