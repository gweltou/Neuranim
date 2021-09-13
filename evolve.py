#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os.path
import argparse
import datetime
import random
import numpy as np
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_LEFT, K_RIGHT, K_UP, K_DOWN, K_k, K_m, K_d, K_q
from nn import *
import creatures
from renderer import Camera
from utils import *

# Box2D.b2 maps Box2D.b2Vec2 to vec2 (and so on)
from Box2D.b2 import (world, polygonShape, edgeShape, staticBody, dynamicBody, pi, vec2, queryCallback, AABB)
from parameters import *


TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600
BREED = True



def build_nn_coords(nn):
    coords = []
    layers = nn.get_layers()
    breadth = max(layers)
    depth = len(layers)
    x_step = SCREEN_WIDTH / depth
    y_step = SCREEN_HEIGHT / (breadth+1)
    x = x_step / 2
    for l in layers:
        l_coords = []
        y = y_step * (breadth - l) / 2
        for n in range(l):
            y += y_step
            l_coords.append((x, y,))
        coords.append(l_coords)
        x += x_step
    return coords




class Evolve:
    def __init__(self, args):
        self.args = args
        self.display_mode = self.args.view
        
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
        if self.display_mode:
            # --- pygame setup ---
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT),
                                                  0, 32)
            pygame.display.set_caption('Neuranim Evolve')
            self.clock = pygame.time.Clock()
            self.camera = Camera(self.world,
                                 self.screen,
                                 18.0,
                                 18.0*SCREEN_HEIGHT/SCREEN_WIDTH)


    def populate(self, n):
        """
            Create generation 0
        """
        creature_class = getattr(creatures, ANIMATRONIC)
        for i in range(n):
            c = creature_class(self.world)
            c.pop_id = FancyWords.generate_two()
            nn = NeuralNetwork()
            layers = [c.n_inputs] + HIDDEN_LAYERS + [c.n_sensors]
            nn.init_weights(layers)
            nn.set_activation(ACTIVATION)
            c.nn = nn   # Link neural netword to body
            self.pool.append(c)
        
        self.generation = 0
        print("Starting from generation 0   ({})".format(ANIMATRONIC))
        print("Layers {}".format(self.pool[0].nn.get_layers()))


    def nextGeneration(self, winners):
        #TODO: could be simplified
        
        # Add previous generation winners to new pool
        new_pool = [w[1] for w in winners]
        # Add winners offspring to new pool
        offspring = []
        # Make 10 copies of every winner
        for i in range(10):
            offspring.extend([d.copy() for d in new_pool])
        # Mutate the copies
        mutation_count = 0
        for c in offspring:
            mutation_count += c.mutate(self.args.mutate)
        print(f" ## number of mutations : {mutation_count}")
        new_pool += offspring

        self.generation += 1
        self.pool = new_pool
        print("New pool of {} drones".format(len(self.pool)))


    def saveCreatures(self, population):
        c = population[0][1]
        directory = self.get_path(c)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = "gen{}.txt".format(self.generation)
        save_generation(os.path.join(directory, filename), population,
                       self.stats.var_dict, self.generation)


    def importCreatures(self, filename):
        data = import_generation(filename, self.world)
        self.pool = data["population"]
        self.generation = data["generation"]
        self.stats.reset()
        if "stats" in data:
            self.stats.var_dict = data["stats"]
        print("Starting from generation {}".format(self.generation))
        print('Population "{}"'.format(self.pool[0].pop_id))
        print("{} drones imported".format(len(self.pool)))
        print("layers (input+hidden+output): {}".format(self.pool[0].nn.get_layers()))
        print("activation: {}".format(self.pool[0].nn.activation))
    
    
    def get_path(self, c):
        return os.path.join("run",
            c.morpho.lower()+'_'+str(c.nn.get_total_neurons()), c.pop_id.lower())
    
    
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
        nn_coords = build_nn_coords(creature.nn)
        while running:
            if self.args.view:
                # Process keyboard events
                for event in pygame.event.get():
                    if event.type == QUIT:
                        running = False
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE or event.key == K_q:
                            running = False
                        if event.key == K_k:
                            steps = MAX_STEPS - 10
                        elif event.key == K_m:
                            mirror = not mirror
                            if mirror: print('mirror')
                            if not mirror: print('not mirror')
                        elif event.key == K_d:
                            display_nn = not display_nn

            if display_nn:
                creature.nn.save_state = True
            else:
                creature.nn.save_state = False
            creature.update(self.world.contactListener.sensors[creature.id], mirror)

            if SCORE_MIN:
                score_min = min(score_min, (creature.target - creature.body.position).length)

            if self.display_mode:
                self.screen.fill((0, 0, 0, 0))

                # Set camera center on current creature
                # A little bit above the subject
                #self.camera.set_target(creature.body.position+vec2(0.0,0.1)) # Ground texturing not working
                self.camera.render()

                if display_nn:
                    white = (255, 255, 255)
                    for j in range(len(nn_coords)-1):
                        for i in range(len(nn_coords[j])):
                            for w in range(len(nn_coords[j+1])):
                                p1 = nn_coords[j][i]
                                p2 = nn_coords[j+1][w]
                                neuron_value = creature.nn.state[j][i]
                                weight_value = creature.nn.weights[j][i][w]
                                if weight_value * neuron_value:
                                    pygame.draw.line(self.screen, white, p1, p2, 1)
                    for layer_num in range(len(nn_coords)):
                        for neuron_num in range(len(nn_coords[layer_num])):
                            x, y = nn_coords[layer_num][neuron_num]
                            neuron_value = creature.nn.state[layer_num][neuron_num]
                            green = round(max(0, neuron_value) * 255)
                            red = round(abs(min(0, neuron_value)) * 255)
                            color = (red, green, 0)
                            pygame.draw.circle(self.screen, color, (x, y), 8)


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
                    if self.generation > 1 and self.generation%10 == 0:
                        self.saveCreatures(winners)
                        if PLOT_EVOLUTION:
                            c = winners[0][1]
                            filename = "gen{}.png".format(self.generation)
                            filename = os.path.join(self.get_path(c), filename)
                            title = "{} {}".format(c.pop_id, str(c.nn.get_layers()))
                            self.stats.savePlot(filename, title)

                    print("end of generation {}".format(self.generation))
                    print(" ## generation score: {}".format(gen_score))
                    self.nextGeneration(winners)

                    creature = self.pool.pop()
                    target = vec2(TARGET)# vec2(random.choice(TARGETS))
                    creature.set_target(target.x, target.y)
                    creature.set_start_position(STARTPOS[0], STARTPOS[1])
                    creature.init_body()
                    score_min = 100

                if self.generation > END_GEN:
                    running = False



def parseInputs():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-v', '--view', action='store_true', help='enable presentation mode')
    parser.add_argument('-m', '--mutate', type=int, default=2, help='mutation frequency multiplier', choices=range(1,6))
    parser.add_argument('-f', '--file', type=str, help='population file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parseInputs()
    
    evolve = Evolve(args)
    
    if args.file:
        evolve.importCreatures(args.file)
        evolve.nextGeneration([(d.score, d) for d in evolve.pool])  #TODO: could be simplified
    else:
        evolve.populate(START_POP)
    
    evolve.mainLoop()
    
    pygame.quit()
    print('Done!')
