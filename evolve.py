#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os.path
import argparse
import datetime
import random
import re
import numpy as np
import pygame
from pygame.locals import *
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
        
        ### Box2D ###
        self.build_ground()
        
        self.target = vec2(TARGET) #vec2(random.choice(TARGETS))
        self.display_nn = False
        self.speed_multiplier = 1.0
        
        if self.display_mode:
            # --- pygame setup ---
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Neuranim Evolve')
            self.clock = pygame.time.Clock()
            self.camera = Camera(self.world,
                                 self.screen,
                                 18.0,
                                 18.0*SCREEN_HEIGHT/SCREEN_WIDTH)

    
    def build_ground(self):
        # A static body to hold the ground shape
        elevation = 0
        if hasattr(self, 'ground'):
            self.world.DestroyBody(self.ground)
        self.ground = self.world.CreateStaticBody()
        start_posx = round(STARTPOS[0])
        assert -50 < start_posx < 50, "Starting position should be between -50 and 50"
        for x in range(-50, 50):
            prev_elevation = elevation
            elevation = prev_elevation + (random.random()-0.5) * self.args.terrain_roughness*0.01
            self.ground.CreateEdgeFixture(vertices=[(x,prev_elevation), (x+1,elevation)],
                                              friction=1.0,
                                              userData='ground')
            if x == start_posx:
                self.startpos_elevation = prev_elevation
    
    
    def populate(self):
        """
            Create generation 0
        """
        creature_class = getattr(creatures, ANIMATRONIC)
        for i in range(args.pool_size):
            c = creature_class(self.world)
            c.pop_id = FancyWords.generate_two()
            nn = NeuralNetwork()
            layers = [c.n_inputs] + HIDDEN_LAYERS + [c.n_contact_sensors]
            nn.init_weights(layers) # Set random weights
            nn.set_activation(ACTIVATION)
            c.nn = nn   # Link neural netword to body
            self.pool.append(c)
        
        self.generation = 0
        print("Layers {}".format(self.pool[0].nn.get_layers()))
        print("Starting from generation 0   ({})".format(ANIMATRONIC))


    def next_generation(self, winners):
        """
            Fill the pool of creatures with the next generation
        """
        #TODO: could be simplified
        
        # Add previous generation winners to new pool
        parents = [w[1] for w in winners]
        # Make copies of every winner
        offspring = []
        num_copies = round((args.pool_size/len(winners)) - 1)
        for i in range(num_copies):
            offspring.extend([d.copy() for d in parents])
        # Mutate the copies
        mutation_count = 0
        for c in offspring:
            mutation_count += c.mutate(self.args.mutate)
        
        self.pool = offspring + parents
        self.generation += 1
        print(f"# New pool of {len(self.pool)} drones")
        print(f"    Total number of mutations: {mutation_count}")


    def save_population(self, population):
        c = population[0][1]
        directory = self.get_path(c)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(directory, "gen{}.txt".format(self.generation))
        print(f"Saving to disk... ({filename})")
        save_generation(filename, population,
                       self.stats.var_dict, self.generation)


    def load_population(self, filename):
        if os.path.isdir(filename):
            p = re.compile(r'gen(\d+).txt' ,re.IGNORECASE)
            highest_gen = 0
            for f in os.listdir(filename):
                m = p.match(f)
                if m and int(m[1]) > highest_gen:
                    highest_gen = int(m[1])
            filename = os.path.join(filename, f'gen{highest_gen}.txt')
        
        data = import_generation(filename, self.world)
        self.pool = data["population"]
        self.generation = data["generation"]
        self.args.end_generation += self.generation
        self.stats.reset()
        if "stats" in data:
            self.stats.var_dict = data["stats"]
        print('Population "{}"'.format(self.pool[0].pop_id))
        print("Layers (input+hidden+output): {}".format(self.pool[0].nn.get_layers()))
        print("Activation: {}".format(self.pool[0].nn.activation))
        print("{} drones imported".format(len(self.pool)))
        print("Starting from generation {}".format(self.generation))
    
    
    def get_path(self, c):
        return os.path.join("run",
            c.morpho.lower()+'_'+str(c.nn.get_total_neurons()), c.pop_id.lower())
    
    
    def pop_creature(self):
        creature = self.pool.pop()
        creature.set_start_position(STARTPOS[0], STARTPOS[1] + self.startpos_elevation)
        # Choose a new target
        self.target = vec2(TARGET)  # vec2(random.choice(TARGETS))
        creature.set_target(self.target.x, self.target.y)
        creature.init_body()
        #self.score_min = 100
        if self.display_mode:
            self.nn_coords = build_nn_coords(creature.nn)
            creature.nn.save_state = True   # Used when displaying nn structure
        return creature
        
        
    def mainLoop(self):
        podium = []
        creature = self.pop_creature()
        steps = 0
        mirror = False
        mouse_drag = False
        selected_neuron = None
        show_downstream = False
        paused = False
        running = True
        score = 0
        while running:
        
            #### PyGame ####
            if self.args.view:
                # Process keyboard events
                # 'q' or 'ESC'  Quit
                # 'n'   next creature
                # 'm'   mirror mode
                # 'd'   show neural network
                # 'f'   center on creature and follow
                # 's'   slow motion
                # 'p'   pause
                # 'w'   toggle downstream/upstream pathway
                for event in pygame.event.get():
                    if event.type == MOUSEBUTTONDOWN:
                        mouse_drag = True
                        pygame.mouse.get_rel()
                    elif event.type == MOUSEBUTTONUP:
                        mouse_drag = False
                    elif event.type == MOUSEWHEEL:
                        self.camera.zoom(event.y)
                    elif event.type == MOUSEMOTION:
                        if self.display_nn:
                            mouse_x, mouse_y = pygame.mouse.get_pos()
                            selected_neuron = None
                            for i, l in enumerate(self.nn_coords):
                                if abs(l[0][0]-mouse_x) < 6:
                                    for j, n in enumerate(l):
                                        if abs(n[1]-mouse_y) < 6:
                                            selected_neuron = (i, j)
                                            break
                                    break
                        if mouse_drag:
                            mouse_dx, mouse_dy = pygame.mouse.get_rel()
                            self.camera.move(-mouse_dx*0.006, mouse_dy*0.006)
                    elif event.type == KEYDOWN:
                        if event.key == K_ESCAPE or event.key == K_q:
                            running = False
                        elif event.key == K_n:    # Next
                            steps = self.args.limit_steps - 10
                        elif event.key == K_m:  # Mirror
                            mirror = not mirror
                            if mirror: print('mirror')
                            if not mirror: print('not mirror')
                        elif event.key == K_d:  # Display Neural Network
                            self.display_nn = not self.display_nn
                        elif event.key == K_f:
                            self.camera.follow(creature)
                        elif event.key == K_s:  # Slow motion
                            if self.speed_multiplier == 1.0:
                                self.speed_multiplier = 0.1
                            else:
                                self.speed_multiplier = 1.0
                        elif event.key == K_p:  # Pause
                            paused = not paused
                        elif event.key == K_w:  # upstream/downstream pathway
                            show_downstream = not show_downstream
                    elif event.type == QUIT:
                        running = False
                            
            if not paused:
                creature.update(self.world.contactListener.sensors[creature.id][:-1], mirror)
            
            #### PyGame ####
            if self.display_mode:
                self.camera.render()

                # Display neural network
                if self.display_nn:
                    white = (255, 255, 255)
                    # Draw synapses
                    for i in range(len(self.nn_coords)-1):
                        for j in range(len(self.nn_coords[i])):
                            p1 = self.nn_coords[i][j]
                            for w in range(len(self.nn_coords[i+1])):
                                p2 = self.nn_coords[i+1][w]
                                neuron_value = creature.nn.state[i][j]
                                weight_value = creature.nn.weights[i][j][w]
                                if weight_value * neuron_value:
                                    pygame.draw.line(self.screen, white, p1, p2, 1)
                    
                    # Draw synapses of selected neuron
                    if selected_neuron:
                        i, j = selected_neuron
                        p1 = self.nn_coords[i][j]
                        if show_downstream and i < len(self.nn_coords)-1 or i == 0:
                            for n in range(len(self.nn_coords[i+1])):
                                p2 = self.nn_coords[i+1][n]
                                neuron_value = creature.nn.state[i][j]
                                weight_value = creature.nn.weights[i][j][n]
                                intensity = weight_value * neuron_value
                                if weight_value:
                                    width = max(1, int(round(abs(weight_value)*8)))
                                    green = round(max(0, intensity) * 255)
                                    red = round(abs(min(0, intensity)) * 255)
                                    color = (red, green, 0)
                                    pygame.draw.line(self.screen, color, p1, p2, width)
                        elif not show_downstream and i > 0 or i == len(self.nn_coords)-1:
                            for n in range(len(self.nn_coords[i-1])):
                                p2 = self.nn_coords[i-1][n]
                                neuron_value = creature.nn.state[i-1][n]
                                weight_value = creature.nn.weights[i-1][n][j]
                                intensity = weight_value * neuron_value
                                if weight_value:
                                    width = max(1, int(round(abs(weight_value)*8)))
                                    green = round(max(0, intensity) * 255)
                                    red = round(abs(min(0, intensity)) * 255)
                                    color = (red, green, 0)
                                    pygame.draw.line(self.screen, color, p1, p2, width)
                    
                    # Draw neurons
                    for layer_num in range(len(self.nn_coords)):
                        for neuron_num in range(len(self.nn_coords[layer_num])):
                            x, y = self.nn_coords[layer_num][neuron_num]
                            neuron_value = creature.nn.state[layer_num][neuron_num]
                            green = round(max(0, neuron_value) * 255)
                            red = round(abs(min(0, neuron_value)) * 255)
                            color = (red, green, 0)
                            if selected_neuron == (layer_num, neuron_num):
                                pygame.draw.circle(self.screen, color, (x, y), 8)
                            else:
                                pygame.draw.circle(self.screen, color, (x, y), 6)

                pygame.display.flip()
                self.clock.tick(TARGET_FPS)
            
            if paused:
                continue
            
            self.world.Step(TIME_STEP*self.speed_multiplier, 6, 2)
            
            if SLOUCHING_PENALTY != 0:
                if self.world.contactListener.sensors[creature.id][-1]:
                    score += SLOUCHING_PENALTY
            
            steps += 1 * self.speed_multiplier
            if steps >= self.args.limit_steps or not creature.body.awake:
                # End of trial for this creature
                steps = 0
                score += (creature.target - creature.body.position).length
                podium.append((score, creature,))
                creature.destroy()

                if len(self.pool) > 0:
                    # Evaluate next creature in pool
                    creature = self.pop_creature()
                    score = 0
                else:
                    # Pool is empty
                    if not BREED:
                        running = False
                        break
                    
                    gen_score = sum([l[0] for l in podium]) / len(podium)
                    podium.sort(key=lambda x: x[0])
                    winners_number = int(len(podium) * args.winners_percent/100)
                    winners = podium[:winners_number]
                    podium.clear()
                    self.stats.feed(self.generation, gen_score, winners[0][0], winners[-1][0])
                    # Save winners to file every X generations
                    if self.generation > 1 and self.generation%self.args.save_interval == 0:
                        self.save_population(winners)
                        if PLOT_EVOLUTION:
                            c = winners[0][1]
                            filename = "gen{}.png".format(self.generation)
                            filename = os.path.join(self.get_path(c), filename)
                            title = "{} {}".format(c.pop_id, str(c.nn.get_layers()))
                            self.stats.savePlot(filename, title)
                    print(f"    Generation score: {gen_score}")
                    print(f"    {len(winners)} creatures selected")
                    print(f"# End of generation {self.generation}\n")
                    
                    if self.generation >= args.end_generation:
                        running = False
                    else:
                        self.build_ground() # Change ground topology
                        self.next_generation(winners)
                        creature = self.pop_creature()




def parseInputs():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-v', '--view', action='store_true', help='enable presentation mode')
    parser.add_argument('-m', '--mutate', type=int, default=2,
                        help='mutation frequency multiplier (defaults to 2)')
    parser.add_argument('-f', '--file', type=str, help='population file')
    parser.add_argument('-t', '--terrain_roughness', type=int, default=30, help='terrain variation in elevation (in percent)')
    parser.add_argument('-s', '--save_interval', type=int, default=10, help='save population to disk every X generations')
    parser.add_argument('-l', '--limit_steps', type=int, default=500, help='max number of steps for each individual trial (defaults to 500)')
    parser.add_argument('-p', '--pool_size', type=int, default=200, help='size of creature population (defaults to 200)')
    parser.add_argument('-w', '--winners_percent', type=int, default=10, help='percent of selected individuals per generation')
    parser.add_argument('-e', '--end_generation', type=int, default=500, help='limit simulation to this number of generations (defaults to 500)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parseInputs()
    #print(args)
    args.mutate = max(1, args.mutate)
    args.terrain_roughness = max(0, args.terrain_roughness)
    args.save_interval = max(1, args.save_interval)
    args.limit_steps = max(50, args.limit_steps)
    args.winners_percent = min(100, max(1, args.winners_percent))
    
    evolve = Evolve(args)
    print("Parameters :")
    for k,v in args.__dict__.items():
        print(f"  {k}: {v}")
    
    if args.file:
        evolve.load_population(args.file)
        evolve.next_generation([(d.score, d) for d in evolve.pool])  #TODO: could be simplified
        if args.view:
            pygame.display.set_caption('Neuranim Evolve  --  ' + 
                    evolve.pool[0].pop_id +
                    f' [{args.file.split(os.path.sep)[-1]}]')
    else:
        evolve.populate()
    
    evolve.mainLoop()
    
    pygame.quit()
    print('Done!')
