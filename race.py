#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os.path
import argparse
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


TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600



class Evolve:
    def __init__(self, args):
        self.args = args
        
        self.world = world(contactListener=nnContactListener(),
                           gravity=(0, -10),
                           doSleep=True)
        self.pool = []
        self.stats = Stats()
        
        self.speed_multiplier = 1.0
        
        # --- pygame setup ---
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Neuranim Evolve')
        self.clock = pygame.time.Clock()
        self.camera = Camera(self.world,
                             self.screen,
                             18.0, 18.0*SCREEN_HEIGHT/SCREEN_WIDTH)
        
        ### Box2D ###
        self.target = vec2(TARGET) #vec2(random.choice(TARGETS))
        self.build_ground()
        
    
    def build_ground(self):
        # A static body to hold the ground shape
        elevation = 0
        if hasattr(self, 'ground'):
            self.world.DestroyBody(self.ground)
        self.ground = self.world.CreateStaticBody()
        start_posx = round(STARTPOS[0])
        assert -50 < start_posx < 50, "Starting position should be between -50 and 50"
        for x in range(-50, 100):
            prev_elevation = elevation
            elevation = prev_elevation + (random.random()-0.5) * self.args.terrain_roughness*0.01
            self.ground.CreateEdgeFixture(vertices=[(x,prev_elevation), (x+1,elevation)],
                                              friction=1.0,
                                              userData='ground')
            if x == start_posx:
                self.startpos_elevation = prev_elevation
            elif abs(self.target.x-x) < 0.5:
                self.camera.set_pole(self.target.x, prev_elevation)
        self.ground.CreateEdgeFixture(vertices=[(x+1,elevation), (x+1,50)],
                                              friction=1.0,
                                              userData='ground')


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
        
        print('Population "{}"'.format(self.pool[0].pop_id))
        print("Layers (input+hidden+output): {}".format(self.pool[0].nn.get_layers()))
        print("{} drones imported".format(len(self.pool)))
    
    
    def get_path(self, c):
        return os.path.join("run",
            c.morpho.lower()+'_'+str(c.nn.get_total_neurons()), c.pop_id.lower())
    
    
    def pop_creature(self):
        creature = self.pool.pop()
        creature.set_start_position(STARTPOS[0], STARTPOS[1] + self.startpos_elevation)
        # Choose a new target
        creature.set_target(self.target.x, self.target.y)
        r = random.randrange(160, 240)
        g = random.randrange(120, 200)
        b = random.randrange(120, 200)
        creature.color = (r, g, b)
        #creature.init_body()
        return creature
        
    
    def next_runners(self):
        for c in self.creatures:
            c.destroy()
        self.creatures = [self.pop_creature() for i in range(args.num_participants)]
    
    
    def mainLoop(self):
        podium = []
        creatures = [self.pop_creature() for i in range(args.num_participants)]
        for i, c in enumerate(creatures):
            c.set_start_position(c.start_position.x-i, c.start_position.y)
            c.init_body()
            c.set_category(i+1)
        self.camera.follow(creatures[0])
        steps = 0
        mirror = False
        mouse_drag = False
        paused = False
        running = True
        
        while running:
        
            #### PyGame ####
            # Process keyboard events
            # 'q' or 'ESC'  Quit
            # 'f'   center on creature and follow
            # 's'   slow motion
            # 'p'   pause
            for event in pygame.event.get():
                if event.type == MOUSEBUTTONDOWN:
                    mouse_drag = True
                    pygame.mouse.get_rel()
                elif event.type == MOUSEBUTTONUP:
                    mouse_drag = False
                elif event.type == MOUSEWHEEL:
                    self.camera.zoom(event.y)
                elif event.type == MOUSEMOTION:
                    if mouse_drag:
                        mouse_dx, mouse_dy = pygame.mouse.get_rel()
                        self.camera.move(-mouse_dx*0.006, mouse_dy*0.006)
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        running = False
                    elif event.key == K_f:
                        sorted_creatures = sorted(creatures,
                                                  key=lambda c : c.body.position.x)
                        self.camera.follow(sorted_creatures[-1])
                    elif event.key == K_s:  # Slow motion
                        if self.speed_multiplier == 1.0:
                            self.speed_multiplier = 0.1
                        else:
                            self.speed_multiplier = 1.0
                    elif event.key == K_p:  # Pause
                        paused = not paused
                    elif event.key == K_n:  # Next batch
                        steps = 0
                        self.camera.set_center(vec2(0, 0))
                        self.build_ground()
                        for c in creatures:
                            c.destroy()
                        creatures = [self.pop_creature() for i in range(args.num_participants)]
                        for i, c in enumerate(creatures):
                            c.set_start_position(c.start_position.x-i, c.start_position.y)
                            c.init_body()
                            c.set_category(i+1)
                        self.camera.follow(creatures[0])
                elif event.type == QUIT:
                    running = False
            
            if not paused:
                for c in creatures:
                    c.update(self.world.contactListener.sensors[c.id], mirror)
            
            #### PyGame ####
            self.camera.render()
            pygame.display.flip()
            self.clock.tick(TARGET_FPS)
            
            if paused:
                continue
            
            self.world.Step(TIME_STEP*self.speed_multiplier, 6, 2)
            steps += 1 * self.speed_multiplier
            if steps >= self.args.limit_steps:
                # End of trial for this creature
                steps = 0
                for c in creatures:
                    c.destroy()
                running = False

                if len(self.pool) > 0:
                    # Evaluate next creature in pool
                    #creature = self.pop_creature()
                    pass



def parseInputs():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m', '--mutate', type=int, default=2,
                        help='mutation frequency multiplier (defaults to 2)')
    parser.add_argument('-f', '--file', type=str, help='population file')
    parser.add_argument('-t', '--terrain_roughness', type=int, default=20, help='terrain variation in elevation (in percent)')
    parser.add_argument('-l', '--limit_steps', type=int, default=2000, help='max number of steps for each individual trial (defaults to 500)')
    parser.add_argument('-n', '--num-participants', type=int, default=4, help='')
    parser.add_argument('-p', '--pool_size', type=int, default=200, help='size of creature population (defaults to 200)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parseInputs()
    args.mutate = max(1, args.mutate)
    args.terrain_roughness = max(0, args.terrain_roughness)
    args.limit_steps = max(50, args.limit_steps)
    
    evolve = Evolve(args)
    print("Parameters :")
    for k,v in args.__dict__.items():
        print(f"  {k}: {v}")
    
    if args.file:
        evolve.load_population(args.file)
        pygame.display.set_caption('Neuranim Evolve  --  ' + 
                evolve.pool[0].pop_id +
                f' [{args.file.split(os.path.sep)[-1]}]')
    
    evolve.mainLoop()
    
    pygame.quit()
    print('Done!')
