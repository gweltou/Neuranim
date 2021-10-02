#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from math import (floor, ceil)
import pygame
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, pi, vec2, queryCallback, AABB)
from creatures import Animatronic



# ======================== CAMERA CLASS ========================
class Camera(queryCallback):

    def __init__(self, world, screen, w, h, center=(0,0)):
        self.world = world      # Box2D World
        self.width = w
        self.height = h
        self.screen = screen    # PyGame Screen
        self.screen_width = self.screen.get_width()
        self.screen_height = self.screen.get_height()
        self.HPPM = self.screen_width / self.width      # Horizontal pixels per meter
        self.VPPM = self.screen_height / self.height    # Vertical pixels per meter
        self.center = vec2(center)
        self.aabb = AABB(lowerBound=self.center - (self.width/2, self.height/2),
                         upperBound=self.center + (self.width/2, self.height/2))
        self.creatures_in_view = set()
        self.following = False
        self.draw_pole = False
        queryCallback.__init__(self)
    
    
    def ReportFixture(self, fixture):
        # Draw fixture
        shape = fixture.shape
        
        if isinstance(fixture.userData, Animatronic):
            self.creatures_in_view.add(fixture.userData)
        elif isinstance(fixture.userData, tuple):
            pass
        elif fixture.userData == 'ground' and shape.type == 1:
            #Ground line
            p0 = self.world_to_px(shape.vertices[0])
            p1 = self.world_to_px(shape.vertices[1])
            px_points = [(p0[0], p0[1]),
                         (p1[0], p1[1]),
                         (p1[0], self.screen_height),
                         (p0[0], self.screen_height)]
            pygame.draw.polygon(self.screen, (64, 64, 64, 255), px_points)
            
        else:
            if shape.type == 2:  # Polygon shape
                # Convert vertices local coord to absolute px coord
                vertices = [self.world_to_px(fixture.body.transform * v) for v in shape.vertices]
                pygame.draw.polygon(self.screen, (160, 160, 160, 255), vertices)
            
            if shape.type == 0: # Circle shape
                color = (160, 160, 160, 255)
                # TODO: replace with pygame.draw.ellipse()
                pygame.draw.circle(self.screen, color,
                                   self.world_to_px(fixture.body.transform * shape.pos),
                                   int(shape.radius * self.HPPM))
        
        # Continue the query by returning True
        return True
    
    
    def world_to_px(self, pos):
        """ Reverse height coordinates (up is positive in Box2D) """
        return (int((pos[0]-self.center.x)*self.HPPM)+self.screen_width//2,
                int(self.screen_height//2 - (pos[1]-self.center.y)*self.VPPM))
    
    
    def set_center(self, pos):
        self.center = pos
        #self.following = False
        self.updateAABB()
    
    
    def zoom(self, value):
        self.HPPM += self.HPPM * value * 0.04
        self.VPPM += self.VPPM * value * 0.04
        self.width = self.screen_width / self.HPPM
        self.height = self.screen_height / self.VPPM
        self.updateAABB()
    
    
    def draw_creature(self, creature):
        main_color = (160, 160, 160)
        if hasattr(creature, 'color'):
            main_color = creature.color
        for b in creature.bodies:
            for f in b.fixtures:
                color = main_color
                if f.shape.type == 2: # Polygons
                    vertices = [self.world_to_px(f.body.transform * v) for v in f.shape.vertices]
                    pygame.draw.polygon(self.screen, color, vertices)
                elif f.shape.type == 0: # Circles
                    if isinstance(f.userData, tuple):
                        color = tuple( [max(0, int(c*0.84)) for c in main_color] )
                        if self.world.contactListener.sensors[f.userData[0]][f.userData[1]] == True:
                            color = (0, 185, 0, 255)
                    # TODO: replace with pygame.draw.ellipse()
                    pygame.draw.circle(self.screen, color,
                                   self.world_to_px(b.transform * f.shape.pos),
                                   int(f.shape.radius * self.HPPM))
                else:
                    print(f.shape.type)
    
    
    def move(self, x, y):
        self.center[0] += 100 * x / self.HPPM
        self.center[1] += 100 * y / self.VPPM
        self.updateAABB()
        self.following = False
    
    
    def set_target(self, pos):
        self.set_center(self.center + (pos-self.center)/20)
    
    
    def follow(self, creature):
        """ Auto-update self.center on a given Box2D body position """
        self.following = True
        self.body_to_follow = creature.body
    
    
    def set_pole(self, x, y):
        self.draw_pole = True
        self.flag_pos = (x, y)
    
    
    def updateAABB(self):
        self.aabb = AABB(lowerBound=self.center - (self.width/2, self.height/2),
                         upperBound=self.center + (self.width/2, self.height/2))
    
    
    def render(self):
        """ Render every Box2D bodies on screen's bounding box"""
        if self.following and self.body_to_follow:
            self.set_target(self.body_to_follow.position)
            self.updateAABB()
        
        # render background
        self.screen.fill((0, 0, 0))
        cam_left = self.center.x - self.width/2
        cam_right = self.center.x + self.width/2
        meter = ceil(self.HPPM)
        for i in range(floor(cam_left), ceil(cam_right)):
            px_x = (i-self.center.x+self.width/2) * self.HPPM
            if i%2 == 1:
                pygame.draw.rect(self.screen, (235, 235, 255, 255),
                                 ((px_x, 0), (meter, self.screen_height)))
            else:
                pygame.draw.rect(self.screen, (225, 225, 245, 255),
                                 ((px_x, 0), (meter, self.screen_height)))
        
        if self.draw_pole:
            # Draw background flag pole
            vertices = [(self.flag_pos[0]+1, self.flag_pos[1]-1),
                        (self.flag_pos[0]+1.2, self.flag_pos[1]-1),
                        (self.flag_pos[0]+1.2, self.flag_pos[1]+6),
                        (self.flag_pos[0]+1, self.flag_pos[1]+6)]
            vertices = [self.world_to_px(v) for v in vertices]
            pygame.draw.polygon(self.screen, (140, 140, 140), vertices)
        
        # Render Box2D World
        self.creatures_in_view.clear()
        self.world.QueryAABB(self, self.aabb)
        for c in sorted(self.creatures_in_view, key=lambda c: c.id):
            self.draw_creature(c)
        
        if self.draw_pole:
            # Draw foreground flag pole
            vertices = [(self.flag_pos[0]-1, self.flag_pos[1]+5),
                        (self.flag_pos[0]+1, self.flag_pos[1]+6),
                        (self.flag_pos[0]+1, self.flag_pos[1]+6-1),
                        (self.flag_pos[0]-1, self.flag_pos[1]+5-1)]
            vertices = [self.world_to_px(v) for v in vertices]
            pygame.draw.polygon(self.screen, (180, 180, 180), vertices)
            
            vertices = [(self.flag_pos[0]-1, self.flag_pos[1]-2),
                        (self.flag_pos[0]-1.25, self.flag_pos[1]-2),
                        (self.flag_pos[0]-1.25, self.flag_pos[1]+5),
                        (self.flag_pos[0]-1, self.flag_pos[1]+5)]
            vertices = [self.world_to_px(v) for v in vertices]
            pygame.draw.polygon(self.screen, (140, 140, 140), vertices)
        
        
