#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from math import (floor, ceil)
import pygame
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, pi, vec2, queryCallback, AABB)



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
        self.aabb = AABB(lowerBound=self.center-(self.width/2, self.height/2),
                         upperBound=self.center+(self.width/2, self.height/2))
        self.follow = False
        queryCallback.__init__(self)
    
    
    def ReportFixture(self, fixture):
        # TODO: à nettoyer
        # Draw fixture
        shape = fixture.shape
        if shape.type == 2:  # Polygon shape
            # Convert vertices local coord to absolute px coord
            vertices = [self.world_to_px(-self.center+(fixture.body.transform * v)) for v in shape.vertices]
            pygame.draw.polygon(self.screen, (160, 160, 160, 255), vertices)
        
        if shape.type == 0: # Circle shape
            color = (160, 160, 160, 255)
            # Check if it is a sensor
            if isinstance(fixture.userData, tuple):
                color = (200, 200, 200, 255)
                if self.world.contactListener.sensors[fixture.userData[0]][fixture.userData[1]] == True:
                    color = (0, 255, 0, 255)
            # TODO: replace with pygame.draw.ellipse()
            pygame.draw.circle(self.screen, color,
                               self.world_to_px(-self.center+(fixture.body.transform * shape.pos)),
                               int(shape.radius * self.HPPM))
        
        if shape.type == 1 and fixture.userData == 'ground':
            #Ground line
            p0 = self.world_to_px(-self.center+shape.vertices[0])
            p1 = self.world_to_px(-self.center+shape.vertices[1])
            px_points = [(p0[0], p0[1]),
                         (p1[0], p1[1]),
                         (p1[0], self.screen_height),
                         (p0[0], self.screen_height)]
            pygame.draw.polygon(self.screen, (64, 64, 64, 255), px_points)
                
        # Continue the query by returning True
        return True
    
    
    def world_to_px(self, pos):
        """ Reverse height coordinates (up is positive in Box2D) """
        return int(pos[0]*self.HPPM)+self.screen_width//2, int(self.screen_height//2 - pos[1]*self.VPPM)
    
    
    def set_center(self, pos):
        self.center = pos
        self.follow = False
        self.aabb = AABB(lowerBound=self.center-(self.width/2, self.height/2),
                         upperBound=self.center+(self.width/2, self.height/2))
    
    
    def move(self, x, y):
        self.center[0] += x
        self.center[1] += y
        self.aabb = AABB(lowerBound=self.center-(self.width/2, self.height/2),
                         upperBound=self.center+(self.width/2, self.height/2))
    
    def set_target(self, pos):
        self.set_center(self.center + (pos-self.center)/10)
    
    def follow(self, body):
        """ Auto-update self.center on a given Box2D body position """
        ### NOT WORKING YET
        self.follow = True
        self.body_to_follow = body
    
    def render(self):
        """ Render every Box2D bodies on screen's bounding box"""
        if self.follow:
            self.set_target(self.body_to_follow.position)
            self.aabb = AABB(lowerBound=self.center-(self.width/2, self.height/2),
                         upperBound=self.center+(self.width/2, self.height/2))
        
        
        # render background
        #self.screen.fill((0, 0, 0, 0))
        cam_left = self.center.x-self.width/2
        cam_right = self.center.x+self.width/2
        meter = ceil(self.HPPM)
        for i in range(floor(cam_left), ceil(cam_right)):
            px_x = (i-self.center.x+self.width/2) * self.HPPM
            if i%2 == 1:
                pygame.draw.rect(self.screen, (235, 235, 255, 255),
                                 ((px_x, 0), (meter, self.screen_height)))
            else:
                pygame.draw.rect(self.screen, (225, 225, 245, 255),
                                 ((px_x, 0), (meter, self.screen_height)))
        
        # Render Box2D World
        self.world.QueryAABB(self, self.aabb)
        
