#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
import pygame
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody, pi, vec2, queryCallback, AABB)



# ======================== CAMERA CLASS ========================
class Camera:
    class myQueryCallback(queryCallback):
        def __init__(self, camera):
            queryCallback.__init__(self)
            self.cam = camera
            self.screen = self.cam.screen
        
        def ReportFixture(self, fixture):
            # TODO: à nettoyer
            # Draw fixture
            shape = fixture.shape
            if shape.type == 2:  # Polygon shape
                # Convert vertices local coord to absolute px coord
                vertices = [self.cam.world_to_px(-self.cam.center+(fixture.body.transform * v)) for v in shape.vertices]
                pygame.draw.polygon(self.screen, (200, 200, 200, 255), vertices)
            if shape.type == 0: # Circle shape
                color = (255, 255, 255, 255)
                # Check if it is a sensor
                if isinstance(fixture.userData, tuple) and \
                    self.cam.world.contactListener.sensors[fixture.userData[0]][fixture.userData[1]] == True:
                    color = (0, 255, 0, 255)
                # TODO: replace with pygame.draw.ellipse()
                pygame.draw.circle(self.screen, color,
                                   self.cam.world_to_px(-self.cam.center+(fixture.body.transform * shape.pos)),
                                   int(shape.radius * self.cam.HPPM))
            
            # Continue the query by returning True
            return True
    
    def __init__(self, world, screen, w, h, center=(0,0)):
        self.world = world
        self.width = w
        self.height = h
        self.screen = screen
        self.SW = self.screen.get_width()
        self.SH = self.screen.get_height()
        self.HPPM = self.SW / self.width      # Horizontal pixels per meter
        self.VPPM = self.SH / self.height    # Vertical pixels per meter
        self.center = vec2(center)
        self.aabb = AABB(lowerBound=self.center-(self.width/2, self.height/2),
                         upperBound=self.center+(self.width/2, self.height/2))
        self.callback = self.myQueryCallback(self)
        self.follow = False
    
    def world_to_px(self, pos):
        """ Reverse height coordinates (up is positive in Box2D) """
        return int(pos.x*self.HPPM)+self.SW//2, int(self.SH//2 - pos.y*self.VPPM)
    
    def set_center(self, pos):
        self.center = pos
        self.follow = False
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
        """ Render every Box2D bodies """
        if self.follow:
            self.set_target(self.body_to_follow.position)
            self.aabb = AABB(lowerBound=self.center-(self.width/2, self.height/2),
                         upperBound=self.center+(self.width/2, self.height/2))
                         
        self.world.QueryAABB(self.callback, self.aabb)
