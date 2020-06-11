# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:34:11 2019

@author: Baris ALHAN
"""

import pygame
from pygame.locals import *

import os
import numpy as np
import os.path

filepath = os.path.dirname(__file__)

DISPLAY_STATUS = 1

class display:

    # TODO: adjust code to enable no-rendering mode.
    def __init__(self, game):

        self._game = game
        self.render = True if DISPLAY_STATUS else False
        #######################################################################
        #####                   BACKGROUND PROPERTIES                     #####
        #######################################################################
        self._background_color = (150, 150, 150)
        self._text_color = (255, 255, 255)

        self._line_width = 1
        self._line_height = 10
       
        self._vehcl_width = 45
        self._vehcl_height = 25 
        
        self._window_width = int(
            self._game._dynamics._max_veh_inlane *
            (self._vehcl_width + 2 * self._vehcl_height))
        
        self._window_height = int((self._vehcl_height + 2 *
                                   (self._vehcl_height // 2.5)) *
                                  self._game._dynamics._num_lane)

        self._width_of_lane = (
            self._window_height // self._game._dynamics._num_lane)
        #######################################################################
        #######################################################################

        #######################################################################
        #####                INITIALIZATION OF THE PYGAME                 #####
        #######################################################################
        os.environ['SDL_VIDEO_CENTERED'] = '1'
        # Set up the pygame
        pygame.init()

        if(self.render):
            self._window_surface = pygame.display.set_mode((self._window_width,
                                                        self._window_height))

        self._main_clock = pygame.time.Clock()

        pygame.display.set_caption('Highway*')
        # Set the mouse cursor
        pygame.mouse.set_visible(True)
        #######################################################################
        #######################################################################

        #######################################################################
        #####                        THE IMAGES                           #####
        #######################################################################
        if(self.render):
            # _images_veh holds the all possible vehicle images.
            self._images_veh = self.import_images(7)
            # the images of road lines used in displaying
            self._line_image, self._emergency_line_image = self.import_line_images()
            # _vehcls_images holds the image of each vehicle in the game.
            self._vehcls_image = self.assign_images_to_vehicles(self._images_veh)
            # _vehcls_rect holds the rectangle of each vehicle in the game.
            self._vehcls_rect = self.get_vehcls_rect()
            # _lines_rect holds the rectangle of each road line in the map.
            self._lines_rect, self._emergency_lines_rect = self.get_lines_rect()
        #######################################################################
        #######################################################################

    ###########################################################################

    # The method imports images from the Display/Image folder.
    # Possible future erroreneous change:
    # If you change the file path, make sure you really understood that it is
    # the relative path with respect to main.py, not display.py!
    # PyGame related function.
    def import_images(self, num_images):

        #ego_vehcl = self._game.get_vehicle_with_id(self._game._ego_id)

        images_veh = []
        ego_id = int((self._game._dynamics._num_veh - 1) / 2)
        
        
        
        for im_i in range(num_images):
            if im_i == ego_id:
                
                file = os.path.join(filepath[:-8], 'assets/Car6.png') #+ str(im_i)
                images_veh.append(pygame.image.load(file).convert())
            else:
                file = filepath[:-8]+'/assets/Car'+str(im_i)+'.png'
                images_veh.append(pygame.image.load(file).convert())

            images_veh[im_i] = pygame.transform.scale(
                images_veh[im_i], (self._vehcl_width, self._vehcl_height))

        return images_veh

    #PyGame related function.
    def import_line_images(self):

        file2 = os.path.join(filepath[:-8]+'/assets/white.png')
        line_image = pygame.image.load(file2).convert()
        line_image = pygame.transform.scale(
            line_image, (self._line_height, self._line_width))

        file3 = os.path.join(filepath[:-8]+'/assets/white.png')
        emergency_line_image = (pygame.image.load(file3).convert())
        emergency_line_image = pygame.transform.scale(
            emergency_line_image, (self._window_width, self._line_width))

        return line_image, emergency_line_image

    # PyGame related function.
    def assign_images_to_vehicles(self, image_veh):

        result_vehcls_image = []
        ego_id = int((self._game._dynamics._num_veh - 1) / 2)

        for veh in range(self._game._dynamics._num_veh):
            if veh != ego_id:
                new_image = image_veh[0]
                #        time.sleep(0.1)
                result_vehcls_image.append(new_image)
            else:
                new_image = image_veh[ego_id]
                #        time.sleep(0.1)
                result_vehcls_image.append(new_image)


        return result_vehcls_image

    #PyGame related function.
    def get_vehcls_rect(self):

        result_vehcls_rect = []

        for car in range(self._game._dynamics._num_veh):
            new_rect = pygame.Rect(0, 0, self._vehcl_width,
                                   self._vehcl_height)
            result_vehcls_rect.append(new_rect)

        return result_vehcls_rect

    #PyGame related function
    def get_lines_rect(self):
        # The lists to hold the rectangles of the line images.
        lines_rect = []
        emergency_lines_rect = []

        #Determining the position of the road lines.
        for id_of_lane in range(self._game._dynamics._num_lane - 1):
            for coordinates_of_rect in range(self._window_width //
                                             (self._line_height * 2)):
                line_x_coord = coordinates_of_rect * self._line_height * 2
                line_y_coord = (id_of_lane + 1) * self._width_of_lane
                new_line_rect = pygame.Rect(line_x_coord, line_y_coord, 0, 0)
                lines_rect.append(new_line_rect)

        #Determining the position of the emergency lines.
        for id_of_lane in range(self._game._dynamics._num_lane - 1):
            line_y_coord = id_of_lane * self._game._dynamics._num_lane * self._width_of_lane
            new_line_rect = pygame.Rect(0,
                                        id_of_lane * (line_y_coord - 10) + 5, 0,
                                        0)
            emergency_lines_rect.append(new_line_rect)

        return lines_rect, emergency_lines_rect

    # This method is the point where the visual environment
    # of the game is first created.
    # PyGame related function.
    def env_init(self, total_reward):
        if(self.render):
            self._window_surface.fill(self._background_color)

            # Drawing lines to the screen
            for line in range(0, len(self._lines_rect)):
                self._window_surface.blit(self._line_image, self._lines_rect[line])
            # Drawing emergency lines to the screen
            for emergency_line in range(0, len(self._emergency_lines_rect)):
                self._window_surface.blit(
                    self._emergency_line_image,
                    self._emergency_lines_rect[emergency_line])

            half_lane = (self._width_of_lane // 2)

            # Drawing vehicles to the screen
            for vehcl in self._game._vehicles:
                self._vehcls_rect[vehcl._id].center = (vehcl._position[1] * 10,
                                                    half_lane + 2 * half_lane *
                                                    (vehcl._position[0]))
                self._window_surface.blit(self._vehcls_image[vehcl._id],
                                        self._vehcls_rect[vehcl._id])

            pygame.display.update()
            
        self.env_update(total_reward)

    # PyGame related function.
    def env_update(self, total_reward):
        #print("car positions are updated!")
        if(self.render):
            self._window_surface.fill(self._background_color)
            
            ego_vehcl = self._game.get_vehicle_with_id(self._game._ego_id)
            shift = ego_vehcl._position[1] - self._window_width / 20

            # Shifting the lines and drawing to the screen.
            for line in range(0, len(self._lines_rect)):
                self._lines_rect[line].centerx = (
                    self._lines_rect[line].centerx - shift) % self._window_width
                self._window_surface.blit(self._line_image, self._lines_rect[line])
            # Drawing the emergency lines to the screen
            for emergency_line in range(0, len(self._emergency_lines_rect)):
                self._window_surface.blit(
                    self._emergency_line_image,
                    self._emergency_lines_rect[emergency_line])

            half_lane = (self._width_of_lane // 2)
            # hello worlds
            font = pygame.font.SysFont(None, 20)
            self.draw_text("reward: " + str(round(total_reward, 2)), font, self._window_surface,
                        1, 15)
            # Drawing vehicles and speeds to the screen
            for vehcl in self._game._vehicles:
                self._vehcls_rect[vehcl._id].center = ((
                    vehcl._position[1] - shift) * 10, half_lane + 2 * half_lane *
                                                (vehcl._position[0]))
                self._window_surface.blit(self._vehcls_image[vehcl._id],
                                        self._vehcls_rect[vehcl._id])

                self.draw_text(
                    str(int(vehcl._velocity)),
                    font, self._window_surface,
                    self._vehcls_rect[vehcl._id].centerx - 30,
                    self._vehcls_rect[vehcl._id].centery - 5)
                self.draw_text(
                    str(int(vehcl._desired_v)),
                    font, self._window_surface,
                    self._vehcls_rect[vehcl._id].centerx - 10,
                    self._vehcls_rect[vehcl._id].centery - 5)
                self.draw_text(
                    str(format(vehcl._position[1],'.2f')),
                    font, self._window_surface,
                    self._vehcls_rect[vehcl._id].centerx - 30,
                    self._vehcls_rect[vehcl._id].centery - 30)

                self.draw_text(
                    str(format(vehcl._position[0],'.2f')),
                    font, self._window_surface,
                    self._vehcls_rect[vehcl._id].centerx - 30,
                    self._vehcls_rect[vehcl._id].centery - 20)
                self.draw_text(
                    str(vehcl._id), font, self._window_surface,
                    self._vehcls_rect[vehcl._id].centerx - 30,
                    self._vehcls_rect[vehcl._id].centery + 10)
                self.draw_text(
                    str(round(vehcl._acceleration,1)), font, self._window_surface,
                    self._vehcls_rect[vehcl._id].centerx - 10,
                    self._vehcls_rect[vehcl._id].centery + 10)
                
                # if vehcl._is_ego == True:
                #     self.draw_text(
                #     str("EGO"), font, self._window_surface,
                #     self._vehcls_rect[vehcl._id].centerx - 30,
                #     self._vehcls_rect[vehcl._id].centery -30)
                
            pygame.display.flip()
            
        self._main_clock.tick()
        pygame.event.pump()

    # PyGame related function.
    def draw_text(self, text, font, surface, x, y):
        text_obj = font.render(text, 1, self._text_color)
        text_rect = text_obj.get_rect()
        text_rect.topleft = (x, y)
        surface.blit(text_obj, text_rect)
