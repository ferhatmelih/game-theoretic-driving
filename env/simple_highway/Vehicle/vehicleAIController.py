# -*- coding: utf-8 -*-

import math,pdb
import copy
import numpy as np

from .policy import Policy_Level_0, Policy_Level_1,Policy_Level_2


class vehicleAIController: 
    
    def __init__(self, vehcl, vehicles, mode, dynamics,level_k = 0): 
        
        self._vehcl = vehcl
        self._vehicles = vehicles
        self._mode = mode
        self._dynamics = dynamics
        
        self._id = self._vehcl._id
        
        if(level_k==0):
            self._policy = Policy_Level_0()
        elif level_k == 1:
            self._policy = Policy_Level_1()
        elif level_k == 2:
            self._policy = Policy_Level_2()

        self.DEBUG_LANE_CHANGE = True
    

    def apply_agent_action(self):
        if self._vehcl._is_lane_changing is False:
                            
            position = (self._vehcl._position[0],self._vehcl._position[1])

            front_vehcl = vehicleAIController.find_front_vehicle(self._vehicles,
                                                                position)

            #TODO:FMD get the default values from paper 
            relative_loc_longitudanal = 200
            relative_speed = 10

            if(front_vehcl):
                relative_loc_longitudanal = front_vehcl._position[1] - position[1]
                relative_speed = front_vehcl._velocity - self._vehcl._velocity

            observation = [relative_loc_longitudanal,relative_speed]
            action = self._policy.get_action(observation)
            acc_to_be = self._policy.action_to_acc_map(action)
            lane_change_action = self._policy.action_to_lane_change_map(action)
            self._vehcl._acceleration = acc_to_be

            self._vehcl._lane_change_decision = lane_change_action
            if self._vehcl._lane_change_decision!=0:
                self._vehcl._is_lane_changing = True
                self._vehcl._target_lane = self._vehcl._current_lane + lane_change_action
                self._vehcl._is_lane_changing = True        

    def control(self, action):

        if(self._vehcl._is_ego == False):
            # agents just perfrom their action 
            self.apply_agent_action()

        elif self._vehcl._is_ego == True:
            ## for ego, action is already given by the network. just apply it. 
            if self._vehcl._is_lane_changing is False:
                
                acc_to_be = self._policy.action_to_acc_map(action)

                self._vehcl._acceleration = acc_to_be

                lane_change_action = self._policy.action_to_lane_change_map(action)
                
                self._vehcl._lane_change_decision = lane_change_action
                
                if self._vehcl._lane_change_decision != 0:
                    self._vehcl._is_lane_changing = True
                    self._vehcl._target_lane = self._vehcl._current_lane + lane_change_action
                    if(self.DEBUG_LANE_CHANGE ):
                        print("current lane:{}, decision:{}, targetLane:{}".format(self._vehcl._current_lane,
                         self._vehcl._lane_change_decision,self._vehcl._target_lane))


                   
    ###########################################################################
    ######                    STATIC METHODS                              #####
    ########################################################################### 
                   
    # TODO: find functions are not optimal.
    @staticmethod
    def find_rear_vehicle(vehicles, position):
        
        min_dist = 99999999
        result_vehcl = None
        result_id = -1
        
        vehcl_id = 0    
        for vehcl in vehicles:
            if abs(vehcl._position[0] - position[0]) < 0.7 :
                if position[1] - vehcl._position[1] > 0.0001:
                    if position[1] - vehcl._position[1] < min_dist:
                        min_dist = position[1] - vehcl._position[1]
                        result_id = vehcl_id
            vehcl_id += 1
        
        if result_id!=-1:
            result_vehcl = vehicles[result_id]
            
        return result_vehcl
    
    
    @staticmethod
    def find_front_vehicle(vehicles, position):
        
        min_dist = 99999999
        result_vehcl = None
        result_id = -1
        
        vehcl_id = 0    
        for vehcl in vehicles:
            if abs(vehcl._position[0] - position[0]) < 0.7:
                if  vehcl._position[1] - position[1] > 0.0001:
                    if  vehcl._position[1] - position[1] < min_dist:
                        min_dist = vehcl._position[1] - position[1]
                        result_id = vehcl_id
            vehcl_id += 1
        
        if result_id!=-1:
            result_vehcl = vehicles[result_id]
        
        return result_vehcl
    
    ###########################################################################
    ###########################################################################       

