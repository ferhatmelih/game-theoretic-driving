# -*- coding: utf-8 -*-

import math,pdb
import copy
import numpy as np

#from .Behavior.policy import Policy_Level_0 as tfasd



class Policy(object):
    def __init__(self):
        self.close_th = 30
        self.nominal_th = 70
        self.approaching_th = 0.0
        self.stable_th = 0.5
        self.a1 = 2.5
        self.a2 = 5.0

    def asses_relative_location(self,agent_relative):
        close = False
        nominal = False
        far = False
        agent_relative = abs(agent_relative)
        if(agent_relative<self.close_th):
            close = True
        elif(agent_relative<self.nominal_th):
            nominal = True
        elif(agent_relative>self.nominal_th):
            far = True
        else:
            print("[Error] in lead position")
        
        return close,nominal,far

    def asses_relative_speed(self,agent_relative):
        approaching = False
        stable = False
        movingaway = False
        if(agent_relative<self.approaching_th):
            approaching = True
        elif(agent_relative<self.stable_th):
            stable = True
        elif(agent_relative>self.stable_th):
            movingaway = True
        else:
            print("[Error] in lead position")
        return approaching,stable,movingaway

    def action_to_acc_map(self,action):
        if(action==0):
            return 0.0
        elif(action==1):
            return self.a1
        elif(action==2):
            return -self.a1
        elif(action==3):
            return self.a2
        elif(action==4):
            return -self.a2
        else:
            return 0.0
    
    def action_to_lane_change_map(self,action):
        if(action==5):
            return -1 #left
        elif(action==6):
            return 1 #right
        else:
            return 0
    

class Policy_Level_0(Policy):
    def __init__(self):
        super().__init__()
        pass
    
    def get_action(self,observation):
        # decides action on only from lead vehicle
        lead_relative_position = observation[0]
        lead_relative_speed = observation[1]
        
        close,nominal,far = self.asses_relative_location(lead_relative_position)
        approaching,stable,movingaway = self.asses_relative_speed(lead_relative_speed)

        action = 0
        if( (close and stable) or (nominal and approaching) ):
            action = 2 #decelearte 
        elif( close and approaching):
            action = 4 # hard decelerate 
        else:
            action = 0
        return action


class Policy_Level_1(Policy_Level_0):
    def __init__(self):
        super().__init__()
        pass

class Policy_Level_2(Policy_Level_0):
    def __init__(self):
        super().__init__()
        pass

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
    

    def control(self, action):
        if(self._vehcl._is_ego == False):                
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
            lane_change_to_be = self._policy.action_to_lane_change_map(action)
            self._vehcl._acceleration = acc_to_be
        else:
            lane_change_to_be = action
        

        if self._vehcl._is_ego == True:
            if self._vehcl._is_lane_changing is False:
                self._vehcl._lane_change_decision = lane_change_to_be
                if self._vehcl._lane_change_decision != 0:
                    self._vehcl._is_lane_changing = True

                    self._vehcl._target_lane = self._vehcl._current_lane + action
                    # if self._vehcl._target_lane < 0 or self._vehcl._target_lane > 2:
                    #     self._vehcl._target_lane = self._vehcl._current_lane
                    #     self._vehcl._is_lane_changing = False
                    if(self.DEBUG_LANE_CHANGE ):
                        print("id:{}, decision:{}, targetLane:{}".format(self._id,
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

