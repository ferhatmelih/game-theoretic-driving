# -*- coding: utf-8 -*-

import math,pdb
import copy
import numpy as np

"""
# action space
0 maintain 
1 accelerate at a1 = 2.5m/s2
2 decelarate at -a1 
3 hard acc at a2 = 5m/s2
4 hard dec at -a2
5 change lane to left
6 change lane to right 



other agents 

close. 0-30 
nominal. 30-70
far 70-..

"""



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


class vehicleAIController: 
    
    def __init__(self, vehcl, vehicles, mode, dynamics): 
        
        self._vehcl = vehcl
        self._vehicles = vehicles
        self._mode = mode
        self._dynamics = dynamics
        
        self._id = self._vehcl._id
        self._policy = Policy_Level_0()
        self.DEBUG_LANE_CHANGE = True
    
    
    # TODO: check is there any calculation made in the step related to the deltas.
    def calculate_acceleration(vehcl_v,
                               vehcl_pos,
                               vehcl_desired_v,
                               front_vehcl_v,
                               front_vehcl_pos):
        
        new_delta_v = vehcl_v - front_vehcl_v
        new_delta_dist = front_vehcl_pos[1] - vehcl_pos[1]
            
        new_acceleration = vehicleAIController.IDM(vehcl_v,
                                                   vehcl_desired_v,
                                                   new_delta_v,
                                                   new_delta_dist)
        return new_acceleration   
    
    
    '''
        One of the main functions for traffic simulation.
        It controls the longitudinal accelerations of vehicles.
        For reference, please check the paper itself.
        
        Inputs:
            velocity   : current speed of the vehicle
            desired_v   : desired speed of the vehicle
            delta_v     : Speed diffrence with the leading vehicle
            delta_dist  : Gap with the leading vehicle
        Outputs: 
            acceleration : Reference Acceleration
    '''
    @staticmethod
    def IDM(velocity,
            desired_v,
            delta_v,
            delta_dist):
        
        amax = 0.7  # Maximum acceleration    (m/s^2) 
        S = 4  # Acceleration exponent
        d0 = 2  # Minimum gap
        T = 1.6  # Safe time headaway    (s)
        b = 1.7  # Desired deceleration (m/s^2)
        
        dstar = d0 + (velocity * T) + ((velocity * delta_v) / (2 * math.sqrt(amax * b)))
        acceleration = amax * (1 - math.pow( ( velocity/desired_v ), S) - math.pow( (dstar/(delta_dist + 0.001)) , 2) )
    
        # Lower bound for acceleration, -20 m/s^2
        if acceleration < -5:
            acceleration = -5
        
        return acceleration
    

    def control(self, action):

        if(self._vehcl._is_ego == False):                
            position = (self._vehcl._position[0],
                                self._vehcl._position[1])

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
        # Checks if the traffic rule enables lane changing.
        if self._mode._rule_mode !=0 or  self._vehcl._is_ego == True:
            if self._vehcl._is_lane_changing is False and self._vehcl._is_ego is not True:
                self._vehcl._lane_change_decision = lane_change_to_be
                #print("id:{}, decision:{}".format(self._id, self._vehcl._lane_change_decision))
                if self._vehcl._lane_change_decision!= 0:
                    self._vehcl._is_lane_changing = True
                    # mobil has a reverse left right order
                    self._vehcl._target_lane = (self._vehcl._current_lane +
                                                self._vehcl._lane_change_decision)
            elif self._vehcl._is_lane_changing is False and self._vehcl._is_ego is True:
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

