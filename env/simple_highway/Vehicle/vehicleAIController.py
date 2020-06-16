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
        
        self.level_k = level_k

        if level_k==0 :
            self._policy = Policy_Level_0()
        elif level_k == 1:
            self._policy = Policy_Level_1()
        elif level_k == 2:
            self._policy = Policy_Level_2()

        self.DEBUG_LANE_CHANGE = True
    

    def make_observation_level_0(self,lane_pos_list,pos_vehicle_map):    

        agent_vehicle = self._vehcl

        #TODO:FMD get the default values from paper 
        relative_loc_longitudanal = 200
        relative_speed = 10

        agent_lane = agent_vehicle._current_lane
        agent_pos_x = agent_vehicle._position[1]
        agent_speed = agent_vehicle._velocity # the name _velocity should have been speed
        index_in_lane = lane_pos_list[agent_lane].index(agent_pos_x)
        l_front = index_in_lane  + 1
        agent_lane_list = lane_pos_list[agent_lane]

        if vehicleAIController.is_index_valid(l_front,agent_lane_list):
            front_pos_x = agent_lane_list[l_front]
            relative_loc_longitudanal = front_pos_x - agent_pos_x
            
            front_agent_ref = pos_vehicle_map[front_pos_x]
            front_agent_speed = front_agent_ref._velocity
            relative_speed = front_agent_speed - agent_speed

        observation = [relative_loc_longitudanal,relative_speed]

        return observation 


    def apply_agent_action(self,action):
        acc_to_be = self._policy.action_to_acc_map(action)
        lane_change_action = self._policy.action_to_lane_change_map(action)
        self._vehcl._acceleration = acc_to_be

        self._vehcl._lane_change_decision = lane_change_action
        if self._vehcl._lane_change_decision!=0:
            self._vehcl._is_lane_changing = True
            self._vehcl._target_lane = self._vehcl._current_lane + lane_change_action
            self._vehcl._is_lane_changing = True        



    def make_observation_level_k(self,lane_pos_list,pos_vehicle_map):
        agent_ref = self._vehcl
        num_lanes = self._dynamics.num_lane 
        obs = vehicleAIController.level_k_observation(agent_ref,num_lanes,lane_pos_list,pos_vehicle_map)

        return obs



    

    def control(self, action,lane_pos_list,pos_vehicle_map):
        # agents just perfrom their action 
        if(self._vehcl._is_ego == False):
            if self._vehcl._is_lane_changing is False:
                # make observation
                if self.level_k == 0:
                    observation = self.make_observation_level_0(lane_pos_list,pos_vehicle_map)
                else:
                    observation = self.make_observation_level_k(lane_pos_list,pos_vehicle_map)
                
                action = self._policy.get_action(observation)
                
                self.apply_agent_action(action)

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
    
    @staticmethod
    def relative_pos_to_bin(rel_pos):
        out = 1.0
        if abs(rel_pos) > 27:
            out = 1.0 # far
        elif abs(rel_pos) > 11:
            out = 0.6 # nomina;
        elif abs(rel_pos) >=0.0:
            out = 0.2 #near
        if(rel_pos==0): rel_pos = 0.0001
        sign = rel_pos/abs(rel_pos)
        out = sign*out
        return out

    @staticmethod
    def relative_vel_to_bin(rel_v):
        out = 0.0
        if abs(rel_v) > 0.36:
            out = 1.0 # approach or move away
        elif abs(rel_v) >= 0.0:
            out = 0.0 # stable;
        
        if rel_v == 0.0: rel_v = 0.00001
        sign = rel_v/abs(rel_v)
        out = sign*out
        return out

    @staticmethod
    def is_index_valid(in_index,in_list):
        if(in_index<0):
            return False
        elif(in_index >= len(in_list)):
            return False
        else:
            return True
    
    @staticmethod
    def get_index_of_front_and_back(lane_lst,ego_pos_x):
        front_index = -1
        rear_index = -1
        if(len(lane_lst)>0):
            MAX_NUM = 5e9
            diff_list = [MAX_NUM if i-ego_pos_x < 0 else i-ego_pos_x for i in lane_lst]
            min_val = min(diff_list)
            if min_val == MAX_NUM:
                # no front vehicle
                rear_index = len(lane_lst)
            else:
                front_index = diff_list.index(min_val)
                rear_index = front_index - 1
        return front_index,rear_index

    @staticmethod
    def get_obs_x_v(front_index,lane_lst,ego_pos_x,ego_speed,pos_vehicle_map):
        obs_rel_x = 1.0
        obs_rel_v = 1.0
        if vehicleAIController.is_index_valid(front_index,lane_lst):
            agent_pos_x = lane_lst[front_index]
            rel_pos = agent_pos_x - ego_pos_x
            obs_rel_x = vehicleAIController.relative_pos_to_bin(rel_pos)

            agent_ref = pos_vehicle_map[agent_pos_x]
            agent_speed = agent_ref._velocity
            rel_vel = agent_speed - ego_speed
            obs_rel_v = vehicleAIController.relative_vel_to_bin(rel_vel) 
        return obs_rel_x,obs_rel_v

    @staticmethod
    def level_k_observation(vehicle_ref,num_lanes,lane_pos_list,pos_vehicle_map):
        
        """ 
        sort all the vehicles in order of their positions on the highway

        the location matrix is then in the form

        also, create a dictionary on the form 

        pos_vehicle_map = {'position':'vehicle_ref'}

        for each lane, there is a list of sorted vectors


        Create observation from these matrixes. 

        • Relative position and velocity of the car in front on the
        same lane (lane l).
        • Relative position and velocity of the car in front on the
        left lane (lane l +1).
        • Relative position and velocity of the car in rear on the
        left lane (lane l +1).
        • Relative position and velocity of the car in front on the
        right lane (lane l −1).
        • Relative position and velocity of the car in rear on the
        right lane (lane l −1).
        • Own lane number (l) 

        Observation length  = 2*3*2 + 1 = 13
        
        """ 

        # fill observation with default values
        obs = np.full((13, ), 1)

        # find front agent
        ego_lane = vehicle_ref._current_lane
        ego_pos_x = vehicle_ref._position[1]
        ego_speed = vehicle_ref._velocity # the name _velocity should have been speed
        index_in_lane = lane_pos_list[ego_lane].index(ego_pos_x)
        l_front = index_in_lane  + 1
        l_rear = index_in_lane - 1
        ego_lane_list = lane_pos_list[ego_lane]

        obs = np.full((13, ), 1)
        obs[0],obs[1] = vehicleAIController.get_obs_x_v(l_front,ego_lane_list,ego_pos_x,ego_speed,pos_vehicle_map)
        obs[2],obs[3] = vehicleAIController.get_obs_x_v(l_rear,ego_lane_list,ego_pos_x,ego_speed,pos_vehicle_map)

        if ego_lane < num_lanes-1:
            right_lane = lane_pos_list[ego_lane + 1]
            l_right_front,l_right_rear = vehicleAIController.get_index_of_front_and_back(right_lane,ego_pos_x)
            obs[4],obs[5] = vehicleAIController.get_obs_x_v(l_right_front,ego_lane_list,ego_pos_x,ego_speed,pos_vehicle_map)
            obs[6],obs[7] = vehicleAIController.get_obs_x_v(l_right_rear,ego_lane_list,ego_pos_x,ego_speed,pos_vehicle_map)   

        if ego_lane > 0:
            left_lane = lane_pos_list[ego_lane - 1]
            l_left_front,l_left_rear = vehicleAIController.get_index_of_front_and_back(left_lane,ego_pos_x)
            obs[8],obs[9] = vehicleAIController.get_obs_x_v(l_left_front,ego_lane_list,ego_pos_x,ego_speed,pos_vehicle_map)
            obs[10],obs[11] = vehicleAIController.get_obs_x_v(l_left_rear,ego_lane_list,ego_pos_x,ego_speed,pos_vehicle_map)   

        obs[12] = ego_lane

        return obs
    ###########################################################################
    ###########################################################################       

