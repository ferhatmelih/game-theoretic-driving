# -*- coding: utf-8 -*-

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# gameMode class contains ruleMode, Rendering, total game distance
from .Game.gameMode import gameMode
# gameDynamics class contains number of lanes, number of vehicles
from .Game.gameDynamics import gameDynamics
# vehicle class represents each vehicle in terms of position, velocity, lane change decision, controller
from .Vehicle.vehicle import vehicle
# display class contains the pygame rendering setup such as plotting velocity / id of vehicles / their pictures etc
from .Display.display import display
# AI controller 
from .Vehicle.vehicleAIController import vehicleAIController as AIController
# types
from .Vehicle.policy import DriverAction,DistanceBins

import numpy as np
import pygame, pdb
from pygame.locals import *
import pickle
import time

import datetime

"""
# action space
0 maintain 
1 accelerate at a1 = 2.5m/s2
2 decelarate at -a1 
3 hard acc at a2 = 5m/s2
4 hard dec at -a2
5 change lane to left
6 change lane to right 

"""
MAX_VELOCITY = 40 #m/s
MIN_VELOCITY = 10 #m/s
MEAN_VELOCITY = 25 

FAR_AWAY = 200 # meters

class SimpleHighway(gym.Env):

    def __init__(self,agent_level_k=0,num_lane=3,highway_length=2e3,glob_conf={},logger=None):
        
        #>> Behavior additions
        self.agent_level_k = agent_level_k
        self.num_lane = num_lane 
        self.reward_coefs = np.array([0.6, 0.3, 0.1, 0.1])
        self.size_obervation = 13
        self.seed_number = 3235
        self.should_render = True

        self.logger = logger

        ## overrite some params
        if 'agent_level_k' in glob_conf:
            self.agent_level_k = glob_conf['agent_level_k']
        if 'RENDER' in glob_conf:
            self.should_render = glob_conf['RENDER']

        if 'num_lane' in glob_conf:
            self.num_lane = glob_conf['num_lane']
        
        if 'seed' in glob_conf:
            self.seed_number = glob_conf['seed']
                
        if 'w1' in glob_conf:
            self.reward_coefs[0] = glob_conf['w1']
        if 'w2' in glob_conf:
            self.reward_coefs[1] = glob_conf['w2']
        if 'w3' in glob_conf:
            self.reward_coefs[2] = glob_conf['w3']
        if 'w4' in glob_conf:
            self.reward_coefs[3] = glob_conf['w4']

        self._num_episodes = 0
        #<< 
        # Seeding
        self.np_random = None
        
        self.seed(self.seed_number)

        #: int: Time of the simulation (s)
        self._time = 0
        #: float: Analog of the real time required to do just one step (s)
        self._dt = 0.05
        self._gym = 1
        # reward
        self._reward = 0
        self._reward_total = 0
        self._steps = 0
        # The below constructors are created with default parameters,
        # to read about the parameters of a class, go to the related class.
        self._mode = gameMode(distance_goal=highway_length,is_rendering=self.should_render)
        self._dynamics = gameDynamics(num_actions=7,num_lane=self.num_lane)
        self._display = display(self)

        #: int: Id of the ego vehicle, it is always at the median index.
        self._ego_id = int((self._dynamics._num_veh	 - 2) / 2)
        #: bool: is ego blocked? if not, we should re-spawn every car!
        self._is_ego_blocked = False
        #: list of vehicle: Stores vehicle objects
        self._vehicles = None

        high = (2 * np.ones((self.size_obervation, 1)))
        # self.action_space = spaces.Discrete(self._dynamics._num_veh - 1)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(self._dynamics._num_actions)

        self._vehicles = self.create_vehicles()
        while not self._is_ego_blocked:
            self.spawn_vehicles(self.np_random)
        
        #: Starts the visual game environment.
        if self._mode._is_rendering:
            self._display.env_init(self._reward_total)

        # statistics
        self._num_hard_crash = 0
        self._num_soft_crash = 0
        self._num_wrong_exit = 0
        self._init_vehicle_info = []
        self._init_input_state = []
        self.is_init_state_saved = False
        self.num_steps_taken = 0
        self.last_game_state = []
        self._did_accident_just_occur = False
        self.DEBUG_LIMITS = False


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #: Creates the vehicle objects in the game.
    def create_vehicles(self):

        vehicles = []
        # Checks whether it is the ego vehicle or not.
        for vehcl_id in range(self._dynamics._num_veh):
            if vehcl_id != self._ego_id:
                vehicles.append(vehicle(vehcl_id, False))
            else:
                vehicles.append(vehicle(vehcl_id, True))

        return vehicles

    #: Spawns the vehicle to the map with related values calculated.
    def spawn_vehicles(self, np_random=None):
        vehicle.generate_init_positions(self,
                                        self._ego_id,
                                        self._dynamics._num_veh,
                                        self._dynamics._num_lane,
                                        self._display._window_width,
                                        np_random)

        vehicle.generate_init_velocities(self,
                                         self._ego_id,
                                         self._dynamics._num_veh,
                                         np_random)

        vehicle.calculate_desired_v(self,
                                    self._ego_id,
                                    self._dynamics._num_veh,
                                    self._dynamics._desired_min_v,
                                    self._dynamics._desired_max_v,
                                    np_random)
        self._init_vehicle_info = []
        for vehcl in self._vehicles:
            # if vehcl._is_ego == False:
            vehcl._current_lane = self.calculate_current_lane(vehcl._position[0], vehcl._current_lane)
            lane_id = vehcl._current_lane
            init_pos_x = vehcl._position[0]
            init_pos_y = vehcl._position[1]
            desired_v = vehcl._desired_v
            
            self._init_vehicle_info.append([lane_id,init_pos_y,desired_v])

            vehcl._AIController = AIController(vehcl,
                                               self._vehicles,
                                               self._mode,
                                               self._dynamics,
                                               level_k=self.agent_level_k)
            if vehcl._is_ego == True:
                if vehcl._AIController.find_front_vehicle(self._vehicles, vehcl._position):
                    self._is_ego_blocked = True
                    #print("BLOCKED")

                else:
                    # we are inhibiting the scenarios that egovehicle get high scores without a lane change
                    self._is_ego_blocked = False
                    #print("FREE")

    def get_vehicle_with_id(self, vehcl_id):
        return self._vehicles[vehcl_id]

    def relative_pos_to_bin(self,rel_pos):
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

    def relative_vel_to_bin(self,rel_v):
        out = 0.0
        if abs(rel_v) > 0.36:
            out = 1.0 # approach or move away
        elif abs(rel_v) >= 0.0:
            out = 0.0 # stable;
        
        if rel_v == 0.0: rel_v = 0.00001
        sign = rel_v/abs(rel_v)
        out = sign*out
        return out

    def is_index_valid(self,in_index,in_list):
        if(in_index<0):
            return False
        elif(in_index >= len(in_list)):
            return False
        else:
            return True
    
    def get_index_of_front_and_back(self,lane_lst,ego_pos_x):
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

    def get_obs_x_v(self, front_index,lane_lst,ego_pos_x,ego_speed,pos_vehicle_map):
        obs_rel_x = 1.0
        obs_rel_v = 1.0
        if self.is_index_valid(front_index,lane_lst):
            agent_pos_x = lane_lst[front_index]
            rel_pos = agent_pos_x - ego_pos_x
            obs_rel_x = self.relative_pos_to_bin(rel_pos)

            agent_ref = pos_vehicle_map[agent_pos_x]
            agent_speed = agent_ref._velocity
            rel_vel = agent_speed - ego_speed
            obs_rel_v = self.relative_vel_to_bin(rel_vel) 
        return obs_rel_x,obs_rel_v

    def get_input_states(self):

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


        lane_pos_list = []
        pos_vehicle_map = {}
        for i in range(self.num_lane):
            lane_pos_list.append([])
                
        for vehcl in self._vehicles:
            lane = vehcl._current_lane
            pos = vehcl._position
            pos_x = pos[1]
            if pos_x not in pos_vehicle_map:
                pos_vehicle_map[pos_x] = vehcl
            else:
                print("two equal pos_x vals found")
            lane_pos_list[lane].append(pos_x)

        ## sort all the columns
        for lst in lane_pos_list: lst.sort()

        # fill observation with default values
        obs = np.full((13, ), 1)

        # find front agent
        ego_vehicle =  self._vehicles[self._ego_id]
        ego_lane = ego_vehicle._current_lane
        ego_pos_x = ego_vehicle._position[1]
        ego_speed = ego_vehicle._velocity # the name _velocity should have been speed
        index_in_lane = lane_pos_list[ego_lane].index(ego_pos_x)
        l_front = index_in_lane  + 1
        l_rear = index_in_lane - 1
        ego_lane_list = lane_pos_list[ego_lane]

        obs[0],obs[1] = self.get_obs_x_v(l_front,ego_lane_list,ego_pos_x,ego_speed,pos_vehicle_map)
        obs[2],obs[3] = self.get_obs_x_v(l_rear,ego_lane_list,ego_pos_x,ego_speed,pos_vehicle_map)

        if ego_lane < self.num_lane-1:
            right_lane = lane_pos_list[ego_lane + 1]
            l_right_front,l_right_rear = self.get_index_of_front_and_back(right_lane,ego_pos_x)
            obs[4],obs[5] = self.get_obs_x_v(l_right_front,ego_lane_list,ego_pos_x,ego_speed,pos_vehicle_map)
            obs[6],obs[7] = self.get_obs_x_v(l_right_rear,ego_lane_list,ego_pos_x,ego_speed,pos_vehicle_map)   

        if ego_lane > 0:
            left_lane = lane_pos_list[ego_lane - 1]
            l_left_front,l_left_rear = self.get_index_of_front_and_back(left_lane,ego_pos_x)
            obs[8],obs[9] = self.get_obs_x_v(l_left_front,ego_lane_list,ego_pos_x,ego_speed,pos_vehicle_map)
            obs[10],obs[11] = self.get_obs_x_v(l_left_rear,ego_lane_list,ego_pos_x,ego_speed,pos_vehicle_map)   

        obs[12] = ego_lane

        ego_lead_relative = FAR_AWAY
        if self.is_index_valid(l_front,ego_lane_list):
            lead_pos_x = ego_lane_list[l_front] 
            ego_lead_relative = lead_pos_x - ego_pos_x
        
        return obs,ego_lead_relative

    # PyGame related function.
    def terminate(self):
        pygame.quit()
        return False
        
    # PyGame related function.
    def wait_for_player_to_press_key(self):
        keys = pygame.key.get_pressed()
        if keys[K_TAB]:
            return 1

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.terminate()
                if event.type == pygame.KEYDOWN:
                    # escape quits
                    if event.key == pygame.K_ESCAPE:
                        self.terminate()
                        return
                    elif event.key == pygame.K_LEFT:
                        return 0
                    elif event.key == pygame.K_RIGHT:
                        return 2
                    else:
                        return 1


    def check_ego_accidents(self, ego_veh):
        collision = 0
        for vehcl in self._vehicles:
            if vehcl._is_ego == 0:
                dist = abs(vehcl._position[1] - ego_veh._position[1])
                if dist < 5.0:
                    if vehcl._current_lane == ego_veh._current_lane: #abs((vehcl._position[0] - ego_veh._position[0])) < 0.5
                        collision = 1
                        #print("***********CRASH***********")
                elif dist < 7.5:
                    if vehcl._current_lane == ego_veh._current_lane: #abs((vehcl._position[0] - ego_veh._position[0])) < 0.5
                        collision = 2
                        #print("***********SOFT-CRASH***********")
        return collision

    def calculate_current_lane(self, position, cur_lane):
        # lanes are numbered as 0: top, 1: middle, 2:bottom lane
        current_lane = cur_lane
        if position < 0.45:
            current_lane = 0
        elif position > 0.5 and position < 1.5:
            current_lane = 1
        elif position > 1.55:
            current_lane = 2

        return current_lane

    def lead_distance_to_reward(self,rel_lead_dist):
        close_max = DistanceBins.Close[1]
        nominal_max = DistanceBins.Nominal[1]
        out = 1
        if abs(rel_lead_dist) < close_max:
            out = -1
        elif abs(rel_lead_dist) < nominal_max:
            out = 0
        else:
            out = 1
        return out


    def calc_comfort_reard(self,action):
        out = 0
        if action == DriverAction.Maintain:
            out = 0
        elif action == DriverAction.SmallAcc or action == DriverAction.SmallDec:
            out = -0.25
        elif action == DriverAction.HardAcc or action == DriverAction.HardAcc:
            out = -0.50
        elif action == DriverAction.ToLeftLane or action == DriverAction.ToRightLane:
            out = -1
        else:
            out = 0
        
        return out
    

    def calculate_reward(self, ego_veh, action,ego_lead_relative=200):
        """
        w1 = 0.6, w2 = 0.3, and w3 = 0.1 etc

        R = c + s + d + e

        c  if crash -1 
        s   (ego_speed - MEAN)/MAX
        d  if lead car is close:-1 nominal:0 else:1
        e  if action is maintain:0 Acc:-0.25 HardAcc:-0.5
            left:-1 right:-1

        """
        is_done = False

        collision_reward = 0
        dist_reward = 0
        e_comf_reward = 0 
        if self.check_ego_accidents(ego_veh) == 1:
            self._num_hard_crash = self._num_hard_crash + 1
            collision_reward = -1
            is_done = True

        # if there are 3 lanes, beyond 2.01, if there are 2 lanes, beyond 1.01 is out of bounds.
        lane_departure = self._dynamics._num_lane - 1 + 0.01
        if ego_veh._position[0] > lane_departure or ego_veh._position[0] < 0.0:
            collision_reward = -1
            self._num_wrong_exit = self._num_wrong_exit + 1
            is_done = True

        if ego_veh._position[1] >= self._mode._distance_goal:
            is_done = True

        ego_speed = ego_veh._velocity 

        speed_reward = (ego_speed - MEAN_VELOCITY)/MAX_VELOCITY

        dist_reward = self.lead_distance_to_reward(ego_lead_relative)

        e_comf_reward = self.calc_comfort_reard(action)

        final_reward = np.dot(self.reward_coefs,
            np.array([collision_reward, speed_reward, dist_reward, e_comf_reward]))


        self._reward = final_reward

        return is_done,final_reward


    def step(self, action):
        # find fron vehicles then calculate delta x, v ,a
        for vehcl in self._vehicles:
            vehcl._delta_v, \
            vehcl._delta_dist = vehcl.calculate_deltas(self._vehicles,
                                                       vehcl)
        ego_veh = self.get_vehicle_with_id(self._ego_id)
        lane_change_lock = 1
        lane_change_complete = 0
        i = 0
        action_init = action
        while lane_change_lock and i < 30:
            i = i + 1
            # AIControlller calculates acceleration and lane change decision.
            for vehcl in self._vehicles:
                if vehcl._is_ego is not True:
                    # get the action according to the policy 
                    # for now olny Level_0 policy exists

                    vehcl._AIController.control(0)
                else:
                    # action is needed to call lane change for EGO
                    if lane_change_complete or i > 15:
                        # this is to force to go straight after completing a lane change
                        action = 0
                    vehcl._AIController.control(action)

                if vehcl._is_lane_changing:
                    # Update the position and heading angle.
                    vehcl._position[1], \
                    vehcl._position[0], \
                    vehcl._psi, lane_change_complete = vehcl.lane_change(vehcl._position,
                                                   vehcl._psi,
                                                   vehcl._velocity,
                                                   vehcl._target_lane)
                    vehcl._current_lane = self.calculate_current_lane(vehcl._position[0],vehcl._current_lane)
                    if vehcl._is_ego:
                        if(self.check_ego_accidents(vehcl) == 1):
                            lane_change_lock = 0
                            break
                else:
                    # Update the x-position of non-lane-changing vehicle.
                    vehcl._position[1] = vehcl._position[1] + (vehcl._velocity * self._dt)
                    vehcl._current_lane = self.calculate_current_lane(vehcl._position[0], vehcl._current_lane)
                # Updating the velocities of the vehicles
                if(MAX_VELOCITY>vehcl._velocity>MIN_VELOCITY):
                    vehcl._velocity = vehcl._velocity + (vehcl._acceleration * self._dt)
                else:
                    if(self.DEBUG_LIMITS):
                        print("Speed limit reached for agent ID %d, speed is: %2.1f"%(vehcl._id,vehcl._velocity))
            self._time = self._time + self._dt
            # Updating the visual environment of the simulation
            if self._mode._is_rendering:
                self._display.env_update(self._reward_total)
            if not ego_veh._is_lane_changing and i >= 30:
                lane_change_lock = 0

        observation,ego_lead_relative = self.get_input_states()
        is_done,reward = self.calculate_reward(ego_veh, action_init,ego_lead_relative==ego_lead_relative)
        
        self._reward = reward
        self._reward_total = self._reward_total + self._reward

        self._steps = self._steps + 1
        if(self.logger):
            self.logger.log_scalar("total_hard_accidents", self._num_hard_crash)
            self.logger.log_scalar("reward_total", self._reward_total)
            self.logger.log_scalar("reward", self._reward)
            

        summary = {"total_hard_accidents": self._num_hard_crash, "total_soft_accidents": self._num_soft_crash,
                   "total_wrong_exits": self._num_wrong_exit,"init_vehicle_info":self._init_vehicle_info,
                   "init_input_state":self._init_input_state,"num_steps_taken":self._steps}
        return observation, self._reward, is_done, summary

    def reset(self):
        self._is_ego_blocked = 0
        self._num_episodes +=1
        self._time = 0
        self._reward_total = 0
        #: float: Analog of the real time required to do just one step (s)
        self._dt = 0.05

        #: int: Id of the ego vehicle, it is always at the median index.
        self._ego_id = int((self._dynamics._num_veh - 1) / 2)

        #: list of vehicle: Stores vehicle objects
        self._vehicles = None

        self._vehicles = self.create_vehicles()
        while not self._is_ego_blocked:
            self.spawn_vehicles(self.np_random)
        
        
        obs, _, _, _ = self.step(1)
        self.is_init_state_saved = False
        
        if(self.logger):
            self.logger.log_scalar("episode_num", self._num_episodes)
            self.logger.log_scalar("num_steps_taken", self.num_steps_taken)

        self.num_steps_taken = 0
        return obs
