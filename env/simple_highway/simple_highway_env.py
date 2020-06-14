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
# AI controller class contains MOBIL / IDM
from .Vehicle.vehicleAIController import vehicleAIController as AIController

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

class SimpleHighway(gym.Env):

    def __init__(self):
        # Seeding
        self.np_random = None
        
        #np.random.seed(456481)
        self.seed(56555)
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
        self._mode = gameMode()
        self._dynamics = gameDynamics(num_actions=7)
        self._display = display(self)

        #: int: Id of the ego vehicle, it is always at the median index.
        self._ego_id = int((self._dynamics._num_veh	 - 2) / 2)
        #: bool: is ego blocked? if not, we should re-spawn every car!
        self._is_ego_blocked = False
        #: list of vehicle: Stores vehicle objects
        self._vehicles = None
        # TODO: implement more generic agent state space
        # gym action / state  / observations => pos0 pos1 vel
        high = (10 * np.ones((self._dynamics._num_veh * 3 - 1, 1)))
        # self.action_space = spaces.Discrete(self._dynamics._num_veh - 1)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(self._dynamics._num_actions)

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
        self.DEBUG_LIMITS = True


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
                                               self._dynamics)
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

    def get_input_states(self):
        v_ego = 0
        s_max = 100  # GOAL_DISTANCE  # distance_long/2+ safety margin will be 50
        # TODO find easier way to catch ego vehicle
        ego_veh = self.get_vehicle_with_id(self._ego_id)
        #front_id = ego_veh._AIController.find_front_vehicle(self._vehicles, ego_veh._position)
        v_max = ego_veh._desired_v
        v_ego = ego_veh._velocity
        pos_ego = ego_veh._position
        v_init = 15
        max_delta_v = 14
        v_min = 12

        a_max = 5
        # normalized ego_speed
        input_vector = np.array([(v_ego - v_min) / (v_max - v_init)], dtype="f")
        # normalized acceleration
        #input_vector = np.append(input_vector, ego_veh._acceleration / a_max)
        # normalized lateral position
        input_vector = np.append(input_vector, pos_ego[0] / (self._dynamics._num_lane -1))
        #input_vector = np.append(input_vector, ego_veh._acceleration/1)

        for vehcl in self._vehicles: #reversed makes states upside down, this makes the performance worse!
            if vehcl._id != self._ego_id:
                if abs(vehcl._position[1]) - pos_ego[1] < 100:
                    input_vector = np.append(input_vector, (-vehcl._position[1] + pos_ego[1]) / s_max)
                    input_vector = np.append(input_vector, (-vehcl._position[0] + pos_ego[0]) / (self._dynamics._num_lane - 1))
                    input_vector = np.append(input_vector, (-vehcl._velocity + ego_veh._velocity ) / max_delta_v)
                    #input_vector = np.append(input_vector, (vehcl._current_lane - ego_veh._current_lane) / 2)
                    #input_vector = np.append(input_vector, (-vehcl._acceleration + ego_veh._acceleration))
                else:
                    # TODO: asses giving 0 to the distant vehicles help at all?
                    input_vector = np.append(input_vector, [-1.0, 0.0, 0.0])


        input_vector = input_vector.reshape((len(input_vector), 1))
        if(not self.is_init_state_saved):
            self._init_input_state = input_vector
            self.is_init_state_saved = True
        return input_vector

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

    def calculate_reward(self, ego_veh, action):
        is_done = False
        out_of_bounds = 0
        # if there are 3 lanes, beyond 2.01, if there are 2 lanes, beyond 1.01 is out of bounds.
        lane_departure = self._dynamics._num_lane - 1 + 0.01
        if ego_veh._position[0] > lane_departure or ego_veh._position[0] < 0.0:
            out_of_bounds = 1
            self._reward = self._reward - 100.0
            self._num_wrong_exit = self._num_wrong_exit + 1
            is_done = True
        if action != 0:
            # punish each turn
            self._reward = self._reward - 1
        else:
            self._reward = self._reward - 0.2
            foo = 0
            # hard crash punishment
        self._did_accident_just_occur = False
        if self.check_ego_accidents(ego_veh) == 1:
            self._reward = self._reward - 100.0
            self._num_hard_crash = self._num_hard_crash + 1
            is_done = True
            self._did_accident_just_occur = True
            # soft crash punishment
        elif self.check_ego_accidents(ego_veh) == 2:
            self._reward = self._reward - 10.0
            self._num_soft_crash = self._num_soft_crash + 1
        elif not out_of_bounds:
            # Speed Factor - encourage to go faster
            speed_error = ego_veh._desired_v - ego_veh._velocity
            # try to give more pos rewards
            speed_rew = ((ego_veh._velocity  - 15) / (ego_veh._desired_v - 15))
            accel_rew = 0
            #speed_rew = 0
            if abs(ego_veh._acceleration) < 0.005:
                accel_rew = -speed_rew

            # acceleration factor - avoid to settling intermediate speeds
            K_accel = 4 # KP like accel error, settling to the other vehicle speed suffers -0.1
            #accel_rew = ego_veh._acceleration * K_accel
            # cancel accel_rew
            #accel_rew = 0
            if speed_error < 1 and not out_of_bounds:
                speed_rew = 100
                is_done = 1
            self._reward = self._reward + speed_rew + accel_rew
        return is_done


    def gym_to_lanechange_action(self,action):
        action = action - 1
        return action

    def step(self, action):
        
        action = self.gym_to_lanechange_action(action)

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

        is_done = self.calculate_reward(ego_veh, action_init)
        observation = self.get_input_states()
        self._reward_total = self._reward_total + self._reward

        if ego_veh._position[1] >= self._mode._distance_goal:
            is_done = True
        self._steps = self._steps + 1
        summary = {"total_hard_accidents": self._num_hard_crash, "total_soft_accidents": self._num_soft_crash,
                   "total_wrong_exits": self._num_wrong_exit,"init_vehicle_info":self._init_vehicle_info,
                   "init_input_state":self._init_input_state,"num_steps_taken":self._steps}
        return observation, self._reward, is_done, summary

    def reset(self):
        self._is_ego_blocked = 0
        
        self._time = 0
        self._reward_total = 0
        #: float: Analog of the real time required to do just one step (s)
        self._dt = 0.05
        
        self._mode = gameMode()
        self._dynamics = gameDynamics()
        self._display = display(self)

        #: int: Id of the ego vehicle, it is always at the median index.
        self._ego_id = int((self._dynamics._num_veh - 1) / 2)

        #: list of vehicle: Stores vehicle objects
        self._vehicles = None

        self._vehicles = self.create_vehicles()
        while not self._is_ego_blocked:
            self.spawn_vehicles(self.np_random)
        
        #: Starts the visual game environment.
        if self._mode._is_rendering:
            self._display.env_init(self._reward_total)
        
        obs, _, _, _ = self.step(1)
        self.is_init_state_saved = False
        self.num_steps_taken = 0
        
        return obs
