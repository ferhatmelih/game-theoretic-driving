# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:02:26 2019

@author: Baris ALHAN
"""
# gym time:
import gym
from gym import error, spaces, utils
import os

os.chdir("/home/nlztrk/eatron_gym/gym/gym/envs")
from Game.gameMode import gameMode
from Game.gameDynamics import gameDynamics
from Vehicle.vehicle import vehicle
from Display.display import display
from Vehicle.vehicleAIController import vehicleAIController as AIController
import numpy as np
import pygame, pdb
from pygame.locals import *
import pickle
import time


# TODO: add all vehicle related variables to the vehicle class.
# TODO: add reset method.
# TODO: make [x_pos, y_pos]
#class gamePlay2(gym.Env):
class gamePlay2():
    # gym time
    metadata = {'render.modes': ['human']}
    '''
        Controls the flow of the game.
        Velocity unit is m/s.
        Time unit is second.
    '''

    def __init__(self):
        # pdb.set_trace()
        '''
            Initalizes all necessary modules to start the game.
        '''
        #: int: Time of the simulation (s)
        self._time = 0
        #: float: Analog of the real time required to do just one step (s)
        self._dt = 0.05
        self._gym = 1
        # reward
        self._reward = 0
        self._reward_total = 0

        self._SaveLoad = [0, 0]
        # The below constructors are created with default parameters,
        # to read about the parameters of a class, go to the related class.
        self._mode = gameMode()
        self._dynamics = gameDynamics()
        self._display = display(self)

        #: int: Id of the ego vehicle, it is always at the median index.
        self._ego_id = int((self._dynamics._num_veh - 1) / 2)
        # print("ego is chosen?",self._ego_id)
        #: list of vehicle: Stores vehicle objects
        self._vehicles = None

        # gym action / state  / observations
        high = 10 * np.ones((self._dynamics._num_veh * 4, 1))
        self.action_space = spaces.Discrete(self._dynamics._num_veh - 1)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(self._dynamics._num_actions)

        if self._SaveLoad[1] != True:
            self._vehicles = self.create_vehicles()
            self.spawn_vehicles()
        else:
            self.load()

            #: Starts the visual game environment.
        self._display.env_init()

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
    def spawn_vehicles(self):
        vehicle.generate_init_positions(self,
                                        self._ego_id,
                                        self._dynamics._num_veh,
                                        self._dynamics._num_lane,
                                        self._display._window_width)

        vehicle.generate_init_velocities(self,
                                         self._ego_id,
                                         self._dynamics._num_veh)

        vehicle.calculate_desired_v(self,
                                    self._ego_id,
                                    self._dynamics._num_veh,
                                    self._dynamics._desired_min_v,
                                    self._dynamics._desired_max_v)

        for vehcl in self._vehicles:
            # if vehcl._is_ego == False:
            vehcl._AIController = AIController(vehcl,
                                               self._vehicles,
                                               self._mode,
                                               self._dynamics)
        if self._SaveLoad[0]:
            self.save()

    def get_vehicle_with_id(self, vehcl_id):
        return self._vehicles[vehcl_id]

    def get_input_states(self):
        # s_max = max(abs(states[:, 1] - states[self.ego_veh_id, 1]))
        # s_max = v_ego_max * t
        # v_max = max(V_vehicles)
        v_ego = 0
        s_max = 700  # GOAL_DISTANCE  # distance_long/2+ safety margin will be 50
        # TODO find easier way to catch ego vehicle
        # for vehcl in self._vehicles:
        #     if vehcl._id == True:
        ego_veh = self.get_vehicle_with_id(self._ego_id)
        v_max = ego_veh._desired_v
        v_ego = ego_veh._velocity
        pos_ego = ego_veh._position
        deltaV = 26 - 12 # maximum vehicle speed dif
        # print(v_ego, v_max, pos_ego)
        # v_max = v_d[self._ego_id]
        # input_vector = np.empty()
        # normalize all
        input_vector = np.array([v_ego / v_max])
        # first element of the positon is lane-id
        if pos_ego[0] == 0:
            # input_vector[1] = 0
            input_vector = np.append(input_vector, 0)
        else:
            # input_vector[1] = 1
            input_vector = np.append(input_vector, 1)
        if pos_ego[0] == (self._dynamics._num_lane - 1):
            # input_vector[2] = 0
            input_vector = np.append(input_vector, 0)
        else:
            # input_vector[2] = 1
            input_vector = np.append(input_vector, 1)
        # why not try with APPEND?
        for vehcl in self._vehicles:
            if vehcl._id != self._ego_id:
                # print("pos:",vehcl._position[1]," pos2:",pos_ego[1])
                # print("fucking_Vect:",input_vector)
                input_vector = np.append(input_vector, (vehcl._position[1] - pos_ego[1]) / s_max)
                input_vector = np.append(input_vector, vehcl._velocity - v_max)
                input_vector = np.append(input_vector, (vehcl._position[0] - pos_ego[0]) / 2)
        # for vehcl in self._vehicles:
        #     if vehcl._is_ego != True:
        #         input_vector[3*(i+1)] = (vehcl._position[1] - pos_ego[1]) /s_max
        #         input_vector[3*(i+1)+1] = vehcl._velocity - v_max # this actually refers to maximum desired speed, but let's make it ego for now
        #         input_vector[3*(i+1)+2] = (vehcl._position[0] - pos_ego[0])/2
        #         i = i+1
        # for idx in range(0, self.ego_veh_id):
        #     input_vector[3 * (idx + 1)] = (states[idx, 1] - states[self.ego_veh_id, 1]) / s_max
        #     input_vector[3 * (idx + 1) + 1] = V_vehicles[idx] / v_max
        #     input_vector[3 * (idx + 1) + 2] = (states[idx, 0] - states[self.ego_veh_id, 0]) / 2
        #
        # for idx in range(self.ego_veh_id + 1, NoOfCars):
        #     input_vector[3 * (idx)] = (states[idx, 1] - states[self.ego_veh_id, 1]) / s_max
        #     input_vector[3 * (idx) + 1] = V_vehicles[idx] / v_max
        #     input_vector[3 * (idx) + 2] = (states[idx, 0] - states[self.ego_veh_id, 0]) / 2
        input_vector = input_vector.reshape((len(input_vector), 1))
        return input_vector

    # PyGame related function.
    def terminate(self):
        pygame.quit()
        return False

    def save(self):
        with open("savegame.pkl", "wb") as outputFile:
            pickle.dump(self._vehicles, outputFile, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open("savegame.pkl", "rb") as inputFile:
            self._vehicles = pickle.load(inputFile)

    # PyGame related function.
    def wait_for_player_to_press_key(self):

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
                        return -1
                    elif event.key == pygame.K_RIGHT:
                        return 1
                    else:
                        return 0

    def render(self, mode='human'):
        return 0

    # TODO: explain the general algorithm.
    '''
        1. Determine the longitudinal movement of each vehicle.
            a. Calculate the acceleration in the x direction by using IDM.
        2. Determine the lane change movement.
            a. Find the surrounding vehicles.
            b. Take the lane change decisions according to MOBIL.
    '''

    def check_ego_accidents(self, ego_veh):
        collision = 0
        for vehcl in self._vehicles:
            if vehcl._is_ego == 0:
                dist = abs(vehcl._position[1] - ego_veh._position[1])
                if dist < 3.0:
                    if abs((vehcl._position[0] - ego_veh._position[0])) < 0.5:
                        collision = 1
                        print("***********CRASH***********")
        return collision

    def calculate_reward(self, ego_veh):
        is_done = False
        if self.check_ego_accidents(ego_veh):
            self._reward = self._reward - 100
            self._reward_total = self._reward_total + self._reward
            print("********calc_reward", self._reward_total)

            is_done = True
        else:
            #self._reward = self._reward + 1
            # (ego_veh._velocity / ego_veh._desired_v)
            speed_error = ego_veh._desired_v - ego_veh._velocity
            self._reward = self._reward + (1 / (speed_error*speed_error*speed_error))*10
        return is_done

    def step(self, action):
        if self._gym == 0:
            action = action - 1
        self._reward = 0
        # holds whether the RL episode done or not.
        is_done = False

        # find fron vehicles then calculate delta x, v ,a
        for vehcl in self._vehicles:
            vehcl._delta_v, \
            vehcl._delta_dist = vehcl.calculate_deltas(self._vehicles,
                                                       vehcl)
        ego_veh = self.get_vehicle_with_id(self._ego_id)
        lane_change_lock = 1
        while lane_change_lock:
            # AIControlller calculates acceleration and lane change decision.
            for vehcl in self._vehicles:
                if vehcl._is_ego != True:
                    # 0 has no influence on other cars
                    vehcl._AIController.control(0)
                else:
                    # action is needed to call lane change for EGO
                    vehcl._AIController.control(action)

                if vehcl._is_lane_changing:
                    # Update the position and heading angle.
                    vehcl._position[1], \
                    vehcl._position[0], \
                    vehcl._psi = vehcl.lane_change(vehcl._position,
                                                   vehcl._psi,
                                                   vehcl._velocity,
                                                   vehcl._target_lane)
                else:
                    # Update the x-position of non-lane-changing vehicle.
                    vehcl._position[1] = vehcl._position[1] + (vehcl._velocity * self._dt)

                    # Updating the velocities of the vehicles
                vehcl._velocity = vehcl._velocity + (vehcl._acceleration * self._dt)
            if not ego_veh._is_lane_changing:
                lane_change_lock = 0

        #ego_veh = self.get_vehicle_with_id(self._ego_id)
        #print("#### EGO POS####")
        #print("pos: ", ego_veh._position[1],"lane: ",ego_veh._position[0])
        is_done = self.calculate_reward(ego_veh)
        # Updating the time of the simulation
        self._time = self._time + self._dt
        # Updating the visual environment of the simulation
        self._display.env_update()
        observation = self.get_input_states()
        #print("calc_reward", self._reward)
        #time.sleep(0.1)
        rev_ins = self._reward
        self._reward_total = self._reward_total + self._reward
        if ego_veh._position[1] >= self._mode._distance_goal:
            #reward += 10
            is_done = True
            print("********calc_reward", self._reward_total)
            #rev = self._reward

        return observation, rev_ins, is_done, {}

    def reset(self):

        #: int: Time of the simulation (s)
        self._time = 0
        self._reward_total = 0
        #: float: Analog of the real time required to do just one step (s)
        self._dt = 0.05
        #self._reward = 0
        # self._SaveLoad = SaveLoad
        # The below constructors are created with default parameters,
        # to read about the parameters of a class, go to the related class.
        self._mode = gameMode()
        self._dynamics = gameDynamics()
        self._display = display(self)

        #: int: Id of the ego vehicle, it is always at the median index.
        self._ego_id = int((self._dynamics._num_veh - 1) / 2)

        #: list of vehicle: Stores vehicle objects
        self._vehicles = None

        # gym action / state  / observations
        high = 10 * np.ones((self._dynamics._num_veh * 4, 1))
        self.action_space = spaces.Discrete(self._dynamics._num_veh - 1)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(self._dynamics._num_actions)

        self._vehicles = self.create_vehicles()
        self.spawn_vehicles()
        #: Starts the visual game environment.
        self._display.env_init()
        obs, _, _, _ = self.step(1)
        return obs
