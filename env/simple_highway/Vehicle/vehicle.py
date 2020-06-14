# -*- coding: utf-8 -*-
import math,pdb
import numpy as np


from .vehicleControlModels import PID
from .vehicleControlModels import dynModel as DynModel
from .vehicleAIController import vehicleAIController as AIController

class vehicle():
 
    def __init__(self,
                 vehcl_id,
                 is_ego):
        #: int: Each vehicle has its own unique id.
        self._id = vehcl_id
        #: bool: If the vehicle is controlled by user, it is called ego.
        self._is_ego = is_ego
        #: pair of floats: (Y_pos, X_pos)
        self._position = np.zeros(2)
        #: int: (meters/second)
        self._velocity = 0
        #: float: the velocity and distance difference with the front vehicle.
        self._delta_v, self._delta_dist = 0, 0
        #: float: the desired velocity for traveling
        self._desired_v = 0
        #: float: (meters/second^2)
        self._acceleration = 0
        #: AIController(class): Non-ego agents are directed by AIController.
        self._AIController = None
        #: bool: Is the vehicle currently in the process of lane changing?
        self._is_lane_changing = False
        #  0 => Go straight
        #  1 => Go to the right
        # -1 => Go to the left
        #: int
        self._prev_lane = 0
        #: int:
        self._current_lane = 0
        #: int: 
        self._lane_change_decision = 0
        #: int: Holds the target lane id after lane change decision is made.
        self._target_lane = 0
        #: float: Heading angle of the vehicle. It is used while changing lane.
        self._psi = 0
   
        #if front_vehcl:
         #   print(' id:{}, front:{}, pos:{}, front_pos:{}, delta_dist:{}'.format(self._id,front_vehcl._id,
          #        self._position[1], front_vehcl._position[1],delta_dist))
        
    # Two-Point Visual Control Model of Steering
    # For reference, please check the paper itself.
    # Perform lane change in continuous space with using controller
    # psi -> heading angle.
    def lane_change(self, pos, psi, v, target_lane):
        
        pid = PID()
        dynModel = DynModel()

        x_pos, y_pos = pos[1], pos[0]
        
        dify = target_lane - y_pos
        # Two points are seleceted within 5 meters and 100 meters,
        # then angles are calculated and fed to PID
        near_error = np.subtract(np.arctan2(dify, 5), psi)
        far_error = np.subtract(np.arctan2(dify, 100), psi)
        # TODO: the last variable is dt.
        pid_out = pid.update(near_error, far_error, 0.05)
        # TODO: the last variable is dt.
        z = [x_pos, y_pos, psi, v, 0.05]

        x_next, y_next, psi_next = dynModel.update(z, pid_out)
        # we will use the lane_change complete to make sure there is only 1 lane change during a certain time
        #TODO when other vehicles are lane changing, we have to restrict lane_change_complete flag with only ego
        #if self._is_ego is True:
        lane_change_complete = self.check_lane_change_done(y_next)
        if lane_change_complete:
            y_next = target_lane
        # else:
        #     lane_change_complete = False
        return x_next, y_next, psi_next, lane_change_complete,
    
    def check_lane_change_done(self, y_pos):
        laneChangeCompleteThreshold = 0.08
        if abs(self._target_lane - y_pos) <= laneChangeCompleteThreshold:
            self._is_lane_changing = False
            self._lane_change_decision = 0
            return True
        else:
            return False
            
    ###########################################################################
    ######                    STATIC METHODS                              #####
    ###########################################################################
    
    '''
     This static method that generates the initial positions of vehicles.
     
     Aim: Vehicles are distributed to the highway without collisions.
     
     The algorithm is as follows:
         1. Assign each vehicle to a free lane
             (2. Afterwards, For each lane, evaluate the exact position of each vehicle.)
             2. For each lane in lanes:
                 3. First, randomly calculate a point
                    for the first car in that lane.
                 4. Afterwards, assign a position to the remanining
                    vehicles one by one, ensuring that they do not collide.
     
    The coordinates list is sorted at the end and by that way the ego vehicle 
    is always the vehicle in the middle.
        
     Inputs: init_range, delta_dist
         init_range : range of the initialization horizon(meters)
         delta_dist: minimum distance between vehicles (meters)
         
     Outputs: coordinates
         coordinates : Position of vehicles => [LaneID, X_pos]
    '''
    
    @staticmethod
    def generate_init_positions (game,
                                 ego_id,
                                 num_vehcl,
                                 num_lane,
                                 window_width,
                                 np_random,
                                 init_range=100,
                                 delta_dist=30):
        
        # Safety check for the distance of the vehicles.
        if delta_dist < 10:
            delta_dist = 10
    
        #The result list stores the lane of each vehicle.
        lane_list = []
        #The result list stores the positions of each vehicle.
        #[(LaneID, X_pos)]
        positions = np.zeros((num_vehcl, 2))

        #first randomly select lanes for each vehicle
        for veh in range(0, num_vehcl):
            # Randomly chose lane id for each vehicle
            #lane_list.append(np.random.randint(0, num_lane))
            lane_list.append(np_random.randint(0, num_lane))


        #The map that stores [LaneID <-> number of vehicles in that lane]
        fullness_of_lanes = {x: lane_list.count(x) for x in lane_list}
    
        # Temporary list to store the positions of the each vehicle.
        # [(LaneID : X_pos)]
        tmp_positions = []
    
        # 2nd step of the algorithm.
        for lane, num_vehicles_inlane in fullness_of_lanes.items():
            # First, chose a point for the first vehicle in the selected lane
            # The second parameter in the randomness ensures that there is no
            # accumulation of vehicles at the end of the initial range.
            tmp_point = (np_random.uniform(
                0, init_range - ((num_vehicles_inlane - 1) * delta_dist)))
    
            tmp_positions.append([lane, tmp_point])
    
            for veh_num in range(0, num_vehicles_inlane - 1):
                # put other vehicles in that lane to the remaining space
                tmp_point = (np_random.uniform(
                    tmp_point + delta_dist, (init_range -
                                             (num_vehicles_inlane - 2 - veh_num) * delta_dist)))
                
                tmp_positions.append([lane, tmp_point])
        
        positions = np.asarray(tmp_positions).reshape(positions.shape)
        positions = positions[positions[:, 1].argsort()]
        positions[:, 1] = positions[:, 1] - positions[ego_id,
                                                            1] + window_width / 20

        for vehcl in game._vehicles:
            vehcl._position = positions[vehcl._id]


    # The method that returns the initial velocities of vehicles
    @staticmethod
    def generate_init_velocities(game,ego_id,num_vehcl,np_random):
        #The result list stores the initial velocities of each vehicle.
        init_v = np.zeros((num_vehcl))
        # initial velocity for the ego vehicle is between 15/s and 17 m/s
        init_v[ego_id] = 15 #np.random.uniform(15, 17)

        # randomly define initial speeds for the rear vehicles
        for rear_id in range(0, ego_id):
            init_v[rear_id] = np_random.uniform(15, 20)
            #print("speed of rear vehicles:", init_v[rear_id])
        # randomly define initial speeds for the front vehicles
        for front_id in range(ego_id + 1, num_vehcl):
            init_v[front_id] = np_random.uniform(14, 16)
            #print("speed of front vehicles:", init_v[front_id])


        for vehcl in game._vehicles:
            vehcl._velocity = init_v[vehcl._id]
            
            
    # The method calculates the desired max. velocities for the each vehicle.
    # The desired max. velocity of the ego vehicle is 25 m/s
    @staticmethod
    def calculate_desired_v(game,
                            ego_id,
                            num_vehcl,
                            desired_min_v,
                            desired_max_v,
                            np_random):
        result = np.zeros((num_vehcl))
        # initial velocity for the ego vehicle is between 10m/s and 15 m/s
        result[ego_id] = np.array([25])

        # randomly define initial speeds for the rear vehicles
        for rear_id in range(0, ego_id):
            # we want fast rear vehicles
            result[rear_id] = np_random.uniform(desired_min_v + 6, desired_max_v)
            #print("speed of rear vehicles:", init_v[rear_id])
        # randomly define initial speeds for the front vehicles
        for front_id in range(ego_id + 1, num_vehcl):
            # front vehicle cannot be faster than 20 kph
            result[front_id] = np_random.uniform(desired_min_v, desired_max_v - 6)
            #print("speed of front vehicles:", init_v[front_id])
        for vehcl in game._vehicles:
            vehcl._desired_v = result[vehcl._id]


        # result = np.random.uniform(desired_min_v,
        #                            desired_max_v,
        #                            num_vehcl)
        # result[ego_id] = np.array([25])
        # result.shape = (len(result))
        #
        # for vehcl in game._vehicles:
        #     vehcl._desired_v = result[vehcl._id]
        #print("desired assignment: ", result[vehcl._id], " id:", vehcl._id)

    
    # Calculates delta_v and delta_dist with the front vehicle for each vehicle
    @staticmethod
    def calculate_deltas(vehicles, vehcl):
        front_vehcl = AIController.find_front_vehicle(vehicles, vehcl._position)
        
        if front_vehcl:
            # TODO: check it wheter negative or positive.
            delta_v = vehcl._velocity - front_vehcl._velocity
            delta_dist = abs(vehcl._position[1] - front_vehcl._position[1])
        else:
            delta_v = 0
            delta_dist = 10**5
        
        return delta_v, delta_dist        
    
    ###########################################################################
    ###########################################################################
