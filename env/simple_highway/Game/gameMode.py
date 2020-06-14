# -*- coding: utf-8 -*-


from enum import Enum


class gameMode:
    '''
        The class for the settings of the general game mode.
        @Params:
            is_rendering: if false, there will be no display
            p_yield = 0; mobil does not yield from the rear fast vehicles
            rule_mode:
                0 -> No lane Change
                1 -> USA traffic rules
                2 -> UK traffic rules
            behavior_mode: The behavioral models of the actors.
                # TODO: add behavioral models.
            distance_goal: The goal distance for RL to arrive. When it is
                           arrived, RL episode is done.
    '''

    def __init__(self,
                 p_yield=0,
                 is_rendering=True,
                 rule_mode=0,
                 behavior_mode=0,
                 distance_goal=5000):
        self._is_rendering = is_rendering
        self._rule_mode = rule_mode
        self._behavior_mode = behavior_mode
        self._distance_goal = distance_goal
        self._p_yield = p_yield



'''
    # TODO: change the place of this method and explain what it does.
    # RL related function.
    # check paper table 2.
    def get_input_states(self, states, V_vehicles, t):
        # s_max = max(abs(states[:, 1] - states[self.ego_veh_id, 1]))
        # s_max = v_ego_max * t
        # v_max = max(V_vehicle_s)
        s_max = self._goal_distance  # distance_long/2+ safety margin will be 50
        v_max = self._desired_v[self._ego_id]
        input_vector = np.zeros((3 * self._dynamics._num_veh, 1))
        input_vector[0] = np.random.normal(V_vehicles[self._ego_id] / v_max,
                                           0.1)  # our speed unceert

        if states[self._ego_id, 0] == 0:
            input_vector[1] = 0
            
        else:
            input_vector[1] = 1
        if states[self._ego_id, 0] == (self._dynamics._num_lane - 1):
            input_vector[2] = 0
        else:
            input_vector[2] = 1

        for idx in range(0, self._ego_id):
            input_vector[3 * (idx + 1)] = np.random.normal(
                ((states[idx, 1] - states[self._ego_id, 1]) / s_max),
                0.3)  # uncert for distance
            input_vector[3 * (idx + 1) + 1] = np.random.normal(
                V_vehicles[idx] / v_max, 0.2)  # uncert for speed
            input_vector[3 * (idx + 1) + 2] = (
                states[idx, 0] - states[self._ego_id, 0]) / 2

        for idx in range(self._ego_id + 1, self._dynamics._num_veh):
            input_vector[3 * (idx)] = np.random.normal(
                (states[idx, 1] - states[self._ego_id, 1]) / s_max,
                0.3)  # uncert for distance
            input_vector[3 * (idx) + 1] = np.random.normal(
                V_vehicles[idx] / v_max, 0.2)  # uncert for speed
            input_vector[3 * (idx) + 2] = (
                states[idx, 0] - states[self._ego_id, 0]) / 2
        return input_vector
    '''
