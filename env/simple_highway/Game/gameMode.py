# -*- coding: utf-8 -*-


from enum import Enum


class gameMode:
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

