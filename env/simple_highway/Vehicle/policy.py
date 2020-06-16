
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

class DistanceBins:
    Close = [0,11]
    Nominal = [11,27]
    Far = [11,300]


class DriverAction:
    Maintain = 0
    SmallAcc = 1 # 2.5m/s2
    SmallDec = 2
    HardAcc = 3 # 5.0m/s2
    HardDec = 4 
    ToLeftLane = 5
    ToRightLane = 6



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