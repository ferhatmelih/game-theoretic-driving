class LaneChangeAction:
    NoChange = 0
    Left = -1
    Right = 1


class EgoDriveType:
    RL = 0
    Keyboard = 1
    MOBIL = 2

class TrafficRuleMode:
    NoLaneChange = 0   #             0 -> No lane Change
    USA = 1           # 1 -> USA traffic rules
    UK = 2           # 2 -> UK traffic rules

class ObservationRepresentation:
    ITSC_paper = 0
    BMW_paper = 1

class DriverAction:
    Maintain = 0
    SmallAcc = 1 # 2.5m/s2
    SmallDec = 2
    HardAcc = 3 # 5.0m/s2
    HardDec = 4 
    ToLeftLane = 5
    ToRightLane = 6

