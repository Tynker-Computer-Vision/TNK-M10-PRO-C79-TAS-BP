import math 

def getSensorX(x, angle, sensorAngle, margin):
    return int(x -(margin * math.cos(math.radians(angle+sensorAngle))))

def getSensorY(y, angle, sensorAngle, margin):
    return int(y +(margin * math.sin(math.radians(angle+sensorAngle))))