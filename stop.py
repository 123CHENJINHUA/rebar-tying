import sys
sys.path.append("C:/Users/JH CHEN/Desktop/robot_arm/fairino-python-sdk-v2.0.0\windows/libfairino/fairino")

print(sys.path)

import Robot
# 与机器人控制器建立连接，连接成功返回一个机器人对象
robot = Robot.RPC('192.168.58.2')

ret = robot.RobotEnable(0)