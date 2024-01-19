import sys
import time
sys.path.append("./mycode")

import Robot
# 与机器人控制器建立连接，连接成功返回一个机器人对象
robot = Robot.RPC('192.168.58.2')

ret = robot.RobotEnable(1)
#机器人进入或退出拖动示教模式
ret = robot.Mode(1) #机器人切入手动模式
print("机器人切入手动模式", ret)
ret = robot.DragTeachSwitch(0)
time.sleep(2)
ret = robot.DragTeachSwitch(1)