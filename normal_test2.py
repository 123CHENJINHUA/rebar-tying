import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import math
import Robot
import pyrealsense2 as rs
import time
import target_point

import sys
sys.path.insert(0,sys.path[0]+'/../')
from rebar_tying_main import VisionTest3

if __name__ == "__main__":


    RT = np.load('./data/RT.npy')
    # print(RT)

    # 与机器人控制器建立连接，连接成功返回一个机器人对象
    robot = Robot.RPC('192.168.58.2')

    ret = robot.RobotEnable(1)
    time.sleep(2)

    #机器人进入或退出拖动示教模式
    ret = robot.Mode(1) #机器人切入手动模式
    print("机器人切入手动模式", ret)
    ret = robot.DragTeachSwitch(0)
    time.sleep(1)
    # ret = robot.DragTeachSwitch(1)
    # time.sleep(1)
    ret,state = robot.IsInDragTeach()    #查询是否处于拖动示教模式，1-拖动示教模式，0-非拖动示教模式
    if ret == 0:
        print("当前拖动示教模式状态：", state)
    else:
        print("查询失败，错误码为：",ret)

    param = [5.0,10,30,10,5,0]

    tool = 2#工具坐标系编号
    user = 0 #工件坐标系编号

    t_coord = [-56.973, -207.313, 215.131, 0, 0, 0]#[-50.379, -181.657, 196.704, 0, 0, 0]
    # t_coord = [0, 0, 0, 0, 0, 0]
    error = robot.SetToolCoord(tool,t_coord,0,0)
    print("设置工具坐标系错误码",error)

    

    ###------------------ Detection ---------------------------
    ret = robot.GetActualToolFlangePose()
    point_indx = VisionTest3.VisionProcess(np.array(ret[1]))

    print(point_indx)
    

    x=point_indx[0]
    y=point_indx[1]
    z=point_indx[2]

    normal2RotVec = np.array([point_indx[3],point_indx[4],point_indx[5]])
    normal2RotM,_ = cv2.Rodrigues(normal2RotVec)

    R_mask2cam = normal2RotM
    T_mask2cam = np.array([[x],[y],[z]])

    ret = robot.GetActualToolFlangePose()

    R,T = target_point.target_point(R_mask2cam,T_mask2cam,np.array(ret[1]),RT)

    print('X:%f,Y:%f,Z:%f,R:%f,P:%f,Y:%f' %(round(T[0],2),round(T[1],2),round(T[2],2),round(R[0],2),round(R[1],2),round(R[2],2)))
    # desc_pos1 = [ret[1][0],ret[1][1],ret[1][2],round(R[0],2),round(R[1],2),round(R[2],2)]
    # print(desc_pos1)
    # ret = robot.MoveL(desc_pos1, tool, user, vel=5, acc=100)
    # print(ret)

    offset = [0,0,-20,0,0,0]
    desc_pos1 = [round(T[0],2),round(T[1],2),round(T[2],2),round(R[0],2),round(R[1],2),round(R[2],2)]
    print(desc_pos1)
    ret = robot.MoveL(desc_pos1, tool, user, vel=5, acc=100,offset_flag=2,offset_pos = offset)
    print(ret)

    offset = [-45,-45,-20,0,0,-40]
    ret = robot.MoveL(desc_pos1, tool, user, vel=5, acc=100,offset_flag=2, offset_pos=offset)

    offset = [-45,-45,50,0,0,-40]
    ret = robot.MoveL(desc_pos1, tool, user, vel=5, acc=100,offset_flag=2, offset_pos=offset)
    time.sleep(5)

    offset = [-45,-45,-20,0,0,-40]
    ret = robot.MoveL(desc_pos1, tool, user, vel=5, acc=100,offset_flag=2, offset_pos=offset)

    
    desc_pos1 = [-729.799,-349.729, 231.44, -140.858, -2.429, 99.518]
    ret = robot.MoveL(desc_pos1, tool, user, vel=10, acc=100)
                   

                




