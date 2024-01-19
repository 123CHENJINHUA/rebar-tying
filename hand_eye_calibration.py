import cv2
import numpy as np
from math import *

T_list = np.load('./data/T_list.npy')
R_list = np.load('./data/R_list.npy')
Robot_data = np.load('./data/Robot_data.npy')

def myRPY2R_robot(x, y, z):
    x=x*np.pi/180
    y=y*np.pi/180
    z=z*np.pi/180

    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz@Ry@Rx
    return R

#用于根据位姿计算变换矩阵
def pose_robot(x, y, z, Tx, Ty, Tz):
    thetaX = x / 180 * pi
    thetaY = y / 180 * pi
    thetaZ = z / 180 * pi
    R = myRPY2R_robot(thetaX, thetaY, thetaZ)
    t = np.array([[Tx], [Ty], [Tz]])
    RT1 = np.column_stack([R, t])  # 列合并
    RT1 = np.row_stack((RT1, np.array([0,0,0,1])))
    # RT1=np.linalg.inv(RT1)
    return RT1

R_all_end_to_base_1 = []
T_all_end_to_base_1 = []
R_all_chess_to_cam_1 = []
T_all_chess_to_cam_1 = []

for value in Robot_data:
    R_all_end_to_base_1.append(myRPY2R_robot(value[3],value[4],value[5]))
    T_all_end_to_base_1.append(value[:3].reshape(3, 1))

for value in T_list:
    T_all_chess_to_cam_1.append(1000*value[:3,0].reshape(3, 1))

for value in R_list:
    R_all_chess_to_cam_1.append(value)

R,T=cv2.calibrateHandEye(R_all_end_to_base_1,T_all_end_to_base_1,R_all_chess_to_cam_1,T_all_chess_to_cam_1,cv2.CALIB_HAND_EYE_HORAUD)#手眼标定
RT=np.column_stack((R,T))
RT = np.row_stack((RT, np.array([0, 0, 0, 1])))#即为cam to end变换矩阵
print('相机相对于末端的变换矩阵为：')
print(RT)
filename = open('./data/RT.txt','w')
for value in RT:
    filename.write(str(value))
    filename.write('\n\n')
np.save('./data/RT.npy',RT)

result = []
for i in range(len(R_all_end_to_base_1)):

    RT_end_to_base=np.column_stack((R_all_end_to_base_1[i],T_all_end_to_base_1[i]))
    RT_end_to_base=np.row_stack((RT_end_to_base,np.array([0,0,0,1])))
    # print(RT_end_to_base)

    RT_chess_to_cam=np.column_stack((R_all_chess_to_cam_1[i],T_all_chess_to_cam_1[i]))
    RT_chess_to_cam=np.row_stack((RT_chess_to_cam,np.array([0,0,0,1])))
    # print(RT_chess_to_cam)

    RT_cam_to_end=np.column_stack((R,T))
    RT_cam_to_end=np.row_stack((RT_cam_to_end,np.array([0,0,0,1])))
    # print(RT_cam_to_end)

    RT_chess_to_base=RT_end_to_base@RT_cam_to_end@RT_chess_to_cam#即为固定的棋盘格相对于机器人基坐标系位姿
    # RT_chess_to_base=np.linalg.inv(RT_chess_to_base)

    result.append(RT_chess_to_base[:3,:])

filename = open('./data/result.txt','w')
for value in result:
    filename.write(str(value))
    filename.write('\n\n')