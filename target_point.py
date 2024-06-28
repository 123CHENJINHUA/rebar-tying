from math import *
import numpy as np
import cv2

def myRPY2R_robot(x, y, z):
    x=x*np.pi/180
    y=y*np.pi/180
    z=z*np.pi/180

    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz@Ry@Rx
    return R

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

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = atan2(R[2,1] , R[2,2])
        y = atan2(-R[2,0], sy)
        z = atan2(R[1,0], R[0,0])
    else :
        x = atan2(-R[1,2], R[1,1])
        y = atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])
 

def target_point(camera_rvec,camera_tvec,robot_point,RT):


    R_all_end_to_base_1 = myRPY2R_robot(robot_point[3],robot_point[4],robot_point[5])
    T_all_end_to_base_1 = robot_point[:3].reshape(3, 1)

    T_all_chess_to_cam_1 = 1000*camera_tvec[:3,0].reshape(3, 1)
    R_all_chess_to_cam_1 = camera_rvec

    RT_end_to_base=np.column_stack((R_all_end_to_base_1,T_all_end_to_base_1))
    RT_end_to_base=np.row_stack((RT_end_to_base,np.array([0,0,0,1])))
    # print(RT_end_to_base)

    RT_chess_to_cam=np.column_stack((R_all_chess_to_cam_1,T_all_chess_to_cam_1))
    RT_chess_to_cam=np.row_stack((RT_chess_to_cam,np.array([0,0,0,1])))
    # print(RT_chess_to_cam)

    
    RT_cam_to_end=RT
    # print(RT_cam_to_end)

    RT_chess_to_base=RT_end_to_base@RT_cam_to_end@RT_chess_to_cam#即为固定的棋盘格相对于机器人基坐标系位姿
    

    R = rotationMatrixToEulerAngles(RT_chess_to_base[:3,:3])*180/np.pi
    T = RT_chess_to_base[:3,3]

    return R,T

def offset_trans(tool_cord, end_point, Offect):

    tool_cord = [0,0,0,0,0,0]
    R_end_to_base = myRPY2R_robot(end_point[3],end_point[4],end_point[5])
    T_end_to_base = end_point[:3].reshape(3, 1)

    tool_point = np.array([0,0,0,0,0,0])
    for i in range(len(tool_cord)):
        tool_point[i] = tool_cord[i] + Offect[i]

    R_tool_to_end = myRPY2R_robot(tool_point[3],tool_point[4],tool_point[5])
    T_tool_to_end = tool_point[:3].reshape(3, 1)

    RT_end_to_base=np.column_stack((R_end_to_base,T_end_to_base))
    RT_end_to_base=np.row_stack((RT_end_to_base,np.array([0,0,0,1])))

    RT_tool_to_end=np.column_stack((R_tool_to_end,T_tool_to_end))
    RT_tool_to_end=np.row_stack((RT_tool_to_end,np.array([0,0,0,1])))

    RT_tool_to_base=RT_end_to_base@RT_tool_to_end

    R = rotationMatrixToEulerAngles(RT_tool_to_base[:3,:3])*180/np.pi
    T = RT_tool_to_base[:3,3]

    return R,T


if __name__ == "__main__":

    T_list = np.load('./data/T_list.npy')
    R_list = np.load('./data/R_list.npy')
    Robot_data = np.load('./data/Robot_data.npy')
    RT = np.load('./data/RT.npy')

    R,T = target_point(R_list[0],T_list[0],Robot_data[0],RT)
    print(R)
    print(T)