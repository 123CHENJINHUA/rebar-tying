"""
Framework   : OpenCV Aruco
Description : Calibration of camera and using that for finding pose of multiple markers
Status      : Working
References  :
    1) https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
    2) https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
    3) https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html
"""

import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import math
import Robot
import pyrealsense2 as rs
import time


if __name__ == "__main__":

    # Connect to robot
    robot = Robot.RPC('192.168.58.2')

    ret = robot.RobotEnable(1)
    time.sleep(2)

    # Change robot's mode to dragteach 
    ret = robot.Mode(1) #机器人切入手动模式
    print("机器人切入手动模式", ret)
    ret = robot.DragTeachSwitch(0)
    time.sleep(1)
    ret = robot.DragTeachSwitch(1)  #机器人切入拖动示教模式，必须在手动模式下才能切入拖动示教模式
    print("机器人切入拖动示教模式", ret)
    time.sleep(1)
    ret,state = robot.IsInDragTeach()    #查询是否处于拖动示教模式，1-拖动示教模式，0-非拖动示教模式
    if ret == 0:
        print("当前拖动示教模式状态：", state)
    else:
        print("查询失败，错误码为：",ret)

    robot_pos = []

    # open D435 camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)

    # load mtx and dist
    cv_file = cv2.FileStorage("./hand_eye_calibration/charuco_camera_calibration2.yaml", cv2.FILE_STORAGE_READ)

    # Note : we also have to specify the type
    # to retrieve otherwise we only get a 'None'
    # FileNode object back instead of a matrix
    mtx = cv_file.getNode("camera_matrix").mat()
    dist = cv_file.getNode("dist_coeff").mat()

    p3_x,p3_y = 0,0

    R_list = []
    T_list = []


    ###------------------ ARUCO TRACKER ---------------------------
    while (True):

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays

        depth_image = np.asanyarray(depth_frame.get_data())

        color_image = np.asanyarray(color_frame.get_data())

        frame_copy = color_image.copy()
        #if ret returns false, there is likely a problem with the webcam/camera.
        #In that case uncomment the below line, which will replace the empty frame 
        #with a test image used in the opencv docs for aruco at https://www.docs.opencv.org/4.5.3/singlemarkersoriginal.jpg
        # frame = cv2.imread('./images/test image.jpg') 

        # operations on the frame
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # set dictionary size depending on the aruco marker selected
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        board = cv2.aruco.CharucoBoard((3, 3), 0.041, 0.030, aruco_dict)
        # detector parameters can be set here (List of detection parameters[3])
        parameters = aruco.DetectorParameters()
        parameters.adaptiveThreshConstant = 10

        # lists of ids and the corners belonging to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # font for displaying text (below)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # check if the ids list is not empty
        # if no check is added the code will crash
        if np.all(ids != None):

            # draw a square around the markers
            aruco.drawDetectedMarkers(frame_copy, corners)
            retval,charucoCorners,charucoIds =cv2.aruco.interpolateCornersCharuco(corners,ids,color_image,board)
            if retval:
                retval, rvec_, tvec_ = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, mtx, dist, None, None)

                # If pose estimation is successful, draw the axis
                if retval:
                    org_point = np.array([[0,0,0]],dtype = np.float64)
                    p1, _ = cv2.projectPoints(org_point, rvec_, tvec_, mtx, dist)
                    p1_x = int(p1[0][0][0])
                    p1_y = int(p1[0][0][1])
                
                    total = np.sqrt(sum(np.power((charucoCorners[0][0] - charucoCorners[1][0]), 2)))+np.sqrt(sum(np.power((charucoCorners[1][0] - charucoCorners[3][0]), 2)))+np.sqrt(sum(np.power((charucoCorners[3][0] - charucoCorners[2][0]), 2)))+np.sqrt(sum(np.power((charucoCorners[2][0] - charucoCorners[0][0]), 2)))
                    ratio = 15/(total/4)
                
                    if (p1_x >1280 or p1_y>1280):
                        continue
                    cv2.circle(frame_copy,(p1_x,p1_y),10,(0,0,255))

                    cv2.drawFrameAxes(frame_copy, mtx, dist, rvec_, tvec_, length=0.05, thickness=2)

                    R_mask2cam = np.zeros((3,3),dtype = np.float64)
                    cv2.Rodrigues(rvec_,R_mask2cam)
                    R_list.append(R_mask2cam)
                    T_list.append(tvec_)

                    # If Aruco mark detected well, record the TCP data.
                    ret = robot.GetActualToolFlangePose()
                    robot_pos.append(ret[1])


                else:
                    # code to show 'No Ids' when no markers are found
                    cv2.putText(frame_copy, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

        # display the resulting frame
        cv2.imshow('frame',frame_copy)
        k=cv2.waitKey(1)
        if k==27:
            break

    # When everything done, release the capture
    np.save('./data/R_list.npy',R_list)
    np.save('./data/T_list.npy',T_list)
    filename_R = open('./data/R_list.txt','w')
    filename_T = open('./data/T_list.txt','w')

    for value in R_list:
        filename_R.write(str(value))
        filename_R.write('\n')
    for value in T_list:
        filename_T.write(str(value[0][0]))
        filename_T.write(', ')
        filename_T.write(str(value[1][0]))
        filename_T.write(', ')
        filename_T.write(str(value[2][0]))
        filename_T.write('\n')



    np.save('./data/Robot_data.npy',robot_pos)
    filename = open('./data/Robot_data.txt','w')
    for value in robot_pos:
        filename.write(str(value[0]))
        filename.write(', ')
        filename.write(str(value[1]))
        filename.write(', ')
        filename.write(str(value[2]))
        filename.write(', ')
        filename.write(str(value[3]))
        filename.write(', ')
        filename.write(str(value[4]))
        filename.write(', ')
        filename.write(str(value[5]))
        filename.write('\n')

    pipeline.stop()
    cv2.destroyAllWindows()


