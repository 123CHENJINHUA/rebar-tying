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


width = 1920
height = 1080
fps = 60
img_size = 1080

mark_size = 0.1

cap = cv2.VideoCapture(1)

cap.set(3, width)  #设置宽度
cap.set(4, height)  #设置长度
cap.set(5, fps)  
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


cv_file = cv2.FileStorage("./charuco_camera_calibration.yaml", cv2.FILE_STORAGE_READ)

# Note : we also have to specify the type
# to retrieve otherwise we only get a 'None'
# FileNode object back instead of a matrix
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

p3_x,p3_y = 0,0
left = int((width-img_size)/2)
right = int((width-img_size)/2+img_size)
up = int((height-img_size)/2)
down = int((height-img_size)/2+img_size)
###------------------ ARUCO TRACKER ---------------------------
while (True):
    ret, frame = cap.read()
    frame = frame[up:down,left:right]
    frame_copy = frame.copy()
    #if ret returns false, there is likely a problem with the webcam/camera.
    #In that case uncomment the below line, which will replace the empty frame 
    #with a test image used in the opencv docs for aruco at https://www.docs.opencv.org/4.5.3/singlemarkersoriginal.jpg
    # frame = cv2.imread('./images/test image.jpg') 

    # operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((3, 3), 0.025, 0.018, aruco_dict)
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

        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, mark_size, mtx, dist)
        #(rvec-tvec).any() # get rid of that nasty numpy value array error

        # for i in range(0, ids.size):
        #     # draw axis for the aruco markers
        #     cv2.drawFrameAxes(frame_copy, mtx, dist, rvec[i], tvec[i], 0.1)

        # draw a square around the markers
        aruco.drawDetectedMarkers(frame_copy, corners)
        retval,charucoCorners,charucoIds =cv2.aruco.interpolateCornersCharuco(corners,ids,frame,board)
        if retval:
            retval, rvec_, tvec_ = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, mtx, dist, None, None)

            # If pose estimation is successful, draw the axis
            if retval:
                org_point = np.array([[0,0,0]],dtype = np.float64)
                p1, _ = cv2.projectPoints(org_point, rvec_, tvec_, mtx, dist)
                p1_x = int(p1[0][0][0])
                p1_y = int(p1[0][0][1])
                cv2.circle(frame_copy,(int(charucoCorners[0][0][0]),int(charucoCorners[0][0][1])),10,(0,0,255))
                cv2.circle(frame_copy,(int(charucoCorners[1][0][0]),int(charucoCorners[1][0][1])),10,(255,0,255))
                cv2.circle(frame_copy,(int(charucoCorners[2][0][0]),int(charucoCorners[2][0][1])),10,(0,255,0))
                cv2.circle(frame_copy,(int(charucoCorners[3][0][0]),int(charucoCorners[3][0][1])),10,(0,0,0))
                total = np.sqrt(sum(np.power((charucoCorners[0][0] - charucoCorners[1][0]), 2)))+np.sqrt(sum(np.power((charucoCorners[1][0] - charucoCorners[3][0]), 2)))+np.sqrt(sum(np.power((charucoCorners[3][0] - charucoCorners[2][0]), 2)))+np.sqrt(sum(np.power((charucoCorners[2][0] - charucoCorners[0][0]), 2)))
                ratio = 25/(total/4)
                print(str(ratio))

                cv2.circle(frame_copy,(p1_x,p1_y),10,(0,0,255))

                cv2.drawFrameAxes(frame_copy, mtx, dist, rvec_, tvec_, length=0.05, thickness=2)
        # code to show ids of the marker found

                # text2 = "Ds: " + "X:" + str(round(tvec_[0][0],4)) + "     " + "Y:" + str(round(tvec_[1][0],4)) + "     " + "Z:" + str(round(tvec_[2][0],4))
                # text3 = "Rot: " + "X:" + str(round(rvec_[0][0]*180/np.pi,4)) + "     " + "Y:" + str(round(rvec_[1][0]*180/np.pi,4)) + "     " + "Z:" + str(round(rvec_[2][0]*180/np.pi,4))
                # cv2.putText(frame_copy, text2, (0,100+120*i), font, 0.5, (0,0,255),2,cv2.LINE_AA)
                # cv2.putText(frame_copy, text3, (0,150+120*i), font, 0.5, (0,0,255),2,cv2.LINE_AA)

                P_corn = np.array([[-0.0597],[0],[0]],dtype=np.float64)

                alpha = rvec_[0][0]
                beta = rvec_[1][0]
                gamma = rvec_[2][0]
                T_mask2cam = np.array([[tvec_[0][0]],[tvec_[1][0]],[tvec_[2][0]]],np.float64)
                R_mask2cam = np.zeros((3, 3), dtype=np.float64)
                cv2.Rodrigues(rvec_,R_mask2cam)
                # R_mask2cam = np.array([[1,0, 0],
                #     [0, 1, 0],
                #     [0, 0, 1]
                #     ])
                P_c = np.dot(R_mask2cam,P_corn)+T_mask2cam

                p2, _ = cv2.projectPoints(org_point, rvec_, P_c, mtx, dist)
                p2_x = int(p2[0][0][0])
                p2_y = int(p2[0][0][1])
                cv2.circle(frame_copy,(p2_x,p2_y),10,(0,0,255))
                cv2.drawFrameAxes(frame_copy, mtx, dist, rvec_, P_c, length=0.05, thickness=2)

                
                def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
                    global p3_x,p3_y
                    if event == cv2.EVENT_LBUTTONDOWN:
                        xy = "[%d,%d]" % (x, y)
                        p3_x=x
                        p3_y=y
                        cv2.circle(frame_copy, (x, y), 3, (255, 0, 0), thickness=-1)
                        cv2.putText(frame_copy, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                                    1.0, (107, 73, 251), thickness=1)	#如果不想在图片上显示坐标可以注释该行
                        cv2.imshow("frame", frame_copy)
                        
                        
                # cv2.setMouseCallback('frame', on_EVENT_LBUTTONDOWN)

                p1=np.float32([[p1_x,p1_y,1]]).reshape(3,-1)
                p2=np.float32([[p2_x,p2_y,1]]).reshape(3,-1)
                p3=np.float32([[p3_x,p3_y,1]]).reshape(3,-1)

                Rc=np.linalg.inv(R_mask2cam)
                Tc=T_mask2cam
                p1_p = np.dot(Rc,(p1-Tc))
                p2_p = np.dot(Rc,(p2-Tc))
                p3_p = np.dot(Rc,(p3-Tc))
                ratio = 0.0597/np.sqrt(sum(np.power((p2_p - p1_p), 2)))
        
                Dis = np.sqrt(sum(np.power((p3_p - p1_p), 2)))*ratio
                Dis = round(Dis[0],4)
                cv2.putText(frame_copy, str(Dis), (500, 500), cv2.FONT_HERSHEY_PLAIN,
                                    3.0, (255, 255, 0), thickness=3)


            else:
                # code to show 'No Ids' when no markers are found
                cv2.putText(frame_copy, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

    # display the resulting frame
    frame_copy = cv2.resize(frame_copy, (int(frame_copy.shape[1]/2), int(frame_copy.shape[0]/2)), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('frame',frame_copy)
    k=cv2.waitKey(1)
    if k==27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


