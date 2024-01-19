import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,sys.path[0]+'/../')
from rebar_tying_main.camera import Camera

import torch
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from rebar_tying_main.training import get_model_instance_segmentation, get_transform

import open3d as o3d

from math import *

RT_cam_to_end = np.load('./data/RT.npy')

MIN_SCORE = 0.6
global click_x, click_y
click_x = None
click_y = None

cv_file = cv2.FileStorage("./hand_eye_calibration/charuco_camera_calibration.yaml", cv2.FILE_STORAGE_READ)

# Note : we also have to specify the type
# to retrieve otherwise we only get a 'None'
# FileNode object back instead of a matrix
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

def myRPY2R_robot(x, y, z):
    x=x*np.pi/180
    y=y*np.pi/180
    z=z*np.pi/180

    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz@Ry@Rx
    return R

def get_mask_center(mask):
    "return pixel coordinates of mask center"
    mask_center = np.mean(np.argwhere(mask),axis=0)

    return int(mask_center[1]), int(mask_center[0])

def mask_vertices_colors(mask, color_image, aligned_depth_frame,intrinsics):
    camera = Camera()
    idxes = np.argwhere(mask)
    points = []
    colors = []
    for row in idxes:
        coords = camera.deproject_pixel([row[1],row[0]],aligned_depth_frame,intrinsics)
        points.append(coords)
        colors.append(color_image[row[0],row[1], :])

    return np.asarray(points).astype(np.float32), np.asarray(colors).astype(np.uint8)

def get_surface_normal(mask):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mask[~np.any(mask == 0, axis =1)])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    for i in range(np.asarray(pcd.normals).shape[0]):
        if pcd.normals[i][2] > 0:
            pcd.normals[i][0] = -pcd.normals[i][0]
            pcd.normals[i][1] = -pcd.normals[i][1]
            pcd.normals[i][2] = -pcd.normals[i][2]
    
    normals = np.asarray(pcd.normals)
    return np.sum(normals, axis=0) / normals.shape[0]

#2024/1/5 continue...
def rodrigues_rotation(surface_normal,R_all_end_to_base_1):

    surface_normal = surface_normal/np.linalg.norm(surface_normal)

    # surface_normal = np.array([-0.3,-0.3,-1])
    print("surface_normal",surface_normal)
    print("norm",np.linalg.norm(surface_normal))
    v1 = np.array([0,0,1])
    v2 = -surface_normal
    # R_cam_to_base = np.dot(R_all_end_to_base_1,RT_cam_to_end[:3,:3])
    # v2 = np.dot(np.linalg.inv(R_cam_to_base),np.array([0,0,-1]))


    rotationAxis = np.cross(v1,v2)
    rotationAxis = rotationAxis/np.sqrt(np.sum(rotationAxis*rotationAxis))
    rotationAngle = np.arccos(np.dot(v1,v2)/(np.sqrt(np.sum(v1*v1))*np.sqrt(np.sum(v2*v2))))

    rotationVector = rotationAxis*rotationAngle


    theta = rotationAngle
    r = np.array(rotationAxis).reshape(3,1)

    rx,ry,rz = r[:,0]
    M = np.array([
        [0,-rz,ry],
        [rz,0,-rx],
        [-ry,rx,0]
    ])
    R = np.eye(3)
    R[:3,:3] = np.cos(theta)*np.eye(3)+(1-np.cos(theta)) * r @ r.T + np.sin(theta) * M
    # z=-90*np.pi/180
    # Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    # # R = np.dot(Rz,R)
    # R = np.dot(R,Rz)

    rotationVector,_ = cv2.Rodrigues(R)
    # print('First')
    # print(R)
    return rotationVector

# Function to check if two masks overlap
def masks_overlap(mask1, mask2):
    return torch.any(torch.logical_and(mask1, mask2))

def mouse_callback(event,x,y,flags,param):
    global click_x, click_y
    if event == cv2.EVENT_LBUTTONDBLCLK:
        click_x = x
        click_y = y
        
def VisionProcess(robot_point):

    R_all_end_to_base_1 = myRPY2R_robot(robot_point[3],robot_point[4],robot_point[5])
    camera = Camera()
    time.sleep(1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    model = get_model_instance_segmentation(num_classes)

    model.load_state_dict(torch.load('C:/Users/JH CHEN\Desktop/robot_arm/fairino-python-sdk-v2.0.0/windows/libfairino/rebar_tying_main/cross.pt', map_location=device))
    model.eval().to(device)

    eval_transform = get_transform(train=False)

    total_point = []

    while True:
        color_frame, aligned_depth_frame = camera.get_frames()

        intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        


        color_image, depth_image, vertices, vertices_color = camera.process_frames(color_frame, aligned_depth_frame)
        image = torch.as_tensor(color_image).permute(2,0,1)

        with torch.no_grad():
            x = eval_transform(image)
            x = x[:3, ...].to(device)
            predictions = model([x, ])
            pred = predictions[0]

        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
        image = image[:3, ...]
        pred_labels = [f"cross: {score:.3f}" for label, score in zip(pred["labels"][pred['scores']>MIN_SCORE], pred["scores"][pred['scores']>MIN_SCORE])]
        pred_boxes = pred["boxes"][pred['scores']>MIN_SCORE].long()
        output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

        pred_masks = (pred["masks"][pred['scores']>MIN_SCORE] > 0.7).squeeze(1)
        output_image = draw_segmentation_masks(output_image, pred_masks, alpha=0.5, colors="blue")

        if pred_masks.size(0) == 0:
            cv2.namedWindow('RealSense2',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense2',color_image)
            cv2.waitKey(1)
            continue

         # only keep non-overlapping masks
        unique_masks = []
        unique_masks_ids = []
        unique_nums = 0
        for i, mask in enumerate(pred_masks):
            is_unique = True
            for unique_mask in unique_masks:
                if masks_overlap(mask, unique_mask):
                    is_unique = False
                    break
            if is_unique:
                unique_masks.append(mask)
                unique_masks_ids.append(unique_nums)
                unique_nums+=1

        unique_masks = torch.stack(unique_masks)

        total_normals = []
        for i in range(unique_masks.size(0)):
            selected_mask = unique_masks[i].cpu().numpy()

            #compute surface normal at mask
            vert_mask, color_mask = mask_vertices_colors(selected_mask, color_image, aligned_depth_frame,intrinsics)
            surface_normal = get_surface_normal(vert_mask)
            total_normals.append(surface_normal)
            normal2RotVec = rodrigues_rotation(surface_normal,R_all_end_to_base_1)
            # normal2RotM,_ = cv2.Rodrigues(normal2RotVec)
            # print('next:\n')
            # print(normal2RotM)

            mask_xcoord, mask_ycoord = get_mask_center(selected_mask)
            mask_3Dcoord = camera.deproject_pixel([mask_xcoord,mask_ycoord],aligned_depth_frame,intrinsics)
            # if np.sum(mask_3Dcoord) != 0:
            mask_3Dcoord.extend(normal2RotVec[0])
            mask_3Dcoord.extend(normal2RotVec[1])
            mask_3Dcoord.extend(normal2RotVec[2]) #extend rotation vector
            total_point.append(mask_3Dcoord)
            # print(mask_3Dcoord)
        
        if len(total_point)==0:
            continue
       
        output_image = output_image.permute(1, 2, 0).cpu().numpy()

        num = 0
        for point in total_point:
            tvec = np.array([point[0],point[1],point[2]])
            rvec = np.array([point[3],point[4],point[5]])
            cv2.drawFrameAxes(output_image,mtx,dist,rvec,tvec,length=0.05,thickness=2)
            image_point,_ = cv2.projectPoints(np.array([[0,0,0]],dtype = np.float64),rvec,tvec,mtx,dist)
            image_x = int(image_point[0][0][0])
            image_y = int(image_point[0][0][1])

            output_image = cv2.circle(output_image, (image_x, image_y), radius=10, color=(255, 255, 255), thickness=1)
            text = "["+str(round(total_normals[num][0],4))+" "+str(round(total_normals[num][1],4))+" "+str(round(total_normals[num][2],4))+"]"
            num +=1
            cv2.putText(output_image, text, (image_x+25, image_y+25), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
        

        #out of the predicted masks, we will pick one specified by the user
        user_mask = None
        user_mask_id = None
        global click_x, click_y

        while(1):
            cv2.namedWindow('RealSense2', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback("RealSense2", mouse_callback)
            cv2.imshow('RealSense2', output_image)
            cv2.waitKey(1000)
            
            if None not in (click_x, click_y):

                for i, mask in enumerate(unique_masks):
                    if mask[click_y, click_x]:
                        user_mask = mask
                        user_mask_id = unique_masks_ids[i]
                        break
                    
                if None not in (user_mask, user_mask_id):
                    # display user-selected mask and break the while loop
                    output_image = draw_segmentation_masks(image, user_mask, alpha=0.8, colors="green")
                    cv2.imshow('RealSense2', output_image.permute(1, 2, 0).cpu().numpy())
                    cv2.waitKey(1)
                    time.sleep(1)
                    break
                else:
                    # reset click parameters and continue
                    click_x = None
                    click_y = None

        print("click_xcoord: ", click_x)
        print("click_ycoord: ", click_y)
        print("selected normal: ", total_normals[user_mask_id])
        print("return: ", total_point[user_mask_id])

        return total_point[user_mask_id]



if __name__ == '__main__':

   point_indx = VisionProcess(np.array([0,0,0,0,0,0]))

    
