import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt


class Camera:

    def __init__(self):

        print("init camera")
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))


        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()


        # Create an align object
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        # self.intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        return color_frame, aligned_depth_frame
    

    def process_frames(self, color_frame, aligned_depth_frame):

        depth_image = np.asanyarray(aligned_depth_frame.get_data())*self.depth_scale
        color_image = np.asanyarray(color_frame.get_data())

        pointcloud = rs.pointcloud()
        points = pointcloud.calculate(aligned_depth_frame)
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(640*480,3)# xyz
        vertices_color = color_image.reshape(640*480,3)

        return color_image, depth_image, vertices, vertices_color
    

    def deproject_pixel(self, pixel_coords, aligned_depth_frame,intrinsics):

        depth = aligned_depth_frame.get_distance(pixel_coords[0], pixel_coords[1])
        return rs.rs2_deproject_pixel_to_point(intrinsics, pixel_coords, depth)


    def get_heightmap(self, vertices, vertices_color, cam_pose, workspace_limits, heightmap_resolution):

        # Compute heightmap size
        heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)

        # Transform 3D point cloud from camera coordinates to robot coordinates
        surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(vertices)) + np.tile(cam_pose[0:3,3:],(1,vertices.shape[0])))

        # Sort surface points by z value
        sort_z_ind = np.argsort(surface_pts[:,2])
        surface_pts = surface_pts[sort_z_ind]
        color_pts = vertices_color[sort_z_ind]

        # Filter out surface points outside heightmap boundaries
        heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1]), surface_pts[:,2] < workspace_limits[2][1])
        surface_pts = surface_pts[heightmap_valid_ind]
        color_pts = color_pts[heightmap_valid_ind]

        # Create orthographic top-down-view RGB-D heightmaps
        color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
        depth_heightmap = np.zeros(heightmap_size)
        heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
        heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)
        color_heightmap_r[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[0]]
        color_heightmap_g[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[1]]
        color_heightmap_b[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[2]]
        color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
        depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
        z_bottom = workspace_limits[2][0]
        depth_heightmap = depth_heightmap - z_bottom
        depth_heightmap[depth_heightmap < 0] = 0
        depth_heightmap[depth_heightmap == -z_bottom] = np.nan

        return color_heightmap, depth_heightmap




if __name__ == '__main__':

    camera = Camera()
    time.sleep(1)

    while True:

        color_frame, aligned_depth_frame = camera.get_frames()

        color_image, depth_image, vertices, vertices_color = camera.process_frames(color_frame, aligned_depth_frame)


        camera_pose = np.identity(4)
        workspace_limits = np.asarray([[-0.5, 0.5], [-0.3, 0.5], [0, 1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        heightmap_resolution = 0.002

        color_heightmap, depth_heightmap = camera.get_heightmap(vertices, vertices_color, camera_pose, workspace_limits, heightmap_resolution)

        cv2.namedWindow('RealSense1', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense1', color_image)
        cv2.waitKey(1)

        cv2.namedWindow('RealSense2', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense2', color_heightmap)
        cv2.waitKey(1)





