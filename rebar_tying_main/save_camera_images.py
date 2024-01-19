import pyrealsense2 as rs
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

from camera import Camera


if __name__ == '__main__':

    camera = Camera()
    time.sleep(1)

    count = 72

    while True:

        color_frame, aligned_depth_frame = camera.get_frames()

        color_image, depth_image, vertices, vertices_color = camera.process_frames(color_frame, aligned_depth_frame)

        user_input = input("PRESS ENTER TO SAVE IMAGE")

        if user_input == "":
            cv2.imwrite('C:/Users/anazi/Documents/digging/dataset/raw_images/'+'camera_image'+str(count)+'.jpeg', color_image)

        count += 1







