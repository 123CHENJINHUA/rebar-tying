import cv2
import pyrealsense2 as rs
import time
import numpy as np

# 创建摄像头对象
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)
time.sleep(1)

i=0
while(1):

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    frame = np.asanyarray(color_frame.get_data())

    k=cv2.waitKey(1)
    if k==27:
        break
    elif k==ord('s'):
        # cv2.imwrite('./photo/L'+str(i)+'.jpg',frame)
        cv2.imwrite('./hand_eye_calibration/calibration_12192/'+str(i)+'.png',frame)
        i+=1
        print("save!",i)
    # cv2.imshow("capture1", frame)

    cv2.imshow("capture2", frame)

pipeline.stop()
cv2.destroyAllWindows()