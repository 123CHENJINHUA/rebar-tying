import time
import threading
import numpy as np
import cv2
import pyrealsense2 as rs

# 定义计数子线程
def timer(interval):
    i=0
    while True:  # 无限计时
        time.sleep(interval)
        if(img_temp is not None):
            cv2.imwrite('./calibration_1214/'+str(i)+'.png',img_temp)
            i+=1
            print("save!",i)


"""---------------- 主线程(main) -------------------"""
if __name__ == '__main__':
    img_temp = None  # 占位用的，目的是提升frame的作用域
    interval = 0.1  # 时间间隔(s)

    # 开启一个子线程
    """
        Note:
            1. daemon=
                1. True: 主线程结束，子线程也结束
                2. False：主线程结束，子线程不结束（主线程需等待子线程结束后再结束）
            2. args=(interval, )中的 逗号 不能省略（因为传入的必须是一个tuple）
    """
    # 1. 定义线程
    th1 = threading.Thread(target=timer, daemon=True, args=(interval,))
    # 2. 开启线程
    th1.start()

    # 创建摄像头对象
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    time.sleep(1)

    while True:
        
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())
        # 赋值变量
        img_temp = frame  # 将frame赋给img_temp
        cv2.imshow("capture2", frame)
        k=cv2.waitKey(1)
        if k==27:
            break
        

    # 释放资源
    pipeline.stop()
    cv2.destroyAllWindows()
