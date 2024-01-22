# Rebar-tying
## Rebar tying with fairino robot and D435 camera
### Requirement
opencv-contrib-python
### About calibration:
The key code is `R,T=cv2.calibrateHandEye(R_all_end_to_base_1,T_all_end_to_base_1,R_all_chess_to_cam_1,T_all_chess_to_cam_1)
`

Pay attention to adjust the input data depend on the camera's relative position (Eye-in-hand or Eye-to-hand)
![1705910268600](https://github.com/123CHENJINHUA/rebar-tying/assets/114796134/69bcbb76-97cd-43c2-8ffd-b781db0fe187)
1. Calibrate the camera with charucoboard.
1. Run data_collection.py and move the robot arm to collect the necessary data.
1. Run hand_eye_calibration.py to get the matrix.

![demo](https://github.com/123CHENJINHUA/rebar-tying/assets/114796134/20705303-de2c-489a-9e65-e8fc458ac666)



