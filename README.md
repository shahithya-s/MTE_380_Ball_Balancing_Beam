# MTE_380_Ball_Balancing_Beam

1. Run `simple_cal.py` to calibrate motor and camera -> outputs a config.json file for the camera placement, ball detection and servo motor angle configuration
2. Run `ball detection.py` to verify calibration and make minor adjustments
3. Run `basic_controller.py` to run a PID controller that aims to balance the ball at a designated setpoint. 