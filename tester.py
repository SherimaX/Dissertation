#!/usr/bin/env python3
"""
This script makes MiRo look for a blue ball and kick it

The code was tested for Python 2 and 3
For Python 2 you might need to change the shebang line to
#!/usr/bin/env python
"""
# Imports
##########################
import os
from math import radians  # This is used to reset the head pose
import numpy as np  # Numerical Analysis library
import cv2  # Computer Vision library
import time

import rospy  # ROS Python interface
from sensor_msgs.msg import CompressedImage  # ROS CompressedImage message
from sensor_msgs.msg import JointState  # ROS joints state message
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
from geometry_msgs.msg import TwistStamped  # ROS cmd_vel (velocity control) message
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState


import miro2 as miro  # Import MiRo Developer Kit library

try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel  # Python 2
##########################


class MiRoClient:
    """
    Script settings below
    """
    TICK = 0.02  # This is the update interval for the main control loop in secs
    CAM_FREQ = 1  # Number of ticks before camera gets a new frame, increase in case of network lag
    SLOW = 0.1  # Radial speed when turning on the spot (rad/s)
    FAST = 0.4  # Linear speed when kicking the ball (m/s)
    DEBUG = False  # Set to True to enable debug views of the cameras
    ##NOTE The following option is relevant in MiRoCODE
    NODE_EXISTS = False  # Disables (True) / Enables (False) rospy.init_node

    def __init__(self):
        # Initialise a new ROS node to communicate with MiRo
        if not self.NODE_EXISTS:
            rospy.init_node("kick_blue_ball", anonymous=True)
        # Give it some time to make sure everything is initialised
        rospy.sleep(2.0)
        # Initialise CV Bridge
        # Individual robot name acts as ROS topic prefix
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")

        # Create a new publisher to send velocity commands to the robot
        self.vel_pub = rospy.Publisher(
            topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0
        )

    def drive(self, speed_l=0.1, speed_r=0.1):  # (m/sec, m/sec)
        """
        Wrapper to simplify driving MiRo by converting wheel speeds to cmd_vel
        """
        # Prepare an empty velocity command message
        msg_cmd_vel = TwistStamped()

        # Desired wheel speed (m/sec)
        wheel_speed = [speed_l, speed_r]

        # Convert wheel speed to command velocity (m/sec, Rad/sec)
        (dr, dtheta) = wheel_speed2cmd_vel(wheel_speed)

        # Update the message with the desired speed
        msg_cmd_vel.twist.linear.x = dr
        msg_cmd_vel.twist.angular.z = dtheta

        # Publish message to control/cmd_vel topic
        self.vel_pub.publish(msg_cmd_vel)


    
    
# This condition fires when the script is called directly
def move(direction):
    state_msg = ModelState()
    state_msg.model_name = "miro_model"
  
    get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    resp = get_model_state("miro","")
    
    x = resp.pose.position.x
    y = resp.pose.position.y
    
    
    if direction == "up":
        x += 0.6
    if direction == "down":
        x -= 0.6
    if direction == "left":
        y += 0.6
    if direction == "right":
        y -= 0.6
        
    state_msg.pose.position.x = x
    state_msg.pose.position.y = y
    rospy.wait_for_service('/gazebo/set_model_state')
        
    
    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    resp = set_state(state_msg)


if __name__ == "__main__":

    
    move("up")
    move("up")
    move("up")
    move("up")
    move("left")
    move("left")
