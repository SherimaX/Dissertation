#!/usr/bin/env python
        
        
          


        
        
          


        
        
          
# Imports
        
        
          
##########################
        
        
          
import rospy  # ROS Python interface
        
        
          
import math
        
        
          
from gazebo_msgs.msg import ModelState
        
        
          
from gazebo_msgs.srv import SetModelState, GetModelState
        
        
          


        
        
          
import miro2 as miro  # Import MiRo Developer Kit library
        
        
          
##########################
        
        
          


        
        
          


        
        
          
class Gazebo_Object():
        
        
          


        
        
          
        def __init__(self):
        
        
          
            self.name = "miro_toy_ball"
        
        
          
            self.x0 = 0.6
            self.y0 = -0.3
        
        
          
            self.zo = 0.1
        
        
          
            self.amp = 0.9
        
        
          
            self.period = 1.0
        
        
          


        
        
          
if __name__ == "__main__":
        
        
          
    rospy.init_node('ball_mover')

    OB = Gazebo_Object()
    state_msg = ModelState()

    state_msg.model_name = OB.name
  
    state_msg.pose.position.x = OB.x0
     
    state_msg.pose.position.z = OB.zo
     
    state_msg.pose.orientation.x = 0
     
    state_msg.pose.orientation.y = 0
 
    state_msg.pose.orientation.z = 0
  
    state_msg.pose.orientation.w = 0
        
        
    rospy.wait_for_service('/gazebo/set_model_state')
        
        
          
    start_time = rospy.get_rostime()
        
    current_time = rospy.get_rostime()
        
        
          
    while not rospy.core.is_shutdown():
        
        
          
        t = math.pi * current_time.nsecs / (10**9)
        
        
          
        state_msg.pose.position.y = OB.y0 + OB.amp*math.sin(t)
        
        

        
          
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( state_msg )
    
    
      
        #print(current_time.nsecs/10**6, t)
    
        
          
        if rospy.get_rostime().secs > current_time.secs:
            pass
                  current_time = rospy.get_rostime()

