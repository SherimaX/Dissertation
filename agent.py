"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|

Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""

import itertools
import time
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter
import random


from moviepy.editor import VideoClip

WORLD_HEIGHT = 9
WORLD_WIDTH = 9
WALL_FRAC = .2
NUM_WINS = 5
NUM_LOSE = 10

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

    def reset_head_pose(self):
        """
        Reset MiRo head to default position, to avoid having to deal with tilted frames
        """
        self.kin_joints = JointState()  # Prepare the empty message
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin_joints.position = [0.0, radians(34.0), 0.0, 0.0]
        t = 0
        while not rospy.core.is_shutdown():  # Check ROS is running
            # Publish state to neck servos for 1 sec
            self.pub_kin.publish(self.kin_joints)
            rospy.sleep(self.TICK)
            t += self.TICK
            if t > 1:
                break

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

    def callback_caml(self, ros_image):  # Left camera
        self.callback_cam(ros_image, 0)

    def callback_camr(self, ros_image):  # Right camera
        self.callback_cam(ros_image, 1)

    def callback_cam(self, ros_image, index):
        """
        Callback function executed upon image arrival
        """
        # Silently(-ish) handle corrupted JPEG frames
        try:
            # Convert compressed ROS image to raw CV image
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            # Convert from OpenCV's default BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Store image as class attribute for further use
            self.input_camera[index] = image
            # Get image dimensions
            self.frame_height, self.frame_width, channels = image.shape
            self.x_centre = self.frame_width / 2.0
            self.y_centre = self.frame_height / 2.0
            # Raise the flag: A new frame is available for processing
            self.new_frame[index] = True
        except CvBridgeError as e:
            # Ignore corrupted frames
            pass

    def detect_ball(self, frame, index):
        """
        Image processing operations, fine-tuned to detect a small,
        toy blue ball in a given frame.
        """
        if frame is None:  # Sanity check
            return

        # Debug window to show the frame
        if self.DEBUG:
            cv2.imshow("camera" + str(index), frame)
            cv2.waitKey(1)

        # Flag this frame as processed, so that it's not reused in case of lag
        self.new_frame[index] = False
        # Get image in HSV (hue, saturation, value) colour format
        im_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Specify target ball colour
        rgb_colour = np.uint8([[[255, 0, 0]]])  # e.g. Blue (Note: BGR)
        # Convert this colour to HSV colour model
        hsv_colour = cv2.cvtColor(rgb_colour, cv2.COLOR_RGB2HSV)

        # Extract colour boundaries for masking image
        # Get the hue value from the numpy array containing target colour
        target_hue = hsv_colour[0, 0][0]
        hsv_lo_end = np.array([target_hue - 20, 70, 70])
        hsv_hi_end = np.array([target_hue + 20, 255, 255])

        # Generate the mask based on the desired hue range
        mask = cv2.inRange(im_hsv, hsv_lo_end, hsv_hi_end)
        mask_on_image = cv2.bitwise_and(im_hsv, im_hsv, mask=mask)

        # Debug window to show the mask
        if self.DEBUG:
            cv2.imshow("mask" + str(index), mask_on_image)
            cv2.waitKey(1)

        # Clean up the image
        seg = mask
        seg = cv2.GaussianBlur(seg, (5, 5), 0)
        seg = cv2.erode(seg, None, iterations=2)
        seg = cv2.dilate(seg, None, iterations=2)

        # Fine-tune parameters
        ball_detect_min_dist_between_cens = 40  # Empirical
        canny_high_thresh = 10  # Empirical
        ball_detect_sensitivity = 10  # Lower detects more circles, so it's a trade-off
        ball_detect_min_radius = 5  # Arbitrary, empirical
        ball_detect_max_radius = 50  # Arbitrary, empirical

        # Find circles using OpenCV routine
        # This function returns a list of circles, with their x, y and r values
        circles = cv2.HoughCircles(
            seg,
            cv2.HOUGH_GRADIENT,
            1,
            ball_detect_min_dist_between_cens,
            param1=canny_high_thresh,
            param2=ball_detect_sensitivity,
            minRadius=ball_detect_min_radius,
            maxRadius=ball_detect_max_radius,
        )

        if circles is None:
            # If no circles were found, just display the original image
            return

        # Get the largest circle
        max_circle = None
        self.max_rad = 0
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            if c[2] > self.max_rad:
                self.max_rad = c[2]
                max_circle = c
        # This shouldn't happen, but you never know...
        if max_circle is None:
            return

        # Append detected circle and its centre to the frame
        cv2.circle(frame, (max_circle[0], max_circle[1]), max_circle[2], (0, 255, 0), 2)
        cv2.circle(frame, (max_circle[0], max_circle[1]), 2, (0, 0, 255), 3)
        if self.DEBUG:
            cv2.imshow("circles" + str(index), frame)
            cv2.waitKey(1)

        # Normalise values to: x,y = [-0.5, 0.5], r = [0, 1]
        max_circle = np.array(max_circle).astype("float32")
        max_circle[0] -= self.x_centre
        max_circle[0] /= self.frame_width
        max_circle[1] -= self.y_centre
        max_circle[1] /= self.frame_width
        max_circle[1] *= -1.0
        max_circle[2] /= self.frame_width

        # Return a list values [x, y, r] for the largest circle
        return [max_circle[0], max_circle[1], max_circle[2]]

    def look_for_ball(self):
        """
        [1 of 3] Rotate MiRo if it doesn't see a ball in its current
        position, until it sees one.
        """
        if self.just_switched:  # Print once
            print("MiRo is looking for the ball...")
            self.just_switched = False
        for index in range(2):  # For each camera (0 = left, 1 = right)
            # Skip if there's no new image, in case the network is choking
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            # Run the detect ball procedure
            self.ball[index] = self.detect_ball(image, index)
        # If no ball has been detected
        if not self.ball[0] and not self.ball[1]:
            self.drive(self.SLOW, -self.SLOW)
        else:
            self.status_code = 2  # Switch to the second action
            self.just_switched = True

    def lock_onto_ball(self, error=25):
        """
        [2 of 3] Once a ball has been detected, turn MiRo to face it
        """
        if self.just_switched:  # Print once
            print("MiRo is locking on to the ball")
            self.just_switched = False
        for index in range(2):  # For each camera (0 = left, 1 = right)
            # Skip if there's no new image, in case the network is choking
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            # Run the detect ball procedure
            self.ball[index] = self.detect_ball(image, index)
        # If only the right camera sees the ball, rotate clockwise
        if not self.ball[0] and self.ball[1]:
            self.drive(self.SLOW, -self.SLOW)
        # Conversely, rotate counterclockwise
        elif self.ball[0] and not self.ball[1]:
            self.drive(-self.SLOW, self.SLOW)
        # Make the MiRo face the ball if it's visible with both cameras
        elif self.ball[0] and self.ball[1]:
            error = 0.05  # 5% of image width
            # Use the normalised values
            left_x = self.ball[0][0]  # should be in range [0.0, 0.5]
            right_x = self.ball[1][0]  # should be in range [-0.5, 0.0]
            rotation_speed = 0.03  # Turn even slower now
            if abs(left_x) - abs(right_x) > error:
                self.drive(rotation_speed, -rotation_speed)  # turn clockwise
            elif abs(left_x) - abs(right_x) < -error:
                self.drive(-rotation_speed, rotation_speed)  # turn counterclockwise
            else:
                # Successfully turned to face the ball
                self.status_code = 3  # Switch to the third action
                self.just_switched = True
                self.bookmark = self.counter
        # Otherwise, the ball is lost :-(
        else:
            self.status_code = 0  # Go back to square 1...
            print("MiRo has lost the ball...")
            self.just_switched = True

    # GOAAAL
    def kick(self):
        """
        [3 of 3] Once MiRO is in position, this function should drive the MiRo
        forward until it kicks the ball!
        """
        if self.just_switched:
            print("MiRo is kicking the ball!")
            self.just_switched = False
        if self.counter <= self.bookmark + 2 / self.TICK:
            self.drive(self.FAST, self.FAST)
        else:
            self.status_code = 0  # Back to the default state after the kick
            self.just_switched = True

    def __init__(self):
        # Initialise a new ROS node to communicate with MiRo
        if not self.NODE_EXISTS:
            rospy.init_node("kick_blue_ball", anonymous=True)
        # Give it some time to make sure everything is initialised
        rospy.sleep(2.0)
        # Initialise CV Bridge
        self.image_converter = CvBridge()
        # Individual robot name acts as ROS topic prefix
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
        # Create two new subscribers to recieve camera images with attached callbacks
        self.sub_caml = rospy.Subscriber(
            topic_base_name + "/sensors/caml/compressed",
            CompressedImage,
            self.callback_caml,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.sub_camr = rospy.Subscriber(
            topic_base_name + "/sensors/camr/compressed",
            CompressedImage,
            self.callback_camr,
            queue_size=1,
            tcp_nodelay=True,
        )
        # Create a new publisher to send velocity commands to the robot
        self.vel_pub = rospy.Publisher(
            topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        # Create a new publisher to move the robot head
        self.pub_kin = rospy.Publisher(
            topic_base_name + "/control/kinematic_joints", JointState, queue_size=0
        )
        # Create handle to store images
        self.input_camera = [None, None]
        # New frame notification
        self.new_frame = [False, False]
        # Create variable to store a list of ball's x, y, and r values for each camera
        self.ball = [None, None]
        # Set the default frame width (gets updated on reciecing an image)
        self.frame_width = 640
        # Action selector to reduce duplicate printing to the terminal
        self.just_switched = True
        # Bookmark
        self.bookmark = 0
        # Move the head to default pose
        self.reset_head_pose()

    def loop(self):
        """
        Main control loop
        """
        print("MiRo plays ball, press CTRL+C to halt...")
        # Main control loop iteration counter
        self.counter = 0
        # This switch loops through MiRo behaviours:
        # Find ball, lock on to the ball and kick ball
        self.status_code = 0
        while not rospy.core.is_shutdown():

            # Step 1. Find ball
            if self.status_code == 1:
                # Every once in a while, look for ball
                if self.counter % self.CAM_FREQ == 0:
                    self.look_for_ball()

            # Step 2. Orient towards it
            elif self.status_code == 2:
                self.lock_onto_ball()

            # Step 3. Kick!
            elif self.status_code == 3:
                self.kick()

            # Fall back
            else:
                self.status_code = 1

            # Yield
            self.counter += 1
            rospy.sleep(self.TICK)

def turn_right(robot):
    robot.drive(1,-1)
    time.sleep(0.2)
    robot.drive(1,-1)
    time.sleep(0.5)

def turn_left(robot):
    robot.drive(-1,1)
    time.sleep(0.2)
    robot.drive(-1,1)
    time.sleep(0.5)

def move_forward(robot):
    robot.drive(2,2)
    time.sleep(0.2)
    robot.drive(2,2)
    time.sleep(0.2)
    robot.drive(2,2)
    time.sleep(0.5)

class GridWorld:

    def __init__(self, world_height=3, world_width=4, discount_factor=.5, default_reward=-.5, wall_penalty=-.6,
                 win_reward=10., lose_reward=5., viz=True, patch_side=120, grid_thickness=2, arrow_thickness=3,
                 wall_locs=[[1, 1], [1, 2]], win_locs=[[0, 3]], lose_locs=[[1, 3]], start_loc=[0, 0],
                 reset_prob=.2):
        self.world = [np.ones([world_height, world_width]) * default_reward,
                      np.ones([world_height, world_width]) * default_reward]
        self.reset_prob = reset_prob
        self.world_height = world_height
        self.world_width = world_width
        self.wall_penalty = wall_penalty
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.default_reward = default_reward
        self.discount_factor = discount_factor
        self.patch_side = patch_side
        self.grid_thickness = grid_thickness
        self.arrow_thickness = arrow_thickness
        self.wall_locs = np.array(wall_locs)
        self.win_locs = np.array(win_locs)
        self.lose_locs = np.array(lose_locs)

        self.status = [1, 1]
        self.depleting_rate = [0.15, 0.15]
        self.feeding_rate = [1, 1]
        self.delta = 10
        self.r = [0,0]

        self.at_terminal_state = [False, False]
        self.auto_reset = True
        self.random_respawn = True
        self.step = [0, 0]
        self.viz_canvas = None
        self.viz = viz
        self.path_color = (128, 128, 128)
        self.wall_color = (0, 255, 0)
        self.win_color = [(0, 0, 255), (255, 0, 0)]
        self.lose_color = (255, 0, 0)
        self.world[0][self.wall_locs[:, 0], self.wall_locs[:, 1]] = self.wall_penalty
        self.world[0][self.win_locs[:, 0], self.win_locs[:, 1]] = self.win_reward
        self.world[1][self.wall_locs[:, 0], self.wall_locs[:, 1]] = self.wall_penalty
        self.world[1][self.lose_locs[:, 0], self.lose_locs[:, 1]] = self.lose_reward
        spawn_condn = lambda loc: self.world[0][loc[0], loc[1]] == self.default_reward
        self.spawn_locs = np.array([loc for loc in itertools.product(np.arange(self.world_height),
                                                                     np.arange(self.world_width))
                                    if spawn_condn(loc)])
        self.start_state = [np.array(start_loc), np.array(start_loc)]
        self.bot_rc = None
        self.reset()
        self.actions = [self.up, self.left, self.right, self.down, self.noop]
        self.action_labels = ['UP', 'LEFT', 'RIGHT', 'DOWN', 'NOOP']
        self.q_values = [
            np.ones([self.world[0].shape[0], self.world[0].shape[1], len(self.actions)]) * 1. / len(self.actions),
            np.ones([self.world[0].shape[0], self.world[0].shape[1], len(self.actions)]) * 1. / len(self.actions)]
        if self.viz:
            self.init_grid_canvas()
            self.video_out_fpath = 'shm_dqn_gridsolver.mp4'
            self.clip = VideoClip(self.make_frame, duration=15)

    def make_frame(self, t):

        self.action()
        return self.viz_canvas

    def check_terminal_state(self, reward_index):
        if self.world[reward_index][self.bot_rc[reward_index][0], self.bot_rc[reward_index][1]] == self.lose_reward:
            self.at_terminal_state[reward_index] = True
            if self.auto_reset:
                self.reset()

    def reset(self):
        # print('Resetting')
        if not self.random_respawn:
            self.bot_rc = self.start_state.copy()
        else:
            self.bot_rc = [self.spawn_locs[np.random.choice(np.arange(len(self.spawn_locs)))].copy(),
                           self.spawn_locs[np.random.choice(np.arange(len(self.spawn_locs)))].copy()]
        self.at_terminal_state = [False, False]

    def up(self, i):
        action_idx = 0
        # print(self.action_labels[action_idx])
        new_r = self.bot_rc[i][0] - 1
        if new_r < 0 or self.world[i][new_r, self.bot_rc[i][1]] == self.wall_penalty:
            return self.wall_penalty, action_idx
        self.bot_rc[i][0] = new_r
        reward = self.world[i][self.bot_rc[i][0], self.bot_rc[i][1]]
        self.check_terminal_state(i)
        return reward, action_idx

    def left(self, i):
        action_idx = 1
        # print(self.action_labels[action_idx])
        new_c = self.bot_rc[i][1] - 1
        if new_c < 0 or self.world[i][self.bot_rc[i][0], new_c] == self.wall_penalty:
            return self.wall_penalty, action_idx
        self.bot_rc[i][1] = new_c
        reward = self.world[i][self.bot_rc[i][0], self.bot_rc[i][1]]
        self.check_terminal_state(i)
        return reward, action_idx

    def right(self, i):
        action_idx = 2
        # print(self.action_labels[action_idx])
        new_c = self.bot_rc[i][1] + 1
        if new_c >= self.world[i].shape[1] or self.world[0][self.bot_rc[i][0], new_c] == self.wall_penalty:
            return self.wall_penalty, action_idx
        self.bot_rc[i][1] = new_c
        reward = self.world[i][self.bot_rc[i][0], self.bot_rc[i][1]]
        self.check_terminal_state(i)
        return reward, action_idx

    def down(self, i):
        action_idx = 3
        # print(self.action_labels[action_idx])
        new_r = self.bot_rc[i][0] + 1
        if new_r >= self.world[i].shape[0] or self.world[0][new_r, self.bot_rc[i][1]] == self.wall_penalty:
            return self.wall_penalty, action_idx
        self.bot_rc[i][0] = new_r
        reward = self.world[i][self.bot_rc[i][0], self.bot_rc[i][1]]
        self.check_terminal_state(i)
        return reward, action_idx

    def noop(self, i):
        action_idx = 4
        # print(self.action_labels[action_idx])
        reward = self.world[i][self.bot_rc[i][0], self.bot_rc[i][1]]
        self.check_terminal_state(i)
        return reward, action_idx

    def qvals2probs(self, q_vals, epsilon=1e-4):
        action_probs = q_vals - q_vals.min() + epsilon
        action_probs = action_probs / action_probs.sum()
        return action_probs

    def action(self):
        # print('================ ACTION =================')
        if self.at_terminal_state[0] and self.at_terminal_state[1]:
            exit()
        # print('Start position:', self.bot_rc)

        for i in [0, 1]:
            start_bot_rc = self.bot_rc[i][0], self.bot_rc[i][1]
            q_vals = self.q_values[i][self.bot_rc[i][0], self.bot_rc[i][1]]
            action_probs = self.qvals2probs(q_vals)
            reward, action_idx = np.random.choice(self.actions, p=action_probs)(i)
            # print('End position:', self.bot_rc)
            # print('Reward:', reward)

            # a = -self.a1 * q1 + self.b1 * (1.0 - q1) * np.exp(-self.sigma * (rho - self.rho1) ** 2) * (1.0 + input_u)

            alpha = np.exp(-self.step[i] / 10e9)
            self.step[i] += 1
            qv = (1 - alpha) * q_vals[action_idx] + alpha * (reward + self.discount_factor
                                                             * self.q_values[i][
                                                                 self.bot_rc[i][0], self.bot_rc[i][1]].max())

            self.q_values[i][start_bot_rc[0], start_bot_rc[1], action_idx] = qv
            if self.viz:
                self.update_viz(start_bot_rc[0], start_bot_rc[1])

    def update_viz(self, i, j):
        starty = i * (self.patch_side + self.grid_thickness)
        endy = starty + self.patch_side
        startx = j * (self.patch_side + self.grid_thickness)
        endx = startx + self.patch_side
        patch = np.zeros([self.patch_side, self.patch_side, 3]).astype(np.uint8)

        if self.world[0][i, j] == self.win_reward:
            patch[:, :, :] = self.win_color[0]
        elif self.world[1][i, j] == self.lose_reward:
            patch[:, :, :] = self.win_color[1]
        elif self.world[0][i, j] == self.wall_penalty:
            patch[:, :, :] = self.wall_color
        else:
            patch[:, :, :] = self.path_color

        if self.world[1][i, j] == self.default_reward:
            arrow_canvas = np.zeros_like(patch)
            for reward_index in [0, 1]:
                action_probs = self.qvals2probs(self.q_values[reward_index][i, j])
                x_component = action_probs[2] - action_probs[1]
                y_component = action_probs[0] - action_probs[3]
                magnitude = 1. - action_probs[-1]
                s = self.patch_side // 2
                x_patch = int(s * x_component)
                y_patch = int(s * y_component)
                vx = s + x_patch
                vy = s - y_patch
                cv2.arrowedLine(arrow_canvas, (s, s), (vx, vy), self.win_color[reward_index],
                                thickness=self.arrow_thickness,
                                tipLength=0.5)
                gridbox = (magnitude * arrow_canvas + (1 - magnitude) * patch).astype(np.uint8)
                self.viz_canvas[starty:endy, startx:endx] = gridbox
        else:
            self.viz_canvas[starty:endy, startx:endx] = patch

    def init_grid_canvas(self):
        org_h, org_w = self.world_height, self.world_width
        viz_w = (self.patch_side * org_w) + (self.grid_thickness * (org_w - 1))
        viz_h = (self.patch_side * org_h) + (self.grid_thickness * (org_h - 1))
        self.viz_canvas = np.zeros([viz_h, viz_w, 3]).astype(np.uint8)
        for i in range(org_h):
            for j in range(org_w):
                self.update_viz(i, j)

    def solve(self):
        if not self.viz:
            while True:
                self.action()
        else:
            self.clip.write_videofile(self.video_out_fpath, fps=60)

        return self.q_values

    def showAgentMap(self, status=[10, 10], agent_loc=[7, 4]):
        agentMap = self.viz_canvas.copy()
        eta1, eta2 = motivation.step(self.r[0], self.r[1])
        print(eta1)
        print(eta2)
        print(self.r[0])
        print(self.r[1])
        i = agent_loc[0]
        j = agent_loc[1]

        starty = i * (self.patch_side + self.grid_thickness)
        endy = starty + self.patch_side
        startx = j * (self.patch_side + self.grid_thickness)
        endx = startx + self.patch_side
        patch = np.zeros([self.patch_side, self.patch_side, 3]).astype(np.uint8)

        direction = "up"
        if self.world[0][i, j] == self.default_reward or self.world[1][i, j] == self.default_reward:
            arrow_canvas = np.zeros_like(patch)

            action_probs_1 = self.qvals2probs(self.q_values[0][i, j])
            action_probs_2 = self.qvals2probs(self.q_values[1][i, j])

            x_1_component = action_probs_1[2] - action_probs_1[1]
            x_2_component = action_probs_2[2] - action_probs_2[1]

            y_1_component = action_probs_1[0] - action_probs_1[3]
            y_2_component = action_probs_2[0] - action_probs_2[3]

            # x_component = (x_1_component * status[0] + x_2_component * status[1]) / 2
            # y_component = (y_1_component * status[0] + y_2_component * status[1]) / 2

            x_component = (x_1_component * eta1 + x_2_component * eta2)
            y_component = (y_1_component * eta1 + y_2_component * eta2)

            # self.eta1 = np.exp(-b_sigma * (self.X[2] - self.rho1) ** 2)
            # self.eta2 = np.exp(-b_sigma * (self.X[2] - self.rho2) ** 2)


            # x_component = (x_1_component * status[0] + x_2_component * status[1]) / 4.5
            # y_component = (y_1_component * status[0] + y_2_component * status[1]) / 4.5

            # magnitude = np.sqrt(x_component ** 2 + y_component ** 2)

            # if magnitude == 0:
            #     x_component = 0
            #     y_component = 0
            # else:
            #     x_component /= magnitude * 1.5
            #     y_component /= magnitude * 1.5

            # edit here to set wall
            # if x_component > 0 and agent_loc[]

            x = x_component
            y = y_component

            if x_component > 0 and [agent_loc[0], agent_loc[1] + 1] in self.wall_locs.tolist():
                x = 0
            if x_component < 0 and [agent_loc[0], agent_loc[1] - 1] in self.wall_locs.tolist():
                x = 0
            if y_component > 0 and [agent_loc[0] - 1, agent_loc[1]] in self.wall_locs.tolist():
                y = 0
            if y_component < 0 and [agent_loc[0] + 1, agent_loc[1]] in self.wall_locs.tolist():
                y = 0



            if abs(x) > abs(y):
                if x > 0:
                    direction = "right"
                else:
                    direction = "left"
            elif abs(x) < abs(y):
                if y > 0:
                    direction = "up"
                else:
                    direction = "down"
            else:
                direction = "random"



            # if magnitude == 0:
            #     direction = "noop"

            s = self.patch_side // 2
            x_patch = int(s * x_component)
            y_patch = int(s * y_component)
            vx = s + x_patch
            vy = s - y_patch
            cv2.arrowedLine(arrow_canvas, (s, s), (vx, vy), [255, 255, 255],
                            thickness=self.arrow_thickness,
                            tipLength=0.5)
            gridbox = (arrow_canvas).astype(np.uint8)
            agentMap[starty:endy, startx:endx] = gridbox

        cv2.rectangle(agentMap, (startx, starty), (endx, endy), (255, 255, 255), thickness=self.grid_thickness)
        cv2.imwrite("agent.jpg", agentMap)
        cv2.imwrite("map.jpg", self.viz_canvas)
        return agentMap, direction


def read_img(img_name):
    img = Image.open(img_name)
    return img


def gen_world_from_image(img_name):
    img = Image.open(img_name)
    w, h = img.size
    pixel_info = np.array(img.getdata()).reshape((w, h, 3)).transpose()
    pixel_info = np.transpose(pixel_info, axes=(0, 2, 1))
    pixel_info = np.flip(pixel_info, axis=1)

    wall_locs = np.argwhere((pixel_info[1] == 255))
    win_locs = np.argwhere(pixel_info[2] == 255)
    lose_locs = np.argwhere(pixel_info[0] == 255)
    return w, h, wall_locs, win_locs, lose_locs


result = 0

from model import Model

motivation = Model()

def agent():
    global result


    def update_status():
        g.status[0] = float(input_1.get())
        g.status[1] = float(input_2.get())

        show_map()

    def show_map():
        input_1.delete(0, tkinter.END)
        input_2.delete(0, tkinter.END)
        input_1.insert(0, "{:.2f}".format(g.status[0]))
        input_2.insert(0, "{:.2f}".format(g.status[1]))

        plt.clf()
        plt.axis('off')

        needs = [1 - s for s in g.status]

        for i, s in enumerate(g.status):
            if s == 0:
                if i == 0:
                    print("Game Over. Water Depleted")
                    global result
                    result = 1
                    root.destroy()
                if i == 1:
                    print("Game Over. Food Depleted")
                    result = 2
                    root.destroy()

        for i in range(len(needs)):
            if needs[i] < 0:
                needs[i] = 0

        img, d = g.showAgentMap(needs, [x.get(), y.get()])
        direction.set(d)
        plt.imshow(img)

    def deplete(status, depleting_rate):
        # for i in range(len(status)):
        #     if status[i] - depleting_rate[i] >= 0:
        #         status[i] -= depleting_rate[i]
        #     else:
        #         status[i] = 0

        status[0] = motivation.X[0]
        status[1] = motivation.X[1]

        print("food need: {}".format(1 - motivation.X[1]))
        print("water need: {}".format(1 - motivation.X[0]))


        if g.win_locs[0][0] == x.get() and g.win_locs[0][1] == y.get():
            g.r[0] = 5
        else:
            g.r[0] = 0

        if g.lose_locs[0][0] == x.get() and g.lose_locs[0][1] == y.get():
            g.r[1] = 5
        else:
            g.r[1] = 0

    def move_up():
        x.set(x.get() - 1)
        deplete(g.status, g.depleting_rate)
        show_map()

    def move_down():
        x.set(x.get() + 1)
        deplete(g.status, g.depleting_rate)
        show_map()

    def move_left():
        y.set(y.get() - 1)
        deplete(g.status, g.depleting_rate)
        show_map()

    def move_right():
        y.set(y.get() + 1)
        deplete(g.status, g.depleting_rate)
        show_map()

    def move_random():
        r = random.randint(0, 3)
        if r == 0:
            if [x.get() - 1, y.get()] not in g.wall_locs.tolist():
                move_up()
            else:
                move_random()
        elif r == 1:
            if [x.get() + 1, y.get()] not in g.wall_locs.tolist():
                move_down()
            else:
                move_random()
        elif r == 2:
            if [x.get(), y.get() - 1] not in g.wall_locs.tolist():
                move_left()
            else:
                move_random()
        else:
            if [x.get(), y.get() + 1] not in g.wall_locs.tolist():
                move_right()
            else:
                move_random()

    def no_op():
        deplete(g.status, g.depleting_rate)
        show_map()


    robot = MiRoClient()

    root = tkinter.Tk()
    root.title("Agent")

    tkinter.Label(root, text="Water").grid(row=0, column=0)
    tkinter.Label(root, text="Food").grid(row=0, column=1)
    input_1 = tkinter.Entry(root, width=5)
    input_2 = tkinter.Entry(root, width=5)
    input_1.insert(0, "{:.2f}".format(g.status[0]))
    input_2.insert(0, "{:.2f}".format(g.status[1]))
    input_1.grid(row=1, column=0)
    input_2.grid(row=1, column=1)
    tkinter.Button(text='Update', command=update_status).grid(row=2, column=0)

    x = tkinter.IntVar(root, 7)
    y = tkinter.IntVar(root, 4)
    direction = tkinter.StringVar(root)
    num_moves = tkinter.IntVar(root, 0)

    up = tkinter.Button(text='Up', command=move_up)
    down = tkinter.Button(text='Down', command=move_down)
    left = tkinter.Button(text='Left', command=move_left)
    right = tkinter.Button(text='Right', command=move_right)

    up.grid(row=4, column=1)
    down.grid(row=5, column=1)
    left.grid(row=5, column=0)
    right.grid(row=5, column=2)

    root.bind('<Up>', lambda event: move_up())
    root.bind('<Down>', lambda event: move_down())
    root.bind('<Left>', lambda event: move_left())
    root.bind('<Right>', lambda event: move_right())

    # create axes
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.ion()

    img, d = g.showAgentMap()
    direction.set(d)
    plt.imshow(img)
    plt.show()

    def move():
        if direction.get() == "up":
            move_up()
            turn_right(robot)
        elif direction.get() == "down":
            move_down()
            turn_left(robot)
        elif direction.get() == "left":
            move_left()
            turn_left(robot)
        elif direction.get() == "right":
            move_right()
            turn_right(robot)
        # elif direction.get() == "random":
        #     move_random()
        else:
            no_op()

        root.after(100, move)  # reschedule event in 2 seconds
        num_moves.set(num_moves.get() + 1)
        if num_moves.get() > 10:
            print("Survived!")
            global result
            result = 0
            root.destroy()

    root.after(1000, move)
    root.mainloop()

    return result



if __name__ == '__main__':
    width, height, wall_locs, win_locs, lose_locs = gen_world_from_image('t-maze.pnm')

    results = []
    for i in range(100):
        g = GridWorld(world_height=height, world_width=width,
                      wall_locs=wall_locs, win_locs=win_locs, lose_locs=lose_locs, viz=True)

        g.solve()

        results.append(agent())
        print(i)
    print(results)

