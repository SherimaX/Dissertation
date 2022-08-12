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
        elif direction.get() == "down":
            move_down()
        elif direction.get() == "left":
            move_left()
        elif direction.get() == "right":
            move_right()
        # elif direction.get() == "random":
        #     move_random()
        else:
            no_op()

        root.after(100, move)  # reschedule event in 2 seconds
        num_moves.set(num_moves.get() + 1)
        if num_moves.get() > 1000:
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

