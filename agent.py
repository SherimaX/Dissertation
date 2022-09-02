from q_learning import GridWorld
from model import Model
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import tkinter
import random

class Agent():

    def __init__(self):
        self.world = GridWorld()
        self.motivation = Model()
        self.position = [7, 4]

    def showAgentMap(self, agent_loc=[7, 4]):
        agentMap = self.world.viz_canvas.copy()
        eta1, eta2 = self.motivation.step(self.world.r[0], self.world.r[1])

        i = agent_loc[0]
        j = agent_loc[1]

        starty = i * (self.world.patch_side + self.world.grid_thickness)
        endy = starty + self.world.patch_side
        startx = j * (self.world.patch_side + self.world.grid_thickness)
        endx = startx + self.world.patch_side
        patch = np.zeros([self.world.patch_side, self.world.patch_side, 3]).astype(np.uint8)

        direction = "up"

        print(self.world.world[0].shape)
        if self.world.world[0][i, j] == self.world.default_reward or self.world.world[1][i, j] == self.world.default_reward:
            arrow_canvas = np.zeros_like(patch)

            action_probs_1 = self.world.qvals2probs(self.world.q_values[0][i, j])
            action_probs_2 = self.world.qvals2probs(self.world.q_values[1][i, j])

            x_1_component = action_probs_1[2] - action_probs_1[1]
            x_2_component = action_probs_2[2] - action_probs_2[1]

            y_1_component = action_probs_1[0] - action_probs_1[3]
            y_2_component = action_probs_2[0] - action_probs_2[3]



            x_component = (x_1_component * eta1 + x_2_component * eta2)
            y_component = (y_1_component * eta1 + y_2_component * eta2)
            
            
            x = x_component
            y = y_component

            if x_component > 0 and [agent_loc[0], agent_loc[1] + 1] in self.world.wall_locs.tolist():
                x = 0
            if x_component < 0 and [agent_loc[0], agent_loc[1] - 1] in self.world.wall_locs.tolist():
                x = 0
            if y_component > 0 and [agent_loc[0] - 1, agent_loc[1]] in self.world.wall_locs.tolist():
                y = 0
            if y_component < 0 and [agent_loc[0] + 1, agent_loc[1]] in self.world.wall_locs.tolist():
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


            magnitude = abs(x) + abs(y) / 2
            if magnitude < 0.0001:
                direction = "noop"

            s = self.world.patch_side // 2
            x_patch = int(s * x_component)
            y_patch = int(s * y_component)
            vx = s + x_patch
            vy = s - y_patch
            cv2.arrowedLine(arrow_canvas, (s, s), (vx, vy), [255, 255, 255],
                            thickness=self.world.arrow_thickness,
                            tipLength=0.5)
            gridbox = arrow_canvas.astype(np.uint8)
            agentMap[starty:endy, startx:endx] = gridbox

        cv2.rectangle(agentMap, (startx, starty), (endx, endy), (255, 255, 255), thickness=self.world.grid_thickness)
        cv2.imwrite("agent.jpg", agentMap)
        cv2.imwrite("map.jpg", self.world.viz_canvas)
        return agentMap, direction


    result = 0

    def show_map(self, input_1, input_2):
        input_1.delete(0, tkinter.END)
        input_2.delete(0, tkinter.END)
        input_1.insert(0, "{:.2f}".format(self.world.status[0]))
        input_2.insert(0, "{:.2f}".format(self.world.status[1]))

        plt.clf()
        plt.axis('off')

    def update_status(self, world, root, direction, x, y, input_1, input_2):
        world.status[0] = float(input_1.get())
        world.status[1] = float(input_2.get())
        self.show_map(input_1, input_2)


        for i, s in enumerate(self.world.status):
            if s == 0:
                if i == 0:
                    print("Game Over. Food Depleted")
                    global result
                    result = 1
                    root.destroy()
                if i == 1:
                    print("Game Over. Water Depleted")
                    result = 2
                    root.destroy()

        img, d = self.showAgentMap(self.world.status, [x.get(), y.get()])
        direction.set(d)
        plt.imshow(img)
        
    def move_up(self, x, y, input_1, input_2):
        x.set(x.get() - 1)
        self.deplete(x, y)
        self.show_map(input_1, input_2)

    def move_down(self, x, y, input_1, input_2):
        x.set(x.get() + 1)
        self.deplete(x, y)
        self.show_map(input_1, input_2)

    def move_left(self, x, y, input_1, input_2):
        y.set(y.get() - 1)
        self.deplete(x, y)
        self.show_map(input_1, input_2)

    def move_right(self, x, y, input_1, input_2):
        y.set(y.get() + 1)
        self.deplete(x, y)
        self.show_map(input_1, input_2)

    def move_random(self, x, y, input_1, input_2):
        r = random.randint(0, 3)
        if r == 0:
            if [x.get() - 1, y.get()] not in self.world.wall_locs.tolist():
                self.move_up(x, y, input_1, input_2)
            else:
                self.move_random(x, y, input_1, input_2)
        elif r == 1:
            if [x.get() + 1, y.get()] not in self.world.wall_locs.tolist():
                self.move_down(x, y, input_1, input_2)
            else:
                self.move_random(x, y, input_1, input_2)
        elif r == 2:
            if [x.get(), y.get() - 1] not in self.world.wall_locs.tolist():
                self.move_left(x, y, input_1, input_2)
            else:
                self.move_random(x, y, input_1, input_2)
        else:
            if [x.get(), y.get() + 1] not in self.world.wall_locs.tolist():
                self.move_right(x, y, input_1, input_2)
            else:
                self.move_random(x, y, input_1, input_2)

    def no_op(self, x, y, input_1, input_2):
        self.deplete(x, y)
        self.show_map(input_1, input_2)

    def deplete(self, x, y):
        self.world.status[0] = self.motivation.X[0]
        self.world.status[1] = self.motivation.X[1]

        if self.world.lose_locs[0][0] == x.get() and self.world.lose_locs[0][1] == y.get():
            self.world.r[0] = 20
            print("food")
        else:
            self.world.r[0] = 0

        if self.world.win_locs[0][0] == x.get() and self.world.win_locs[0][1] == y.get():
            self.world.r[1] = 20
            print("water")
        else:
            self.world.r[1] = 0

    def agent(self):
        global result


        root = tkinter.Tk()
        root.title("Agent")

        tkinter.Label(root, text="Food").grid(row=0, column=0)
        tkinter.Label(root, text="Water").grid(row=0, column=1)
        input_1 = tkinter.Entry(root, width=5)
        input_2 = tkinter.Entry(root, width=5)
        input_1.insert(0, "{:.2f}".format(self.world.status[0]))
        input_2.insert(0, "{:.2f}".format(self.world.status[1]))
        input_1.grid(row=1, column=0)
        input_2.grid(row=1, column=1)

        x = tkinter.IntVar(root, 7)
        y = tkinter.IntVar(root, 4)
        direction = tkinter.StringVar(root)
        num_moves = tkinter.IntVar(root, 0)


        # create axes
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.ion()

        img, d = self.showAgentMap()
        direction.set(d)
        plt.imshow(img)
        plt.show()

        def move(self, direction, root, x, y, input_1, input_2):
            if direction.get() == "up":
                self.move_up(x, y, input_1, input_2)
            elif direction.get() == "down":
                self.move_down(x, y, input_1, input_2)
            elif direction.get() == "left":
                self.move_left(x, y, input_1, input_2)
            elif direction.get() == "right":
                self.move_right(x, y, input_1, input_2)
            # elif direction.get() == "random":
            #     move_random()
            else:
                self.no_op()

            root.after(100, move, self, direction, root, x, y, input_1, input_2)  # reschedule event in 2 seconds
            num_moves.set(num_moves.get() + 1)
    #        if num_moves.get() > 10:
    #            print("Survived!")
    #            global result
    #            result = 0
    #            root.destroy()

        root.after(100, move, self, direction, root, x, y, input_1, input_2)
        root.mainloop()

        return result

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

if __name__ == '__main__':
    width, height, wall_locs, win_locs, lose_locs = gen_world_from_image('t-maze.pnm')

    results = []
    for i in range(1):
        agent = Agent()

        agent.world = GridWorld(world_height=height, world_width=width,
                      wall_locs=wall_locs, win_locs=win_locs, lose_locs=lose_locs, viz=False)

        agent.world.solve()

        results.append(agent.agent())
    print(results)


