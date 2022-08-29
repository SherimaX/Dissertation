# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eYQkaesVd23ZMBx2Td1iNmuonXuFNBrq
"""

class cube:
  def __init__(self, id, position):
    self.name = "cube_"+str(id)
    self.position = position
  def output(self):
    output = "        <include>\n"
    output += "            <name>{}</name>\n".format(str(self.name))
    output += "            <uri>model://cube</uri>\n"
    output += "            <pose>{} {} 0 0 0 0</pose>\n".format(self.position[0], self.position[1])
    output += "        </include>\n\n"
    return output

cube_1 = cube(0, [0, 0])

from PIL import Image
import numpy as np

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

x_offset = -7
y_offset = -4
scale = 0.6

walls = gen_world_from_image("t-maze.pnm")[2]
walls = [[-(w[0] + x_offset) * scale, (w[1] + y_offset) * scale] for w in walls]

file = ""
for i, w in enumerate(walls):
  new_cube = cube(i, w)
  file += new_cube.output()

world = open("template.txt", "r")
code = world.read()

out = open("t-maze.world", "w")
out.write(code.replace("cube", file))
