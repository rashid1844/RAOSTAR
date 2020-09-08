#!/usr/bin/env python

# Visual sim for rao* implementation on icaps continuous model
# Matt Deyo 2017
# mdeyo@mit.edu

from Tkinter import *
import time
import numpy as np
from iterative_raostar import *


class Simulator(object):
    def __init__(self, X_DIM, Y_DIM, graph, policy, model, grid_size=100):
        # note the policy here is the sorted one
        self.master = Tk()
        self.master.title(model.name)
        # self.master.title("Hello world"

        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM
        self.border_width = 1
        self.gs = grid_size
        self.margin = 1.5 * grid_size
        self.w = self.X_DIM * grid_size + self.X_DIM + 1
        self.h = self.Y_DIM * grid_size + self.Y_DIM + 1
        self.C = Canvas(self.master, width=self.w + 2 *
                        self.margin, height=self.h + 2 * self.margin)
        self.C.pack()

        self.quad_poly = None
        self.guest_poly = None
        # self.C.bind_all('<space>', self.key_press)
        self.C.bind_all('<w>', self.key_press)
        self.C.bind_all('<a>', self.key_press)
        self.C.bind_all('<d>', self.key_press)
        self.C.bind_all('<s>', self.key_press)
        self.key = 0
        self.policy = policy
        self.graph = graph
        self.model = model
        self.current_state = graph.root
        two_sqrt_two = 1 / np.sqrt(2)
        self.model_actions = {"LEFT": [-1, 0], "UP-LEFT": [-two_sqrt_two, -two_sqrt_two],
                              "UP": [0, -1], "UP-RIGHT": [two_sqrt_two, -two_sqrt_two], "RIGHT": [1, 0], "DOWN-RIGHT": [two_sqrt_two, two_sqrt_two], "DOWN": [0, 1], "DOWN-LEFT": [-two_sqrt_two, two_sqrt_two]}
        self.done_policy = False
        self.draw_grid()

    def convert_coords(self, x, y):
        x1 = x * self.gs + x + 1 + self.margin
        y1 = self.h - (y * self.gs + y) + self.margin
        return (x1, y1)

    def draw_risk_colors(self, x, y, risk):
        x1, y1 = self.convert_coords(x, y)
        gray_num = 100 - 1 - int(risk * 100)
        color = "gray" + str(gray_num)
        # print(color)
        radius = 10
        self.C.create_oval(x1 - radius, y1 - radius, x1 +
                           radius, y1 + radius, fill=color)

    def draw_ego(self, node):
        min_size = 5
        # print('node:', node.state)
        # print(node.state.mean_b)
        center_x, center_y = self.convert_coords(
            float(node.state.mean_b[0]), float(node.state.mean_b[1]))
        width_x = max(min_size, float(node.state.sigma_b[0, 0]) * self.gs)
        width_y = max(min_size, float(node.state.sigma_b[1, 1]) * self.gs)
        # center_x += self.margin
        # center_y += self.margin

        x = center_x - width_x
        y = center_y - width_y
        print(center_x, center_y, width_x, width_y, x, y)

        self.ego_poly_stddev = self.C.create_oval(x, y, x + width_x * 2, y + width_y * 2, outline="gray",
                                                  fill="", width=2)
        self.ego_poly = self.C.create_oval(center_x - min_size, center_y - min_size, center_x + min_size, center_y + min_size, outline="red",
                                           fill="red", width=2)

    def draw_agent(self, node):
        print('risk', self.model.state_risk(node.state))
        min_size = 5
        center_x, center_y = self.convert_coords(
            float(node.state.agent_mean_b[0]), float(node.state.agent_mean_b[1]))
        width_x = max(min_size, float(
            node.state.agent_sigma_b[0, 0]) * self.gs)
        width_y = max(min_size, float(
            node.state.agent_sigma_b[1, 1]) * self.gs)
        x = center_x - width_x
        y = center_y - width_y

        self.agent_poly_stddev = self.C.create_oval(x, y, x + width_x * 2, y + width_y * 2, outline="gray",
                                                    fill="", width=2)
        self.agent_poly = self.C.create_oval(center_x - min_size, center_y - min_size, center_x + min_size, center_y + min_size, outline="red",
                                             fill="red", width=2)

    def update_agent(self, node):
        print('risk', self.model.state_risk(node.state))
        min_size = 5
        center_x, center_y = self.convert_coords(
            float(node.state.agent_mean_b[0]), float(node.state.agent_mean_b[1]))
        width_x = max(min_size, float(
            node.state.agent_sigma_b[0, 0]) * self.gs)
        width_y = max(min_size, float(
            node.state.agent_sigma_b[1, 1]) * self.gs)
        x = center_x - width_x
        y = center_y - width_y

        # self.agent_poly_stddev = self.C.create_oval(x, y, x + width_x * 2, y + width_y * 2, outline = "gray",
#                                             fill="", width=2)
# self.agent_poly = self.C.create_oval(center_x - min_size, center_y - min_size, center_x + min_size, center_y + min_size, outline="red",
#                                      fill="red", width=2)

        self.C.coords(self.agent_poly, (center_x - min_size, center_y -
                                        min_size, center_x + min_size, center_y + min_size))

    def draw_next_action(self, node):
        center_x, center_y = self.convert_coords(
            float(node.state.mean_b[0]), float(node.state.mean_b[1]))
        if node.best_action:
            action_name = node.best_action.name.split("'")[1]
            # print('action_name', action_name)
            action_map = None
            for action_key in self.model_actions:
                # print(action_key)
                if action_key == action_name:
                    # print('matched action!')
                    arrow_dir = self.model_actions[action_key]
                    # print(arrow_dir)
                    arrow_x = center_x + \
                        (arrow_dir[0] * self.model.vel * self.gs) / 2
                    arrow_y = center_y + \
                        (arrow_dir[1] * self.model.vel * self.gs) / 2
                    # print(center_x, center_y, arrow_x, arrow_y)

                    self.C.create_line(center_x, center_y, arrow_x, arrow_y, arrow=LAST,
                                       width=5, arrowshape=(16, 20, 6))
        else:
            print('no best_action')
            self.done_policy = True

    def draw_static_obstacle(self):
        x1 = 3
        y1 = 4
        x2 = 4
        y2 = 6
        x3 = 5
        y3 = 7
        x4 = 6
        y4 = 3

        x1, y1 = self.convert_coords(x1, y1)
        x2, y2 = self.convert_coords(x2, y2)
        x3, y3 = self.convert_coords(x3, y3)
        x4, y4 = self.convert_coords(x4, y4)

        self.C.create_polygon(
            x1, y1, x2, y2, x3, y3, x4, y4, fill="blue")

    def draw_grid(self):
        # grid line is black
        m_r = 255
        m_g = 255
        m_b = 255
        gs = self.gs

        x, y = self.model.goal_area
        x_goal, y_goal = self.convert_coords(x, y)
        x_goal_top, y_goal_top = self.convert_coords(10, 10)

        self.C.create_rectangle(
            x_goal, y_goal, x_goal_top, y_goal_top, fill="green", outline="")
        # draw lines for grid
        for i in range(self.X_DIM + 1):
            linex = (i + 1) + gs * i  # plot vertical lines
            self.C.create_line(linex + self.margin, self.margin,
                               linex + self.margin, self.h + self.margin)
        for j in range(self.Y_DIM + 1):
            liney = (j + 1) + gs * j  # horizontal lines
            self.C.create_line(self.margin, liney + self.margin,
                               self.w + self.margin, liney + self.margin)

        self.draw_static_obstacle()

        self.draw_ego(self.graph.root)
        self.draw_agent(self.graph.root)

        self.draw_next_action(self.graph.root)

        self.master.update()
        return True

    def move_quad(self, pose):
        (posx, posy, postheta, t) = pose
        # find corresponding pixel value
        x = posx * self.gs + posx + int(self.gs / 2)
        y = self.h - (posy * self.gs + posy + int(self.gs / 2))  # fo
        thet = postheta / 360. * 2 * np.pi
        l = self.gs * 1 / 3.
        a = np.pi * 2 / 3.
        [x1, y1] = [int(x + l * np.cos(thet)), int(y - l * np.sin(thet))]
        [x2, y2] = [int(x + l * np.cos(thet - a)),
                    int(y - l * np.sin(thet - a))]
        [x3, y3] = [int(x + l * np.cos(thet + a)),
                    int(y - l * np.sin(thet + a))]
        self.C.coords(self.quad_poly, (x1, y1, x2, y2, x3, y3))

    def move_guest(self, pose):
        (posx, posy, postheta, t) = pose
        # find corresponding pixel value
        x = posx * self.gs + posx + int(self.gs / 2)
        y = self.h - (posy * self.gs + posy + int(self.gs / 2))  # fo
        thet = postheta / 360. * 2 * np.pi
        l = self.gs * 1 / 3.
        a = np.pi * 2 / 3.
        [x1, y1] = [int(x + l * np.cos(thet)), int(y - l * np.sin(thet))]
        [x2, y2] = [int(x + l * np.cos(thet - a)),
                    int(y - l * np.sin(thet - a))]
        [x3, y3] = [int(x + l * np.cos(thet + a)),
                    int(y - l * np.sin(thet + a))]
        self.C.coords(self.guest_poly, (x1, y1, x2, y2, x3, y3))

    def get_policy(self, combined_state):
        state2 = combined_state[1]
        return self.policy[combined_state[0]][(state2[0], state2[1], state2[2], 0)]

    def update_new_state(self):
        act = self.policy[self.current_state[0]][self.current_state[1]]
        newstate = self.model.state_transitions(
            self.current_state, act)[self.event]
        self.current_state = newstate[0]

    def key_press(self, event=None):
        if self.done_policy:
            self.master.quit()
        else:
            if event.char == 'w':
                self.current_state = most_likely_next_state(
                    self.graph, self.model, self.current_state)
                self.draw_ego(self.current_state)
                self.update_agent(self.current_state)
                self.draw_next_action(self.current_state)
                self.master.update()

            elif event.char == 'a':
                self.current_state = agent_move_next_state(
                    self.graph, self.model, self.current_state, "LEFT")
                self.draw_ego(self.current_state)
                self.update_agent(self.current_state)
                self.draw_next_action(self.current_state)
                self.master.update()

            elif event.char == 'd':
                self.current_state = agent_move_next_state(
                    self.graph, self.model, self.current_state, "RIGHT")
                self.draw_ego(self.current_state)
                self.update_agent(self.current_state)
                self.draw_next_action(self.current_state)
                self.master.update()

    def start_sim(self):
        self.master.mainloop()
