from Tkinter import *
import time
import numpy as np


from bebop_api_client import Bebop 

def grid2bebop(x,y,theta):
    # map descrete grid coordinates to bebop coord 
    # (wall at 0)
    b_x = (x-1)*0.3 # x_max in our space is 1.4
    b_y = (y-1)*0.3 # y_max in our space is 2.5
    b_thet = theta/360.*2.*np.pi # bebop takes radians? 
    return [b_x, b_y, b_thet]

class Simulator(object):
    def __init__(self, X_DIM, Y_DIM, policy, model, init_state, grid_size=100, bebop=False):
        # note the policy here is the sorted one
        self.master = Tk()
        w = X_DIM * grid_size + X_DIM + 1
        h = Y_DIM * grid_size + Y_DIM + 1
        self.X_DIM = X_DIM
        self.Y_DIM = Y_DIM
        self.w = w
        self.h = h
        self.C = Canvas(self.master, width=w, height=h)
        self.C.pack()
        self.gs = grid_size
        self.quad_poly = None
        self.guest_poly = None
        self.C.bind_all('<w>', self.key_press)
        self.C.bind_all('<s>', self.key_press)
        self.key = 0
        self.policy = policy
        self.model = model
        self.current_state = init_state
        self.bebop = bebop
        if bebop:
            self.quad = Bebop('quad')
            self.guest = Bebop('guest') 
            self.quad.takeoff()
            self.guest.takeoff() 
            # let quad be at height of 1 and guest be at height of 1
            [qx, qy, qth] = grid2bebop(init_state[0][0], init_state[0][1], init_state[0][2])
            qz = 1
            [gx, gy, gth] = grid2bebop(init_state[1][0], init_state[1][1], init_state[1][2])
            gz = 1
            self.quad.fly_to(qx, qy, qz, theta=qth)
            self.guest.fly_to(gx, gy, gz, theta=gth)

    def draw_grid(self):
        # grid line is black
        m_r = 255
        m_g = 255
        m_b = 255
        gs = self.gs
        # draw lines for grid
        for i in range(self.X_DIM + 1):
            linex = (i + 1) + gs * i  # plot vertical lines
            self.C.create_line(linex, 0, linex, self.h)
        for j in range(self.Y_DIM + 1):
            liney = (j + 1) + gs * j  # horizontal lines
            self.C.create_line(0, liney, self.w, liney)
        self.master.update()
        return True

    def draw_quad(self, pose):  # represent quad as triangle
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
        self.quad_poly = self.C.create_polygon(
            x1, y1, x2, y2, x3, y3, fill='red')
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
        if self.bebop:
            [q_x, q_y, q_thet] = grid2bebop(posx,posy,postheta)
            # print(q_x, q_y, q_thet)
            self.quad.fly_to(q_x, q_y, 1, theta=q_thet)

    def draw_guest(self, pose):  # ind corresponding pixel value
        (posx, posy, postheta, t) = pose  # draw the guest
        x = posx * self.gs + posx + int(self.gs / 2)  # f
        y = self.h - (posy * self.gs + posy + int(self.gs / 2))  # fo
        thet = postheta / 360. * 2 * np.pi
        l = self.gs * 1 / 3.
        a = np.pi * 2 / 3.
        [x1, y1] = [int(x + l * np.cos(thet)), int(y - l * np.sin(thet))]
        [x2, y2] = [int(x + l * np.cos(thet - a)),
                    int(y - l * np.sin(thet - a))]
        [x3, y3] = [int(x + l * np.cos(thet + a)),
                    int(y - l * np.sin(thet + a))]
        self.guest_poly = self.C.create_polygon(
            x1, y1, x2, y2, x3, y3, fill='green')
        self.master.update()
        return True

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
        if self.bebop:
            [g_x, g_y, g_thet] = grid2bebop(posx,posy,postheta)
            # print(g_x, g_y, g_thet)
            self.guest.fly_to(g_x, g_y, 1, theta=g_thet)

    def get_policy(self, combined_state):
        state2 = combined_state[1]
        return self.policy[combined_state[0]][(state2[0], state2[1], state2[2], 0)]

    def update_new_state(self):
        act = self.policy[self.current_state[0]][self.current_state[1]]
        newstate = self.model.state_transitions(
            self.current_state, act)[self.event]
        self.current_state = newstate[0]

    def key_press(self, event=None):
        if event.char == 'w':
            self.event = 1
            self.update_new_state()
            self.move_quad(self.current_state[0])
            self.move_guest(self.current_state[1])
        elif event.char == 's':
            self.event = 0
            self.update_new_state()
            self.move_quad(self.current_state[0])
            self.move_guest(self.current_state[1])

    def done(self):
        self.master.mainloop()


if __name__ == '__main__':
    D = Display(7, 7)
    D.draw_grid()
    D.draw_quad((3, 3, 0))
    D.draw_guest((5, 5, 90))
