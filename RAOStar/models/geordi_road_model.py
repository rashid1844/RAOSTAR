#!/usr/bin/env python

# author: Matt Deyo
# email: mdeyo@mit.edu
# simple vehicle models based on MERS Toyota work for rao star

import pygame
import math


class RoadModel(object):

    def __init__(self, length=100):
        self.lanes = {}
        self.length = length

    def add_road_lane(self, lane_number, lane_name, open_interval, wall=False):
        self.lanes[str(lane_number)] = {
            "lane_number": lane_number,
            "lane_name": lane_name,
            "open_interval": open_interval,
            "min": open_interval[0],
            "max": open_interval[1],
            "wall": wall}

    def get_lane_right(self, lane_num):
        if self.lanes[str(lane_number)]['wall']:
            raise ValueError(
                'Starting right_lane check from divide, not a lane')
        right_lane_num = str(int(lane_number) + 2)
        if right_lane_num not in self.lanes:
            raise ValueError(
                'should not be checking right_lane proximity after right_open was False')
        return int(right_lane_num)

    def right_open(self, position, lane_number):
        if self.lanes[str(lane_number)]['wall']:
            raise ValueError(
                'Starting right_lane check from divide, not a lane')
        right_divide_num = str(int(lane_number) + 1)
        right_lane_num = str(int(lane_number) + 2)
        if right_lane_num not in self.lanes or right_divide_num not in self.lanes:
            return False
        right_divide = self.lanes[right_divide_num]
        right_lane = self.lanes[right_lane_num]
        if self.in_open_interval(right_divide, position) and self.in_open_interval(right_lane, position):
            return True
        return False

    def left_open(self, position, lane_number):
        if self.lanes[str(lane_number)]['wall']:
            raise ValueError(
                'Starting left_open check from divide, not a lane')
        left_divide_num = str(int(lane_number) - 1)
        left_lane_num = str(int(lane_number) - 2)
        if left_lane_num not in self.lanes or left_divide_num not in self.lanes:
            return False
        left_divide = self.lanes[left_divide_num]
        left_lane = self.lanes[left_lane_num]
        if self.in_open_interval(left_divide, position) and self.in_open_interval(left_lane, position):
            return True
        return False

    def valid_forward(self, finish_x, lane_num):
        return self.in_open_interval(self.lanes[str(lane_num)], finish_x)

    def in_open_interval(self, lane_data, position):
        if position >= lane_data['min'] and position <= lane_data['max']:
            return True
        return False


##################################
# Road model visualization tools #
##################################

def plot_road_model(model):
    # Set the HEIGHT and WIDTH of the screen
    # WINDOW_SIZE = [(WIDTH + MARGIN) * X_DIM + MARGIN,
    #                (HEIGHT + MARGIN) * Y_DIM + MARGIN]
    screen = pygame.display.set_mode([1000, 500])

    # Set title of screen
    pygame.display.set_caption("Road Model")

    # Loop until the user clicks the close button.
    done = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)

    road_length = model.length
    road_start_x = 20
    road_lane_width = 20
    line_width = 4
    road_x_scale = 3

    # -------- Main Program Loop -----------
    while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop

        # basicfont = pygame.font.SysFont('Comic Sans MS', 36)
        # Set the screen background
        screen.fill(WHITE)
        for lane in model.lanes:
            # print(model.lanes[lane])
            lane_data = model.lanes[lane]
            if lane_data['wall']:
                color = BLACK
                pygame.draw.line(screen, color, (road_start_x, road_start_x + lane_data['lane_number'] * road_lane_width), (
                    road_start_x + lane_data['min'] * road_x_scale, road_start_x + lane_data['lane_number'] * road_lane_width), line_width)
                pygame.draw.line(screen, color, (road_start_x + lane_data['max'] * road_x_scale, road_start_x + lane_data['lane_number'] * road_lane_width), (
                    road_start_x + road_length * road_x_scale, road_start_x + lane_data['lane_number'] * road_lane_width), line_width)
                draw_dashed_line(screen, color, (road_start_x + lane_data['min'] * road_x_scale, road_start_x + lane_data['lane_number'] * road_lane_width), (
                    road_start_x + lane_data['max'] * road_x_scale, road_start_x + lane_data['lane_number'] * road_lane_width), line_width, line_width)
            else:
                pygame.draw.rect(
                    screen, GREEN, [road_start_x + lane_data['min'] * road_x_scale, road_start_x + lane_data['lane_number'] * road_lane_width - road_lane_width / 2, (lane_data['max'] - lane_data['min']) * road_x_scale, road_lane_width])

        # pygame.draw.rect(screen, BLACK, [10, 10, 50, 50])

        # Limit to 1 frames per second
        clock.tick(1)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

     # Be IDLE friendly. If you forget this line, the program will 'hang'
    # on exit.
    pygame.quit()


class Point:
    # constructed using a normal tupple
    def __init__(self, point_t=(0, 0)):
        self.x = float(point_t[0])
        self.y = float(point_t[1])
    # define all useful operators

    def __add__(self, other):
        return Point((self.x + other.x, self.y + other.y))

    def __sub__(self, other):
        return Point((self.x - other.x, self.y - other.y))

    def __mul__(self, scalar):
        return Point((self.x * scalar, self.y * scalar))

    def __div__(self, scalar):
        return Point((self.x / scalar, self.y / scalar))

    def __truediv__(self, scalar):
        return Point((self.x / scalar, self.y / scalar))

    def __len__(self):
        return int(math.sqrt(self.x**2 + self.y**2))
    # get back values in original tuple format

    def get(self):
        return (self.x, self.y)


def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=10):
    origin = Point(start_pos)
    target = Point(end_pos)
    displacement = target - origin
    length = len(displacement)
    if length > 0:
        slope = displacement / length

        for index in range(0, int(length / dash_length), 2):
            start = origin + (slope * index * dash_length)
            end = origin + (slope * (index + 1) * dash_length)
            pygame.draw.line(surf, color, start.get(), end.get(), width)


#######################
# Road model examples #
#######################

def highway_2_lanes_offramp_ex():
    '''Example showing how to create an explicit cc-MDP with the new model format.
     A simple path planning in a 3 x 2 grid (with no loops).
      ----------------------
      lane-1
      ----------------------
      lane-2
      ----------------------
            \ lane-exit
              --------------'''

    new_road = RoadModel(200)
    new_road.add_road_lane(0, "wall-inner", (0, 0), True)
    new_road.add_road_lane(1, "lane-1", (0, 200))
    new_road.add_road_lane(2, "line-1-2", (0, 200), True)
    new_road.add_road_lane(3, "lane-2", (0, 200))
    new_road.add_road_lane(4, "wall-outer", (60, 120), True)
    new_road.add_road_lane(5, "lane-exit", (60, 200))
    new_road.add_road_lane(6, "wall-exit", (0, 0), True)

    return new_road

def highway_2_lanes():
    '''Example showing how to create an explicit cc-MDP with the new model format.
      ----------------------
      lane-1
      ----------------------
      lane-2
      ----------------------'''

    new_road = RoadModel(200)
    new_road.add_road_lane(0, "wall-inner", (0, 0), True)
    new_road.add_road_lane(1, "lane-1", (0, 200))
    new_road.add_road_lane(2, "line-1-2", (0, 200), True)
    new_road.add_road_lane(3, "lane-2", (0, 200))
    new_road.add_road_lane(4, "wall-outer", (0, 200), True)

    return new_road

def intersection_left_turn_ex():
    new_road = RoadModel(200)
    new_road.add_road_lane(0, "wall-inner", (0, 0), True)
    new_road.add_road_lane(1, "lane-1", (0, 200))
    new_road.add_road_lane(2, "line-1-2", (0, 200), True)
    new_road.add_road_lane(3, "lane-2", (0, 200))
    new_road.add_road_lane(4, "wall-outer", (0, 200), True)

    return new_road