#!/usr/bin/env python

# author: Yun Chang
# email: yunchang@mit.edu
# expanded r2d2 model 
# (introducing fire, ice, and sand)

import numpy as np

class Environment(object):
    def __init__(self, n, start, goal):
        # start and end coordinates cannot have fire  
        # square environment 
        self.start = start
        self.goal = goal
        self.size = n 
        self.environment = np.zeros([n,n])
        self.fireblocks = []
        self.icyblocks = []
        self.sandblocks = []
        # 0 -> normal, 1 -> icy, 2-> sand, 3-> fire 
    def list_of_available(self):
        # return a list of available coordinates
        # available means not icy, not sand, and not fire 
        n = self.size
        avail = []
        for i in range(n):
            for j in range(n):
                if self.environment[i,j] == 0:
                    avail.append((i,j))
        return avail

    def add_fireblocks(self, listofcoords):
        for coord in listofcoords:
            if coord == self.start or coord == self.goal:
                print("cannot place fire on start/end, discarded... ")
            else: 
                self.environment[coord[0], coord[1]] = 3
        listofcoords.sort()
        self.fireblocks = listofcoords

    def add_sandblocks(self, listofcoords):
        for coord in listofcoords:
            self.environment[coord[0], coord[1]] = 2
        listofcoords.sort()
        self.sandblocks = listofcoords

    def add_icyblocks(self, listofcoords):
        for coord in listofcoords:
            self.environment[coord[0], coord[1]] = 1
        listofcoords.sort()
        self.icyblocks = listofcoords

    def add_fireblocks_rand(self, num_per_col):
        not_ins = [] # as to prevent firewall 
        for col in range(self.size):
            if col == 0:
                not_ins.append(self.start[0]) # can't have fire on start 
            if col == self.size - 1:
                not_ins.append(self.goal[0]) # can't have fire on end
            for j in range(self.size): # check no predefined
                if self.environment[j, col] != 0:
                    not_ins.append(j)
            row = [i for i in range(self.size) if i not in not_ins]
            not_ins = []
            for i in range(num_per_col):
                c = np.random.choice(row)
                if c-1 >= 0: 
                    not_ins.append(c-1)
                if c+1 < self.size:
                    not_ins.append(c+1)
                row.remove(c)
                self.environment[c,col] = 3 # add fire to grid 
                self.fireblocks.append([c,col])
        self.fireblocks.sort()

    def add_icyblocks_rand(self):
        # place icy blocks on side of fire 
        for fire in self.fireblocks:
            r = fire[0]
            col = fire[1]
            if r-1 >=0 and self.environment[r-1,col] == 0:
                self.environment[r-1,col] = 1
                self.icyblocks.append([r-1,col])
            if r+1 < self.size and self.environment[r+1,col] == 0:
                self.environment[r+1,col] = 1
                self.icyblocks.append([r+1,col])
        self.icyblocks.sort()

    def add_sandblocks_rand(self):
        # sand blocks should be placed last
        remain = self.list_of_available()
        # of the remaining, half be sand, half be normal 
        for block in remain:
            if np.random.sample() > 0.5:
                self.environment[block[0], block[1]] = 2
                self.sandblocks.append(block)
        self.sandblocks.sort()

    def block_type(self, block):
        d = {0:"normal", 1:"icy", 2:"sand", 3:"fire"}
        num = self.environment[block[0], block[1]]
        return d[num]

    def print_environment(self):
        n = self.size
        print(" ")
        print("    ** FIS environment **")
        for i in range(n):
            row_str = "   "
            for j in range(n):
                if self.environment[i,j] == 3:
                    row_str += " [fire]"
                elif self.environment[i,j] == 1:
                    row_str += " [ice ]"
                elif self.environment[i,j] == 2:
                    row_str += " [sand]"
                else:
                    row_str += " [    ]"
                if [i,j] == self.start:
                    row_str += "*"
                elif [i,j] == self.goal:
                    row_str += "$"
                else:
                    row_str += " "
            print(row_str)

class fire_ice_sand_Model(object):
    # noted there are 7 blocks total, A through G
    # G is the goal, F is fire
    def __init__(self, n, icy_blocks, sand_blocks, fire_blocks, start_coord, goal_coord):
        self.goal = goal_coord
        self.start = start_coord
        # n x n environment 
        self.env = Environment(n, start_coord, goal_coord)
        # fire blocks are the places r2d2 need to avoid (or DIE)
        # fire_blocks: list of coordinates of the fire blocks 
        # sand_blocks: stable (non slippery) but more effort 
        # icy_blocks: icy blocks are defined blocks that are icy (slips)
        self.env.add_fireblocks(fire_blocks)
        self.env.add_icyblocks(icy_blocks)
        self.env.add_sandblocks(sand_blocks)

        if len(fire_blocks) == 0:
            self.env.add_fireblocks_rand(1) # note for now 1 fire per column, can be changed! 
            # ex add_fireblocks_rand(2)
        if len(icy_blocks) == 0:
            self.env.add_icyblocks_rand()
        if len(sand_blocks) == 0:
            self.env.add_sandblocks_rand()

        self.icy_move_forward_prob = 0.8
        self.optimization = 'minimize'  # want to minimize the steps to goal
        self.action_map = {
            "right": 5,
            "left": 6,
            "up": 7,
            "down": 8
        }
        self.action_list = [(1, 0, "down"), (0, 1, "right"),
                            (-1, 0, "up"), (0, -1, "left")]

    def state_valid(self, state):  # check if a particular state is valid
        if state[0] < 0 or state[1] < 0:
            return False
        if state[0] < self.env.size and state[1] < self.env.size:
            return True 

    def in_a_fire(self, state):
        if self.env.block_type(state) == "fire":
            return True 
        return False

    def actions(self, state):
        # print('actions for: ' + str.(state))
        validActions = []
        for act in self.action_list:
            newx = state[0] + act[0]
            newy = state[1] + act[1]
            if self.state_valid((newx, newy)):
                validActions.append(act)
        if state == self.goal:
            return []
        if self.in_a_fire(state):
            return []
        return validActions

    def is_terminal(self, state):
        # Added fire state to terminal to differentiate it from deadends
        return state[0] == self.goal[0] and state[1] == self.goal[1] or self.in_a_fire(state)

    def state_transitions(self, state, action):
        newstates = []
        # intended_new_state = (state[0] + action[0],
        #                       state[1] + action[1])
        # added depth to the state
        intended_new_state = (state[0] + action[0],
                              state[1] + action[1], state[2] + 1)
        if not self.state_valid(intended_new_state):
            return newstates
        if self.env.block_type(state) == "icy":
            newstates.append([intended_new_state, self.icy_move_forward_prob])
            for slip in [-1, 1]:
                slipped = [(action[i] + slip) % 2 * slip for i in range(2)]
                # slipped_state = (state[0] + slipped[0],
                #                  state[1] + slipped[1])
                # added depth to the state
                slipped_state = (state[0] + slipped[0],
                                 state[1] + slipped[1], state[2] + 1)
                if self.state_valid(slipped_state):
                    newstates.append(
                        [slipped_state, (1 - self.icy_move_forward_prob) / 2])
        else:
            newstates.append([intended_new_state, 1.0])

        # Need to normalize probabilities for cases where slip only goes to one
        # cell, not two possible cells
        sum_probs = sum(n for _, n in newstates)
        for child in newstates:
            child[1] = child[1] / sum_probs

        return newstates

    def observations(self, state):
        return [(state, 1.0)] # deterministic for now 

    def state_risk(self, state):
        if self.in_a_fire(state):
            return 1.0
        return 0.0

    def costs(self, action):
        if action[2] == "up":
            return 2  # bias against up action, models climbing above ice as harder
        else:
            return 1

    def values(self, state, action):
        # return value (heuristic + cost)
        # return self.costs(action)
        cost = self.costs(action)
        if self.env.block_type(state) == "sand":
            cost += 1
        return cost + self.heuristic(state)

    def heuristic(self, state):
        # square of euclidean distance as heuristic
        # mdeyo: found this issue! We are trying to minimize the values so the
        # heuristic should be an underestimate,which the square of distance is
        # not if each action then cost 1 or 2
        return np.sqrt(sum([(self.goal[i] - state[i])**2 for i in range(2)]))

    def execution_risk_heuristic(self, state):
        # sqaure of euclidean distance to fire as heuristic
        return 0
        # return sum([(self.fire[i] - state[i])**2 for i in range(2)])

    def print_model(self):
        self.env.print_environment()

    def print_policy(self, policy):
        n = self.env.size
        policy_map = np.zeros([n,n])
        depth_found = {}

        for key in policy:
            coords = key.split(":")[0].split("(")[1].split(")")[0]
            col = int(coords.split(",")[0])
            row = int(coords.split(",")[1])
            depth = int(coords.split(",")[2])
            col_row_str = str(col) + ',' + str(row)
            action_string = policy[key]
            for action_name in self.action_map:
                if action_name in action_string:
                    if col_row_str not in depth_found:  # first depth found
                        depth_found[col_row_str] = depth
                        policy_map[col][row] = self.action_map[action_name]
                        break
                    elif depth < depth_found[col_row_str]:
                        depth_found[col_row_str] = depth
                        policy_map[col][row] = self.action_map[action_name]
                        break
        print(" ")
        print("         ** Policy **")
        for j in range(n):
            # print("row: " + str(j))
            row_str = "   "
            for i in range(n):
                if self.goal == (j, i):
                    row_str += " [goal] "
                elif self.state_risk((j, i)):
                    row_str += " [fire] "
                else:
                    if policy_map[j][i] == 5:
                        row_str += " [ -> ] "
                    if policy_map[j][i] == 6:
                        row_str += " [ <- ] "
                    if policy_map[j][i] == 7:
                        row_str += " [ ^^ ] "
                    if policy_map[j][i] == 8:
                        row_str += " [ vv ] "
                    if policy_map[j][i] == 0:
                        row_str += " [    ] "
            print(row_str)

if __name__ == "__main__":
    e = Environment(8, [7,0], [0,7])
    e.add_fireblocks_rand(2)
    e.add_icyblocks_rand()
    e.add_sandblocks_rand()
    e.print_environment()