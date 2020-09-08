#!/usr/bin/env python

# author: Yun Chang
# email: yunchang@mit.edu
# r2d2 model as simple model to test rao star

import numpy as np
from belief import BeliefState


class R2D2Model(object):
    # noted there are 7 blocks total, A through G
    # G is the goal, F is fire
    def __init__(self, length=1, DetermObs=True):
        # icy blocks are defined blocks that are icy
        self.icy_blocks = []
        self.icy_blocks_lookup = {}
        for i in range(length + 1):
            icy = (1, i)
            self.icy_blocks.append((1, i))
            self.icy_blocks_lookup[icy] = 1

        self.icy_move_forward_prob = 0.8
        self.DetermObs = DetermObs
        # if observation deterministic or not. If determinstic, once move,
        # no excatly where it is. If not, P(o_k+1|s_k+1) = 0.6 for the cell
        # it actually is in and 0.1 for the neighborinf cells
        if not self.DetermObs:
            self.obsProb = 0.6
        # environment will be represented as a 3 x 3 grid, with (2,0) and (2,2) blocked
        # top left corner of grid is (0,0) and first index is row
        self.length = length
        self.env = np.zeros([3, 2 + self.length])

        # setup bottom left and right corners as impassible
        self.env[2, 0] = 1
        self.env[2, self.length + 1] = 1

        # fire located at self.env[2,1]: terminal
        self.fires = []
        for i in range(self.length):
            self.fires.append((2, i + 1))

        # goal position
        self.goal = (1, self.length + 1)

        self.optimization = 'minimize'  # want to minimize the steps to goal
        self.action_map = {
            "right": 5,
            "left": 6,
            "up": 7,
            "down": 8
        }
        self.action_list = [(1, 0, "down"), (0, 1, "right"),
                            (-1, 0, "up"), (0, -1, "left")]
        self.action_list = [(1, 0, "down"), (0, 1, "right"),
                            (-1, 0, "up")]

    def state_valid(self, state):  # check if a particular state is valid
        if state[0] < 0 or state[1] < 0:
            return False
        try:
            return self.env[state[0], state[1]] == 0
        except IndexError:
            return False

    def in_a_fire(self, state):
        # print(state)
        # print(self.fires)

        for fire in self.fires:
            if state[0] == fire[0] and state[1] == fire[1]:
                # print('   risk!!   ')
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
        # if state[0] == 1 and state[1] == 1:
        #     print('got to center cell')
        #     return []
        return validActions

    def is_terminal(self, state):
        # For some reason we get a BeliefState here when deadend state found
        if isinstance(state, BeliefState):
            state = state.belief.keys()[0]
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

        if (state[0], state[1]) in self.icy_blocks and "right" in action:
            # print('got right action!')
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
        if self.DetermObs:
            return [(state, 1.0)]
        else:  # robot only knows if it is on icy or non icy block
            if state in self.icy_blocks:
                return [(self.icy_blocks[i], 1 / len(self.icy_blocks)) for i in range(len(self.icy_blocks))]
            else:
                prob = 1 / (6 - len(self.icy_blocks))
                dist = []
                for i in range(3):
                    for j in range(3):
                        if self.env[i, j] == 0 and (i, j) not in self.icy_blocks:
                            dist.append((i, j), prob)
                return dist

    def state_risk(self, state):
        # For some reason we get a BeliefState here when deadend state found
        if isinstance(state, BeliefState):
            state = state.belief.keys()[0]
        if self.in_a_fire(state):
            return 1.0
        return 0.0

    def costs(self, action):
        return 1  # try uniform cost for all actions
        # if action[2] == "up":
        #     return 2  # bias against up action, models climbing above ice as harder
        # else:
        #     return 1

    def values(self, state, action):
        # return value (heuristic + cost)
        # print('state here', state)
        # if state[0] == 0:
            # return 2.0
        # return 1.0
        return self.costs(action)
        # return self.costs(action) + self.heuristic(state)

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
        height, width = self.env.shape
        print(" ")
        print("    ** Model environment **")
        for j in range(height):
            # print("row: " + str(j))
            row_str = "   "
            for i in range(width):
                if self.env[j][i]:
                    row_str += " [-----] "
                    continue
                row_str += " [" + str(j) + "," + str(i)
                if self.goal == (j, i):
                    row_str += " g] "
                elif self.state_risk((j, i)):
                    row_str += " f] "
                elif (j, i) in self.icy_blocks_lookup:
                    row_str += " i] "
                else:
                    row_str += "  ] "
            print(row_str)

    def print_policy(self, policy):
        height, width = self.env.shape
        policy_map = np.zeros([height, width])
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
        for j in range(height):
            # print("row: " + str(j))
            row_str = "   "
            for i in range(width):
                if self.goal == (j, i):
                    row_str += " [goal] "
                elif self.state_risk((j, i)):
                    row_str += " [fire] "
                elif self.env[j][i]:
                    row_str += " [----] "
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
