import numpy as np


class Gridworld:
    def __init__(self, dim=5, final_reward=100, transition_penalty=-1):
        self.dim = dim
        self.flat_states = np.arange(dim * dim)
        self.state_grid = np.arange(dim * dim).reshape((dim, dim))
        self.terminal_state = self.flat_states[-1]
        self.agent_loc_x, self.agent_loc_y = 0, 0
        self.terminal_state_tuple = (dim - 1, dim - 1)
        self.init_state_feature_dict()
        self.state = self.state_feature_dict[0]
        self.final_reward = final_reward
        self.transition_penalty = transition_penalty

    def _clip_loc_to_grid(self, x, y):
        new_x = min(max(0, x), dim - 1)
        new_y = min(max(0, y), dim - 1)
        return new_x, new_y

    def get_reward(self, x, y):
        if self.state_grid[x][y] == self.terminal_state:
            return self.final_reward
        return self.transition_penalty

    def init_state_feature_dict(self):
        state_feature_dict = {}
        for idx, state in enumerate(self.flat_states):
            state_feature_dict[state] = np.zeros(self.dim * self.dim)
            state_feature_dict[state][idx] = 1.0
        self.state_feature_dict = state_feature_dict

    def step(self, action):
        """Actions: {'L', 'R', 'U', 'D'}"""
        done = False
        if action == "L":
            self.agent_loc_y -= 1
        elif action == "R":
            self.agent_loc_y += 1
        elif action == "U":
            self.agent_loc_x += 1
        elif action == "D":
            self.agent_loc_x -= 1

        # Snap coordiantes back to clip_loc_to_grid
        self.agent_loc_x, self.agent_loc_y = self._clip_loc_to_grid(
            self.agent_loc_x, self.agent_loc_y
        )

        reward = self.get_reward(self.agent_loc_x, self.agent_loc_y)
        next_state = self.state_grid[self.agent_loc_x][self.agent_loc_y]
        if next_state == self.terminal_state:
            done = True

        return self.state_feature_dict[next_state], reward, done

    def show_grid():
        pass

    def get_true_q_values(self):
        pass

    def reset(self):
        self.agent_location = (0, 0)
