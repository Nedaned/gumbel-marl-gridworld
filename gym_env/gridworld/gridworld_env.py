import gym
import numpy as np
from gym import spaces


class GridworldEnv(gym.Env):
    def __init__(self, row, col, n_action, n_agent):
        super(GridworldEnv, self).__init__()

        self.row = row
        self.col = col
        self.n_agent = n_agent

        self.observation_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(n_action)

        self.target_locs = []
        for i_agent in range(self.n_agent):
            if i_agent == 0:
                self.target_locs.append(np.array([1, 0], dtype=np.uint16))
            else:
                self.target_locs.append(np.array([1, self.col - 1], dtype=np.uint16))

        assert n_action == 4, "Implemented only for up, down, right, and left"
        assert n_agent == 2, "Implemented only for two agent"

    def step(self, actions):
        for i_agent, action in enumerate(actions):
            self._take_action(action, i_agent)

        next_observations = self._get_normalize_loc()

        target_reached = 0
        for i_agent in range(self.n_agent):
            if np.array_equal(self.agent_locs[i_agent], self.target_locs[i_agent]):
                target_reached += 1
        reward = 1 if target_reached == int(self.n_agent) else 0
        rewards = [reward for _ in range(self.n_agent)]

        done = True if reward > 0 else False

        return (next_observations, rewards, done, {})

    def reset(self):
        self._init_env()
        return self._get_normalize_loc()

    def render(self):
        raise NotImplementedError()

    def _init_env(self):
        self.agent_locs = []
        for i_agent in range(self.n_agent):
            if i_agent == 0:
                self.agent_locs.append(np.array([0, self.col // 2 - 1], dtype=np.uint16))
            else:
                self.agent_locs.append(np.array([0, self.col // 2 + 1], dtype=np.uint16))

    def _get_normalize_loc(self):
        normalized_locs = []
        for agent_loc in self.agent_locs:
            normalized_loc = np.array([
                agent_loc[0] / float(self.row),
                agent_loc[1] / float(self.col)])
            normalized_locs.append(normalized_loc)

        return normalized_locs

    def _take_action(self, action, i_agent):
        action = np.argmax(action)

        if action == 0:
            new_loc = self.agent_locs[i_agent][0] - 1
            if new_loc < 0:
                new_loc = 0  # Out of bound
            self.agent_locs[i_agent] = np.array([new_loc, self.agent_locs[i_agent][1]])
    
        elif action == 1:
            new_loc = self.agent_locs[i_agent][0] + 1
            if new_loc >= self.row:
                new_loc = self.row - 1  # Out of bound
            self.agent_locs[i_agent] = np.array([new_loc, self.agent_locs[i_agent][1]])
    
        elif action == 2:
            new_loc = self.agent_locs[i_agent][1] + 1
            if new_loc >= self.col:
                new_loc = self.col - 1  # Out of bound
            self.agent_locs[i_agent] = np.array([self.agent_locs[i_agent][0], new_loc])
    
        elif action == 3:
            new_loc = self.agent_locs[i_agent][1] - 1
            if new_loc < 0: 
                new_loc = 0  # Out of bound
            self.agent_locs[i_agent] = np.array([self.agent_locs[i_agent][0], new_loc])
    
        else:
            raise ValueError("Wrong action")
