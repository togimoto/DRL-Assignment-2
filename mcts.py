import copy
import random
import math
import numpy as np
from game_env import *

DECISION = 0
CHANCE = 1

class UCTNode:
    def __init__(self, env: Game2048Env, node_type, visits=0, total_reward=0.0, parent=None, action=None):
        self.env = env
        self.empty_tiles = list(zip(*np.where(self.env.board == 0)))
        self.parent = parent
        self.action = action
        self.children: dict[int | tuple[int, int, int], UCTNode] = {}
        self.visits = visits
        self.total_reward = total_reward
        self.node_type = node_type # 0 = decision, 1 = chance
        self.max_tile = np.max(env.board)
        if node_type == DECISION:
            self.done = len([a for a in range(4) if env.is_move_legal(a)]) == 0
        else:
            self.done = len(self.empty_tiles) == 0

    def fully_expanded(self):
        return len(self.children) > 0


class UCTMCTS:
    def __init__(self, root: UCTNode, approximator_stage1, approximator_stage2, iterations=500, exploration_constant=0.008):
        # hyperparameters taken from Guei 2023
        self.root = root
        self.approximator_stage1 = approximator_stage1
        self.approximator_stage2 = approximator_stage2
        self.iterations = iterations
        self.c = exploration_constant
        self.last_action = None

    def select_child(self, node: UCTNode) -> UCTNode:
        assert node.fully_expanded()
        if node.node_type == DECISION:
            UCT_max = float('-inf')
            best_action = -1
            for action, child in node.children.items():
                Q = child.total_reward / child.visits
                UCT = Q + self.c * math.sqrt(math.log(node.visits) / child.visits)
                if UCT > UCT_max:
                    UCT_max = UCT
                    best_action = action
            assert best_action in node.children
            return node.children[best_action]
        else:
            x, y = random.choice(node.empty_tiles)
            new_tile = 2 if random.random() < 0.9 else 4
            return node.children[(x, y, new_tile)]
    
    def expand(self, node: UCTNode):
        score = node.env.score
        if node.node_type == DECISION:
            legal_actions = [a for a in range(4) if node.env.is_move_legal(a)]
            assert legal_actions
            for action in legal_actions:
                new_env = copy.deepcopy(node.env)
                new_env.step(action, add_random_tile=False)
                new_node = UCTNode(new_env, CHANCE, visits=1, total_reward=0, parent=node, action=action)
                log_board = self.flatten_board(new_env.board)
                if new_node.max_tile < 4096:
                    new_node.total_reward = (score + self.approximator_stage1.value(log_board)) / 50_000
                else:
                    new_node.total_reward = (score + self.approximator_stage2.value(log_board)) / 80_000
                node.children[action] = new_node
        else:
            log_board = self.flatten_board(node.env.board)
            if node.max_tile < 4096:
                board_reward = (score + self.approximator_stage1.value(log_board)) / 50_000
            else:
                board_reward = (score + self.approximator_stage2.value(log_board)) / 80_000
            for x, y in node.empty_tiles:
                for tile_value in (2, 4):
                    new_env = copy.deepcopy(node.env)
                    new_env.board[x, y] = tile_value
                    new_node = UCTNode(new_env, DECISION, parent=node, action=(x, y, tile_value))
                    new_node.visits = 1
                    new_node.total_reward = board_reward
                    node.children[(x, y, tile_value)] = new_node

    def backpropagate(self, node, reward):
        while True:
            node.visits += 1
            node.total_reward += reward
            if node.parent is None:
                break
            node = node.parent

    def run_simulation(self):
        node = self.root

        # Selection
        while node.fully_expanded() and not node.done:
            node = self.select_child(node)

        # Expansion
        done = node.done
        if not done:
            self.expand(node)

        # Rollout
        if not done:
            if node.node_type == DECISION:
                max_value = float("-inf")
                for child_node in node.children.values():
                    if child_node.total_reward > max_value:
                        max_value = child_node.total_reward
                reward_to_propagate = max_value
            else:
                reward_to_propagate = next(iter(node.children.values())).total_reward
        else:
            reward_to_propagate = node.env.score
            
        # Backpropagation
        self.backpropagate(node, reward_to_propagate)

    def get_best_action(self):
        best_visits = -1
        best_action = None
        for action, child in self.root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        assert best_action in range(4)
        return best_action

    def move_root(self, last_action, env: Game2048Env):
        if last_action in self.root.children:
            self.root = self.root.children[last_action]
            for x, y in np.ndindex(env.board.shape):
                if env.board[x, y] != self.root.env.board[x, y]:
                    key = (x, y, env.board[x, y])
                    if key in self.root.children:
                        self.root = self.root.children[key]
                    else:
                        self.root = UCTNode(copy.deepcopy(env), DECISION)
                    break
        else:
            self.root = UCTNode(copy.deepcopy(env), DECISION)
    
    def flatten_board(self, board: np.ndarray):
        flat_board = board.flatten()
        log_board = np.zeros_like(flat_board, dtype=int)
        non_zero_mask = flat_board != 0
        log_board[non_zero_mask] = np.log2(flat_board[non_zero_mask]).astype(int)
        return log_board