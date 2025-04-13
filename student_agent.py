import numpy as np
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math

from ntupleapproximator import load_approximator
from game import Board
from mcts import *
from game_env import *

approximators = []
approximators.append(load_approximator("4x6_LUTs_230000eps_no_random_states.pkl"))
approximators.append(load_approximator("4x6_LUTs_stage4096_140000eps_no_random_states.pkl"))
fast_env = Board()
env = Game2048Env()
state = env.reset()
done = False
previous_score = 0
board_size = 4

root = UCTNode(copy.deepcopy(env), DECISION)
mcts = UCTMCTS(root, approximators[0], approximators[1], iterations=1_500)
last_action = None

def get_action(state, score):
    global last_action
    env.board = state
    env.score = score

    if last_action:
        mcts.move_root(last_action, env)

    for _ in range(mcts.iterations):
        mcts.run_simulation()
    
    action = mcts.get_best_action()

    legal_actions = [a for a in range(4) if env.is_move_legal(a)]
    if legal_actions and action not in legal_actions:
        action = random.choice(legal_actions)
    
    last_action = action

    return action