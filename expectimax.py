import numpy as np

from game import Board, IllegalAction, ACTIONS

def choice_node(env: Board, approximator, depth):
    assert depth >= 0
    best_action = 0
    max_reward = 0
    for a in ACTIONS:
        sim_env = Board(env.copyboard())
        try:
            reward = sim_env.act(a)
        except IllegalAction:
            continue
        if depth == 0:
            reward += approximator.value(sim_env.board)
        else:
            reward += chance_node(sim_env, approximator, depth)
        if reward > max_reward:
            max_reward = reward
            best_action = a
    return (best_action, max_reward)

def chance_node(env: Board, approximator, depth):
    assert depth >= 1
    empty_tiles = env.empty_tiles()
    if not empty_tiles:
        return 0
    weighted_rewards = []
    for tile in empty_tiles:
        for val in [1, 2]:
            sim_env = Board(env.copyboard())
            sim_env.board[tile] = val
            _, r = choice_node(sim_env, approximator, depth-1)
            if val == 1:
                weight = 0.9
            else:
                weight = 0.1
            weighted_rewards.append(weight * (1 / len(empty_tiles)) * r)
    return np.sum(weighted_rewards)