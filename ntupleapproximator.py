import copy
import math
from collections import defaultdict

def step_without_adding_random_tile(env, action):
    assert env.action_space.contains(action), "Invalid action"

    if action == 0:
        moved = env.move_up()
    elif action == 1:
        moved = env.move_down()
    elif action == 2:
        moved = env.move_left()
    elif action == 3:
        moved = env.move_right()
    else:
        moved = False

    env.last_move_valid = moved

    done = env.is_game_over()

    return env.board, env.score, done, {}

# needed for pickle
def default_lut():
    return defaultdict(float)

class NTupleApproximator:
    def __init__(self, board_size, patterns, c=18):
        self.board_size = board_size
        self.patterns = patterns

        self.LUTs = [default_lut() for _ in self.patterns]
        self.c = c
        self.dummy_env = None

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        feature_list = []
        for x, y in coords:
            feature = self.tile_to_index(board[x, y])
            assert 0 <= feature < self.c
            feature_list.append(feature)
        return feature_list
    
    def get_feature_index(self, feature_list):
        """
        We use Horner's rule to compute the index, which is the polynomial
        f_0 * c^(0) + f_1 * c^(1) + ... + f_(n-1) * c^(n-1)
        """
        index = 0
        for feature in reversed(feature_list):
            index = index * self.c + feature
        return index

    def value(self, board):
        v = 0
        for pattern_idx, pattern in enumerate(self.patterns):
            feature_list = self.get_feature(board, pattern)
            feature_index = self.get_feature_index(feature_list)
            pattern_value = self.LUTs[pattern_idx][feature_index]
            v += pattern_value
        return v

    def update(self, board, delta, alpha):
        for pattern_idx, pattern in enumerate(self.patterns):
            feature_list = self.get_feature(board, pattern)
            feature_index = self.get_feature_index(feature_list)
            self.LUTs[pattern_idx][feature_index] += (1/17) * alpha * delta
    
    def get_best_action(self, env, previous_score):
        if self.dummy_env is None:
            self.dummy_env = copy.deepcopy(env)

        legal_moves = [a for a in range(4) if env.is_move_legal(a)]
        assert legal_moves, "env should not be in terminal state."
        max_q = float("-inf")
        best_action = -1
        best_incremental_reward = None
        best_afterstate = None
        best_afterstate_value = None
        for candidate_action in legal_moves:
            self.dummy_env.board = env.board.copy()
            self.dummy_env.score = env.score
            afterstate, new_score, _, _ = step_without_adding_random_tile(self.dummy_env, candidate_action)
            incremental_reward = new_score - previous_score
            afterstate_value = self.value(afterstate)
            curr_q = incremental_reward + afterstate_value
            if curr_q > max_q:
                max_q = curr_q
                best_action = candidate_action
                best_incremental_reward = incremental_reward
                best_afterstate = afterstate
                best_afterstate_value = afterstate_value
        assert best_action in legal_moves
        return best_action, best_incremental_reward, best_afterstate, best_afterstate_value

# We use the 17 4-tuples described in Szubert and Jaskowski (2014)
patterns = [
    # 4 horizontals
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((1, 0), (1, 1), (1, 2), (1, 3)),
    ((2, 0), (2, 1), (2, 2), (2, 3)),
    ((3, 0), (3, 1), (3, 2), (3, 3)),


    # 4 verticals
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 1), (1, 1), (2, 1), (3, 1)),
    ((0, 2), (1, 2), (2, 2), (3, 2)),
    ((0, 3), (1, 3), (2, 3), (3, 3)),
    

    # 9 squares
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (0, 2), (1, 1), (1, 2)),
    ((0, 2), (0, 3), (1, 2), (1, 3)),

    ((1, 0), (1, 1), (2, 0), (2, 1)),
    ((1, 1), (1, 2), (2, 1), (2, 2)),
    ((1, 2), (1, 3), (2, 2), (2, 3)),

    ((2, 0), (2, 1), (3, 0), (3, 1)),
    ((2, 1), (2, 2), (3, 1), (3, 2)),
    ((2, 2), (2, 3), (3, 2), (3, 3)),
]