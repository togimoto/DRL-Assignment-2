from collections import defaultdict
import pickle

from game import Board, IllegalAction, ACTIONS

class NTupleApproximator:
    def __init__(self, symmetric_tuples, c=17):
        self.symmetric_tuples = symmetric_tuples
        self.num_tuple_groups = len(symmetric_tuples)
        self.num_transformations = len(symmetric_tuples[0])
        self.c = c
        self.LUTs = [defaultdict(float) for _ in self.symmetric_tuples]

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
        for tuple_idx, tuple_group in enumerate(self.symmetric_tuples):
            for tuple in tuple_group:
                feature_list = [board[idx] for idx in tuple]
                feature_index = self.get_feature_index(feature_list)
                v += self.LUTs[tuple_idx][feature_index]
        return v

    def update(self, board, delta, alpha):
        for tuple_idx, tuple_group in enumerate(self.symmetric_tuples):
            for tuple in tuple_group:
                feature_list = [board[idx] for idx in tuple]
                feature_index = self.get_feature_index(feature_list)
                self.LUTs[tuple_idx][feature_index] += (1/(self.num_tuple_groups * self.num_transformations)) * alpha * delta

    def get_best_action(self, state):
        """
        Note:
        Due to how the game env is implemented, the env does not stop the game
        immediately after the last legal move is played, but instead allows the
        agent to make one more move, which is illegal and triggers the
        IllegalAction exception.

        Hence, this method must return an action, even if there are no valid
        actions. The way it is done here (as in the original code from where the
        game env is taken from) is to let the reward of any illegal move be 0,
        thereby guaranteeing at least one move will be greater than
        float("-inf").
        """
        best_action = None
        best_reward = float("-inf")
        for candidate in ACTIONS:
            dummy_env = Board(state)
            try:
                incremental_reward = dummy_env.act(candidate)
            except IllegalAction:
                reward = 0
            else:
                afterstate = dummy_env.copyboard()
                reward = incremental_reward + self.value(afterstate)
            if reward > best_reward:
                best_reward = reward
                best_action = candidate
        return best_action

def load_approximator(approximator_path):
    NUM_TUPLE_GROUPS = 4
    NUM_TRANSFORMATIONS = 8
    with open("4x6_symmetric_tuples.pkl", "rb") as file:
        SYMMETRIC_TUPLES = pickle.load(file)
        assert len(SYMMETRIC_TUPLES) == NUM_TUPLE_GROUPS
        for tuple_group in SYMMETRIC_TUPLES:
            assert len(tuple_group) == NUM_TRANSFORMATIONS
    approximator = NTupleApproximator(SYMMETRIC_TUPLES)
    with open(approximator_path, "rb") as file:
        approximator.LUTs = pickle.load(file)
    return approximator