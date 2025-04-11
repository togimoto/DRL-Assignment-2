import sys
import numpy as np
import random
import math
import itertools
import time

def move_pair_to_str(move_pair):
    return ",".join(move_pair)

class UCTNode:
    def __init__(self, turn, winner, empty_tiles: set[tuple], parent=None, action=None):
        assert len(empty_tiles) != 1
        self.parent = parent
        self.action = action
        self.turn = turn
        self.winner = winner
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        if len(empty_tiles) == 0:
            self.untried_stone_pairs = []
        else:
            self.untried_stone_pairs = list(map(lambda x: frozenset(x), itertools.combinations(empty_tiles, 2)))

    def fully_expanded(self):
        return len(self.untried_stone_pairs) == 0

class UCTMCTS:
    def __init__(self, exploration_constant=1.41):
        self.c = exploration_constant

    def select_child(self, node: UCTNode):
        # UCT = Q + c * sqrt(log(parent_visits)/child_visits)
        UCT_max = float('-inf')
        best_action = -1
        for action, child in node.children.items():
            Q = -child.total_reward / child.visits
            UCT = Q + self.c * math.sqrt(math.log(node.visits) / child.visits)
            if UCT > UCT_max:
                UCT_max = UCT
                best_action = action
        assert best_action in node.children
        return node.children[best_action]

    def rollout(self, sim_env: "SimulationGame"):
        while sim_env.empty_tiles and not sim_env.check_win():
            selected_two_stones = random.sample(sim_env.empty_tiles, 2)
            moves_str = ",".join(selected_two_stones)
            sim_env.play_move("B" if sim_env.turn == 1 else "W", moves_str)
        winner = sim_env.check_win()
        return winner

    def backpropagate(self, node: UCTNode, winner: int):
        while True:
            node.visits += 1
            if winner != 0:
                if node.turn == winner:
                    node.total_reward += 2
                else:
                    node.total_reward -= 2
            if node.parent is None:
                break
            node = node.parent

    def run_simulation(self, root: UCTNode, env: "Connect6Game"):
        node = root
        sim_env = SimulationGame(env)

        # Selection
        while node.fully_expanded() and len(node.children) and not node.winner:
            node = self.select_child(node)
            sim_env.play_move("B" if sim_env.turn == 1 else "W", move_pair_to_str(node.action))

        # Expansion
        if not node.fully_expanded() and not node.winner:
            expanded_stone_pair = random.choice(node.untried_stone_pairs)
            node.untried_stone_pairs.remove(expanded_stone_pair)
            sim_env.play_move("B" if sim_env.turn == 1 else "W", move_pair_to_str(expanded_stone_pair))
            node = UCTNode(sim_env.turn, sim_env.check_win(), sim_env.empty_tiles, node, expanded_stone_pair)
            node.parent.children[expanded_stone_pair] = node

        # Rollout
        if not node.winner:
            winner = self.rollout(sim_env)
        else:
            winner = node.winner
        
        # Backpropagation
        self.backpropagate(node, winner)

    def get_best_action(self, root: UCTNode):
        max_visits = float("-inf")
        best_action = None
        for action, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                best_action = action
        return ",".join(best_action)

class SimulationGame:
    def __init__(self, env: "Connect6Game"):
        self.size = env.size
        self.board = env.board.copy()
        self.turn = env.turn
        self.game_over = env.game_over
        self.empty_tiles = env.empty_tiles.copy()

    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))
            self.empty_tiles.remove(stone)

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn

class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False
        self.empty_tiles = self.reset_empty_tiles()

        self.uct_mcts = UCTMCTS()
        self.root = UCTNode(self.turn, 0, self.empty_tiles)
        self.curr_turn_moves = []
        self.new_game = True

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        self.empty_tiles = self.reset_empty_tiles()
        self.uct_mcts = UCTMCTS()
        self.root = UCTNode(self.turn, 0, self.empty_tiles)
        self.curr_turn_moves = []
        self.new_game = True
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        self.empty_tiles = self.reset_empty_tiles()
        self.uct_mcts = UCTMCTS()
        self.root = UCTNode(self.turn, 0, self.empty_tiles)
        self.curr_turn_moves = []
        self.new_game = True
        print("= ", flush=True)
    
    def reset_empty_tiles(self):
        return [f"{self.index_to_label(c)}{r+1}" for r in range(self.size) for c in range(self.size)]

    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))
            self.empty_tiles.remove(stone)

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.curr_turn_moves.extend(stones)
        assert len(self.curr_turn_moves) <= 2

        if self.new_game or len(self.curr_turn_moves) == 2:
            self.new_game = False
            self.turn = 3 - self.turn

            winner = self.check_win()
            self.curr_turn_moves = frozenset(self.curr_turn_moves)
            if self.curr_turn_moves in self.root.children:
                self.root = self.root.children[self.curr_turn_moves]
                self.root.parent = None
                self.root.action = None
            else:
                self.root = UCTNode(self.turn, winner, self.empty_tiles)
            self.curr_turn_moves = []

        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generate MCTS move"""
        if self.game_over:
            print("? Game over")
            return
        
        if np.count_nonzero(self.board) == 0:
            middle = self.size // 2
            move_str = f"{self.index_to_label(middle)}{middle+1}"
            self.play_move(color, move_str)
            print(move_str, flush=True)
            print(move_str, file=sys.stderr)
            return

        max_secs = 14
        start_time = time.monotonic()
        while time.monotonic() - start_time < max_secs:
            self.uct_mcts.run_simulation(self.root, self)
        move_str = self.uct_mcts.get_best_action(self.root)
        
        self.play_move(color, move_str)

        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)
        return
    
    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print("env_board_size=19", flush=True)

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

if __name__ == "__main__":
    game = Connect6Game()
    game.run()