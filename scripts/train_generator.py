#!/usr/bin/env pyton3
"""
Description:    Run the Monte Carlo Tree Search algorithm for molecular design.
Usage:          python run_mcts.py -m path/to/trained/predictor 
"""
import argparse
import math
import joblib 
import typing as ty 
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple 
from random import choice

from rdkit import Chem 

def cli() -> argparse.Namespace:
    """
    Command Line Interface
    
    :return: Parsed command line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to trained predictor.")
    return parser.parse_args()

def load_model(path: str) -> joblib.load:
    """
    Load a trained predictor.
    
    :param str path: Path to trained predictor.
    :return: Trained predictor.
    :rtype: joblib.load
    """
    return joblib.load(path)

class Node(ABC):
    """
    A representation of a single board state. MCTS works by constructing a tree of these Nodes.

    Adapted from: https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
    """
    @abstractmethod
    def find_children(self) -> ty.Set:
        """
        All possible successors of this board state.
        """
        return set()

    @abstractmethod
    def find_random_child(self) -> ty.Optional["Node"]:
        """
        Random successor of this board state (for more efficient simulation)
        """
        return Node

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Returns True if the node has no children/
        """
        return True

    @abstractmethod
    def reward(self) -> float:
        """
        Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc.
        """
        return 0.0

    @abstractmethod
    def __hash__(self) -> int:
        """
        Nodes must be hashable.
        """
        return 123456789

    @abstractmethod
    def __eq__(node1: "Node", node2: "Node") -> bool:
        """
        Nodes must be comparable.
        """
        return True

class MCTS:
    """
    Monte Carlo tree searcher.

    Adapted from: https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
    """
    def __init__(self, exploration_weight: int = 1) -> None:
        """
        Initialize Monte Carlo tree searcher.
        
        :param int exploration_weight: Exploration weight.
        """
        self.Q = defaultdict(int)  # Total reward of each node.
        self.N = defaultdict(int)  # Total visit count for each node.
        self.children = dict()  # Children of each node.
        self.exploration_weight = exploration_weight

    def choose(self, node: Node) -> Node:
        """
        Choose the best successor of node. (Choose a move in the game).

        :param Node node: Node to choose successor from.
        :return: The chosen successor node.
        :rtype: Node
        """
        if node.is_terminal():
            raise RuntimeError(f"Choose called on terminal node {node}!")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf") # Avoid unseen moves.
            return self.Q[n] / self.N[n] # Average reward.

        return max(self.children[node], key=score)

    def do_rollout(self, node: Node) -> None:
        """
        Make the tree one layer better. (Train for one iteration.)

        :param Node node: Node to do rollout from.
        """
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node: Node) -> ty.List[Node]:
        """
        Find an unexplored descendent of `node`.

        :param Node node: Node to select from.
        :return: The path to the selected node.
        :rtype: ty.List[Node]
        """
        path = []

        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # Node is either unexplored or terminal.
                return path
            
            unexplored = self.children[node] - self.children.keys()

            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            
            node = self._uct_select(node) # Descend a layer deeper.

    def _expand(self, node: Node) -> None:
        """
        Update the `children` dict with the children of `node`.

        :param Node node: Node to expand.
        """
        if node in self.children:
            return # Already expanded.
        
        self.children[node] = node.find_children()

    def _simulate(self, node: Node) -> float:
        """
        Returns the reward for a random simulation (to completion) of `node`.

        :param Node node: Node to simulate.
        :return: Reward for simulation.
        :rtype: float
        """
        invert_reward = True

        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            
            node = node.find_random_child()
            invert_reward = not invert_reward

    def _backpropagate(self, path: ty.List[Node], reward: float) -> None:
        """
        Send the reward back up to the ancestors of the leaf.

        :param ty.List[Node] path: Path to leaf node.
        :param float reward: Reward to propagate.
        """
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward # 1 for me is 0 for my enemy, and vice versa.

    def _uct_select(self, node: Node) -> Node:
        """
        Select a child of node, balancing exploration & exploitation.

        :param Node node: Node to select from.
        :return: The selected child node.
        :rtype: Node
        """
        # All children of node should already be expanded.
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n: Node) -> float:
            """
            Upper confidence bound for trees.

            :param Node n: Node to calculate UCT for.
            :return: UCT value.
            """
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(log_N_vertex / self.N[n])

        return max(self.children[node], key=uct)

_TTTB = namedtuple("TicTacToeBoard", "tup turn winner terminal")

# Inheriting from a namedtuple is convenient because it makes the class immutable and predefines:
#   __init__
#   __repr__
#   __hash__    
#   __eq__
#   ... and others

class TicTacToeBoard(_TTTB, Node):
    """
    Define a Tic Tac Toe board.

    Adapted from: https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
    """
    def find_children(board) -> ty.Set["TicTacToeBoard"]:
        """
        All possible successors of this board state.
        
        :param TicTacToeBoard board: Board to find successors of.
        :return: Set of successor boards.
        :rtype: ty.Set["TicTacToeBoard"]
        """
        if board.terminal: # If the game is finished then no moves can be made.
            return set()
        
        # Otherwise, you can make a move in each of the empty spots.
        return {board.make_move(i) for i, value in enumerate(board.tup) if value is None}

    def find_random_child(board) -> "TicTacToeBoard":
        """
        Random successor of this board state (for more efficient simulation).
        
        :param TicTacToeBoard board: Board to find random successor of.
        :return: Random successor board.
        :rtype: "TicTacToeBoard"
        """
        if board.terminal:
            return None # If the game is finished then no moves can be made.
        
        empty_spots = [i for i, value in enumerate(board.tup) if value is None]

        return board.make_move(choice(empty_spots))

    def reward(board) -> float:
        """
        Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc.
        
        :param TicTacToeBoard board: Board to calculate reward for.
        :return: Reward for board.
        :rtype: float
        """
        if not board.terminal:
            raise RuntimeError(f"Reward called on nonterminal board {board}!")
        
        if board.winner is board.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"Reward called on unreachable board {board}!")
        
        if board.turn is (not board.winner):
            return 0  # Your opponent has just won. Bad.
        
        if board.winner is None:
            return 0.5  # Board is a tie.
        
        # The winner is neither True, False, nor None.
        raise RuntimeError(f"Board has unknown winner type {board.winner}!")

    def is_terminal(board) -> bool:
        """
        Returns True if the node has no children.

        :param TicTacToeBoard board: Board to check if terminal.
        :return: True if terminal, False otherwise.
        :rtype: bool
        """
        return board.terminal

    def make_move(board, index: int) -> "TicTacToeBoard":
        """
        Return a board instance like this one but with one move made.
        
        :param TicTacToeBoard board: Board to make move on.
        :param int index: Index of move to make.
        :return: Board with move made.
        :rtype: "TicTacToeBoard"
        """
        tup = board.tup[:index] + (board.turn,) + board.tup[index + 1 :]
        turn = not board.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v is None for v in tup)

        return TicTacToeBoard(tup, turn, winner, is_terminal)

    def to_pretty_string(board) -> str:
        """
        Returns a string representation of the board.
        
        :param TicTacToeBoard board: Board to represent as string.
        :return: String representation of board.
        :rtype: str
        """
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [[to_char(board.tup[3 * row + col]) for col in range(3)] for row in range(3)]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )

def play_game():
    """
    Play a game of tic-tac-toe.
    """
    tree = MCTS()
    board = new_tic_tac_toe_board()
    print(board.to_pretty_string())

    while True:
        row_col = input("enter row,col: ")
        row, col = map(int, row_col.split(","))
        index = 3 * (row - 1) + (col - 1)

        if board.tup[index] is not None:
            raise RuntimeError("Invalid move")
        
        board = board.make_move(index)
        print(board.to_pretty_string())

        if board.terminal:
            break

        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(250):
            tree.do_rollout(board)

        board = tree.choose(board)
        print(board.to_pretty_string())

        if board.terminal:
            break

def _winning_combos():
    """
    All winning combos for tic-tac-toe.

    Adapted from: https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
    """
    for start in range(0, 9, 3): # Three in a row
        yield (start, start + 1, start + 2)

    for start in range(3): # Three in a column
        yield (start, start + 3, start + 6)

    yield (0, 4, 8) # Down-right diagonal
    yield (2, 4, 6) # Down-left diagonal

def _find_winner(tup: ty.Tuple[ty.Optional[bool], ...]) -> ty.Optional[bool]:
    """
    Returns None if no winner, True if X wins, False if O wins.
    
    :param ty.Tuple[ty.Optional[bool], ...] tup: Tuple to find winner of.
    :return: Winner of tuple.
    :rtype: ty.Optional[bool]

    Adapted from: https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
    """
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]

        if False is v1 is v2 is v3:
            return False
        
        if True is v1 is v2 is v3:
            return True
        
    return None

def new_tic_tac_toe_board() -> TicTacToeBoard:
    """
    Returns starting state of the board.

    :return: Starting state of the board.
    :rtype: TicTacToeBoard

    Adapted from: https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
    """
    return TicTacToeBoard(tup=(None,) * 9, turn=True, winner=None, terminal=False)

def main() -> None:
    # Turn RDKit warnings off.
    Chem.rdBase.DisableLog("rdApp.error")

    # Parse command line arguments.
    args = cli()

    # Load trained predictor.
    predictor = load_model(args.model)
    print(f"Loaded predictor from {args.model}.")

    # Run MCTS.
    # TODO: Repurpose tic-tac-toe MCTS for molecular design.

    # ... play a game of tic-tac-toe.
    play_game()
    print("Done.")

    exit(0)

if __name__ == "__main__":
    main()