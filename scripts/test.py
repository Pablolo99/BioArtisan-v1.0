import random
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions
from rdkit.Chem import MolFromSmiles, MolToSmiles
import math
import typing as ty
from abc import ABC, abstractmethod
from typing import List
from collections import defaultdict, namedtuple
from random import choice
import numpy as np

# Load the predictor model
predictor_model = joblib.load("C:/Users/pablo/PycharmProjects/BioArtisan-v1.0/scripts/model.pkl")

# Define the reaction SMARTS pattern
pk_reaction = '[S:1][C:2](=[O:3])([*:4]).[S:5][C:6](=[O:7])[*:8][C:9](=[O:10])[O:11]>>[S:5][C:6](=[O:7])[*:8][C:2](=[O:3])([*:4])'
pk_rxn = rdChemReactions.ReactionFromSmarts(pk_reaction)

extenders = [
    'O=C(O)CC(=O)S',
    'CC(C(=O)O)C(=O)S',
    'CCC(C(=O)O)C(=O)S',
    'O=C(O)C(CCCl)C(=O)S',
    'O=C(O)C(O)C(=O)S',
    'COC(C(=O)O)C(=O)S',
    'NC(C(=O)O)C(=O)S',
]
starters = [
    'CC(S)=O',
    'CCC(S)=O'
]

class Node(ABC):
    """
    A representation of a single molecule state. MCTS works by constructing a tree of these Nodes.

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
        Random successor of this molecule state (for more efficient simulation)
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
        Assumes `self` is terminal node.
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

    def __init__(self, exploration_weight: int = 1, predict_model: str = pk_rxn ) -> None:
        """
        Initialize Monte Carlo tree searcher.

        :param int exploration_weight: Exploration weight.
        """
        self.Q = defaultdict(int)  # Total reward of each node.
        self.N = defaultdict(int)  # Total visit count for each node.
        self.children = dict()  # Children of each node.
        self.exploration_weight = exploration_weight
        self.predict_model = predictor_model # load predictor model
        self.starter_subunits = starters
    def prediction(self, predict_model) -> float:
        """
        Calculate the predictive value of a molecule using an external predictive model.

        :param predict_model: External predictive model.
        :return: Predictive value for the molecule.
        :rtype: float
        """
        if self.is_terminal:
            raise RuntimeError("Reward called on nonterminal molecule!")
        # create fp of the molecule
        smiles_string = self.state
        mol = Chem.MolFromSmiles(smiles_string)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        mol_fingerprint = list(fp)
        # calculate predicitve value of the fp
        predictive_value = predictor_model.predict([mol_fingerprint])[0]
        return predictive_value

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
            #if self.N[n] == 0:
             #   return float("-inf")  # Avoid unseen moves.
            return self.Q[n] / self.N[n]  # Average reward.

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

            node = self._uct_select(node)  # Descend a layer deeper.

    def _expand(self, node: Node) -> None:
        """
        Update the `children` dict with the children of `node`.

        :param Node node: Node to expand.
        """
        if node in self.children:
            return  # Already expanded.

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


_MOL = namedtuple("Molecule", "molecule terminal")

class Molecule(_MOL, Node):
    """
    Define a molecule
    """
    def __int__(self, state = None, is_terminal = None, starter_subunits: List[str] = starters):
        self.state = state
        self.is_terminal = is_terminal
        self.starter_subunits = starters
        #self.extender_subunits = extenders_subunits

    def check_terminal(self) -> None:
        """
        Check if the current molecule is terminal
        """
        if self.state:
            mol = MolFromSmiles(self.state)
            substructure = MolFromSmiles('CC(=O)O')
            #if the molecule do not have CC(=O)O is terminal
            if mol.HasSubstructMatch(substructure):
                self.is_terminal = False
            else:
                self.is_terminal = True

    def find_children(self) -> ty.Set["molecule"]:
        """
        All possible successors of the molecule state.

        :param "molecule": Molecule state to find successors of
        :return: Set of successor boards.
        :rtype: ty.Set["molecule"]
        """
        # if the molecule can not get more extended no more progress is done
        if self.terminal:
            return set()

        # if the molecule has not started
        if self.state == None:
            #run start_synthesis
            return self.start_synthesis()
        else:
            #run make_add function
            return self.make_add()

    def find_random_child(self) -> "molecule":
        """
        Random successor of this board state

        :param molecule: molecule to find random successor of.
        :return: Random successor molecule
        :rtype: "molecule"
        """
        # if the molecule is terminal, no additions can be made
        if self.terminal:
            return None
        # get all the possible children
        children = self.find_children()

        #randomly select a child
        if children:
            return random.choice(list(children))
        else:
            return None

    def start_synthesis(self) -> None:
        """
        Start the synthesis of a molecule by randomly choosing an initial state from starter subunits.
        """
        if not self.starter_subunits:
            raise ValueError("No starter subunits provided.")

        if self.state == None:
            self.state = random.choice(self.starter_subunits)
            self.is_terminal = False

        else:
            raise ValueError("The molecule has already started.")
    def make_add(self, extender_subunits: List[str], reaction: Chem.rdChemReactions.ChemicalReaction) -> List[str]:
        """
        Extend an existing molecule by adding an extender subunit.

        :param List[str] extender_subunits: List of extender subunits.
        """
        products = []
        if not self.state:
            raise ValueError("No initial state provided.")

        if self.is_terminal:
            raise RuntimeError("Cannot extend a terminal molecule.")

        for subunit in extender_subunits:
            mol1 = Chem.MolFromSmiles(self.state)
            mol2 = Chem.MolFromSmiles(subunit)
            reactants = (mol1, mol2)
            reaction_products = reaction.RunReactants(reactants)

            # Check if the reaction produced any products
            if reaction_products:
                for product in reaction_products:
                    combined_mol = product[0]
                    addition_result = Chem.MolToSmiles(combined_mol)
                    products.append(addition_result)

        return list(set(products))

    def reward(self, predictor_model) -> float:
        """
        Calculate the predictive value of a molecule using an external predictive model.

        :param predictor_model: External predictive model.
        :return: Predictive value for the molecule.
        :rtype: float
        """
        if self.is_terminal:
            raise RuntimeError("Reward called on nonterminal molecule!")

        #create fp of the molecule
        smiles_string = self.state
        mol = Chem.MolFromSmiles(smiles_string)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        mol_fingerprint = list(fp)

        #calculate predicitve value of the fp
        predictive_value = predictor_model.predict([mol_fingerprint])[0]
        return predictive_value

def generate_mol():
    """
    Generate a molecule
    """
    tree = MCTS()
    mol = new_mol(starters)
    #print(mol.to_pretty_string())

    while not mol.is_terminal:
        #while the molecule is not terminal, make extension
        products = mol.make_add()

        if mol.terminal:
            break

        #train as we go
        for i in range(100):
            tree.do_rollout(mol)

        mol = tree.choose(mol)
        # need to check in which format is mol
        print(mol)



def new_mol(starter_subunits: List[str]) -> Molecule:
    """
    Returns starting state of the molecule

    :return: Starting state of the molecule
    :rtype: Molecule
    """
    mol = Molecule(molecule = None, terminal = False)
    #proceed to start the synthesis
    mol.start_synthesis()
    return mol

def main() -> None:
    # Turn RDKit warnings off.
    Chem.rdBase.DisableLog("rdApp.error")

    #generate molecule
    generate_mol()


if __name__ == "__main__":
    main()