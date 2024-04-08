import random
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions
from rdkit.Chem import MolFromSmiles, MolToSmiles
import math
import typing as ty
from abc import ABC, abstractmethod
from random import choice
from typing import List
from collections import defaultdict, namedtuple

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
            raise RuntimeError(f'Choose called on terminal node {node}!')

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # Avoid unseen moves.
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
        while True:
            if node.is_terminal():
                reward = node.reward(predictor_model)
                return reward

            node = node.find_random_child()

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



_MOL = namedtuple("Molecule", "SMILES pred_value terminal num_adds")

class Molecule(_MOL, Node):

    def find_children(mol) -> ty.Set["Molecule"]:
        """
        All possible successors of this molecule state.

        :param mol: Molecule: Molecule to find successors of.
        :return: Set of successor molecules.
        :rtype: ty.Set["Molecule"]
        """
        if mol.terminal:  # If the game is finished then no moves can be made.
            return set()

        #if mol had started
        if mol.SMILES :
            # make a progression using each of the extender unions
            return {mol.make_progress(submol) for submol in extenders}

        #if mol had not started
        if not mol.SMILES :
            return {mol.make_progress(submol) for submol in starters}


    def find_random_child(mol) -> "Molecule":
        """
        Random successor of this mol state (for more efficient simulation).

        :param mol: Molecule: Molecule to find random successor of.
        :return: Random successor molecule.
        :rtype: "Molecule"
        """
        # If the molecule do not have C(O)O , no progress can be made
        if mol.terminal:
            return None

        # If the molecule has not started just use the starters
        if mol.SMILES == None:
            possible_sub_ads = starters
        # else use the extenders
        else:
            possible_sub_ads = extenders

        # Otherwise, make a progress in the molecule object
        return mol.make_progress(choice(possible_sub_ads))

    def reward(mol, predictor) -> float:
        """
        Calculates the prediction value of a molecule

        :param mol Molecule: Molecule to predict the value from
        :param predictor: Predictor model used
        :rtype: float
        """
        #obtain the fp of mol
        str_SMILES = mol.SMILES
        molec = Chem.MolFromSmiles(str_SMILES)
        fp = AllChem.GetMorganFingerprintAsBitVect(molec, 2, nBits=2048)
        mol_fingerprint = list(fp)

        # calculate predicitve value of the fp
        predictive_value = predictor_model.predict([mol_fingerprint])[0]
        return predictive_value

    def linear_add(mol, subunit: str) -> str:
        """
        Perform reaction between current molecule and another molecule

        :param mol: Molecule object to perform the linear addition
        :param subunit: Molecule in SMILES format to add
        :return: resulting Molecule in SMILES format.
        :rtype: str
        """
        #if the SMILES is None ( molecule generation has just started )
        if mol.SMILES is None:
            return subunit
        # if not, perform the reaction
        mol1 = Chem.MolFromSmiles(mol.SMILES)
        mol2 = Chem.MolFromSmiles(subunit)

        products = pk_rxn.RunReactants((mol1, mol2))

        if products:
            # Convert the product molecule to SMILES
            new_SMILES = Chem.MolToSmiles(products[0][0])
            return new_SMILES
        else:
            raise ValueError("Linear addition could not happened.")


    def is_terminal(mol) -> bool:
        """
        Returns True if the node has no children.

        :param mol Molecule: Molecule to check if terminal.
        :return: True if terminal, False otherwise.
        :rtype: bool
        """
        return mol.terminal

    def make_progress(mol, subunit) -> "Molecule":
        """
        Return a mol instance with one addition made.

        :param Molecule mol: molecule to make move on.
        :param subunit: molecule to add to the original molecule (mol)
        :return: Molecule with new atributes
        :rtype: Molecule
        """
        new_SMILES = mol.linear_add(subunit)
        new_count = mol.num_adds + 1
        new_mol = Molecule(new_SMILES, mol.pred_value, mol.terminal, new_count)
        #calc the new pred value
        new_pred_value = (new_mol.reward(predictor_model))

        if new_pred_value > 0.9:
            is_terminal = True

        elif new_count > 10:
            is_terminal = True
        else:
            is_terminal = None

        return Molecule(new_SMILES, new_pred_value, is_terminal, new_count)

def new_mol() -> Molecule:
    """
    Returns starting state of the molecule

    :return: Starting state of the molecule
    :rtype: Molecule
    """
    mol = Molecule(SMILES=None, pred_value=None, terminal=None, num_adds=0)
    return mol

def gen_molecule() -> Molecule:
    """
    Generates a molecule
    """
    tree = MCTS()
    mol = new_mol()

    while True:
        # train as we go, 10 rollouts
        for i in range(20):
            tree.do_rollout(mol)

        mol = tree.choose(mol)
        print(mol)
        if mol.terminal:
            break

    return mol


def main() -> None:
    # Turn RDKit warnings off.
    Chem.rdBase.DisableLog("rdApp.error")

    # Generate molecule
    mol = gen_molecule()
    #print(mol)

if __name__ == "__main__":
    main()

#Chem.rdBase.DisableLog("rdApp.error")
#mol1 = Molecule(SMILES='CCC(S)=O', pred_value=None, terminal=None)
#print(mol1.find_random_child())
#prods = Molecule.linear_add(mol1,mol2)
#print(prods)
