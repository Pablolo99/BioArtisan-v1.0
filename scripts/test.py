import warnings
import argparse
import time
import pstats
import random
import joblib
import cProfile
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdChemReactions
from rdkit.Chem import MolFromSmiles, MolToSmiles
import math
import typing as ty
from abc import ABC, abstractmethod
from random import choice
from typing import List
from collections import defaultdict, namedtuple


warnings.filterwarnings('ignore')

# Turn off RDKit warnings
RDLogger.DisableLog('rdApp.error')

#set time
start_time = time.time()

def cli()-> argparse.Namespace:
    """
    Command Line Interface

    :return: Parsed command line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model file .pkl")
    parser.add_argument("--output", required=True, help="Path to output file .txt" )
    parser.add_argument("--num_molecules", required=True, help="Number of wanted molecules")
    parser.add_argument("--pred_limit", required=True, help="Lower limit predictor value desired")
    return parser.parse_args()

# Define the reaction SMARTS pattern
pk_reaction = '[S][C:2](=[O:3])([*:4]).[S:5][C:6](=[O:7])[*:8][C](=[O])[O]>>[S:5][C:6](=[O:7])[*:8][C:2](=[O:3])([*:4])'
pk_rxn = rdChemReactions.ReactionFromSmarts(pk_reaction)

extenders = [
    'O=C(O)CC(=O)S',
    'CC(C(=O)O)C(=O)S',
    'CCC(C(=O)O)C(=O)S',
    'O=C(O)C(CCCl)C(=O)S',
    'O=C(O)C(O)C(=O)S',
    'COC(C(=O)O)C(=O)S',
    'NC(C(=O)O)C(=O)S'
]
starters = [
    'CC(S)=O',
    'CCC(S)=O'
]

cyc_reactions = [
    '([O:4]=[C:3]([*:2])[*:1].[C:6]([*:5])([*:7])[*:8])>>([O:4][C:3]([*:2])([*:1])[C:6]([*:5])([*:7])[*:8])',
    '([O:4]=[C:3]([*:2])[*].[O:7][C:6]([*:5])[*:8])>>([O:4]=[C:3]([*:2])[O:7][C:6]([*:5])[*:8])',
    '([O:4]=[C:3]([*:2])[*:1].[O][C:6]([*:5])[*:8])>>([*:8][C:6]([*:5])[O:4][C:3]([*:2])[*:1])',
    '([C:1]([*:11])[C:2](=[O:8])[C:3][C:4](=[O:9])[C:5][C:6](=[O])[*:7])>>([C:1]([*:11])1=[C:2]([O:8])[C:3]=[C:4]([O:9])[C:5]=[C:6]1([*:7]))',
    '([C:1](=[O:8])[C:2]=[C:3][C:4][C:5](=[O:9]))>>([C:1](=[O:8])[C:2][C:3][C:4][C:5]([O:9]))',
    '([C:1](=[O:8])[C:2]=[C:3][C:4][C:5]([O]))>>([C:1]([O:8]1)=[C:2][C:3][C:4][C:5]1)',
    '([C:1](=[O:9])[C:2][C:3](=[O:10])[C:4][C:5](=[O:11])[C:6][C:7](=[O])[C:8])>>([C:1](=[O:9])[C:2]1=[C:3]([O:10])[C:4]=[C:5]([O:11])[C:6]=[C:7]1[C:8])',
    '([C:1](=[O:9])[C:2][C:3](=[O])[C:4][C:5](=[O:11])[C:6][C:7](=[O])[C:8])>>([C:1](=[O:9])[C:2]1=[C:3][C:4]=[C:5]([O:11])[C:6]=[C:7]1[C:8])',
    '([C:1](=[O:9])[C:2][C:3](=[O:10])[C:4][C:5](=[O])[C:6][C:7](=[O])[C:8])>>([C:1](=[O:9])[C:2]1=[C:3]([O:10])[C:4]=[C:5][C:6]=[C:7]1[C:8])',
    '([C:1](=[O:9])[C:2][C:3](=[O])[C:4][C:5](=[O])[C:6][C:7](=[O])[C:8])>>([C:1](=[O:9])[C:2]1=[C:3][C:4]=[C:5][C:6]=[C:7]1[C:8])',
    '([C:1](=[O:12])[C:2][C:3](=[O:13])[C:4][C:5](=[O:14])[C:6][C:7](=[O])[C:8][C:9](=[O])[C:10][C:11](=[O]))>>([C:1](=[O:12])[C:2]1=[C:3]([O:13])[C:4]=[C:5]([O:14])[C:6]2=[C:7]1[C:8]=[C:9][C:10]=[C:11]2)',
    '([C:1](=[O:12])[C:2][C:3](=[O])[C:4][C:5](=[O:14])[C:6][C:7](=[O])[C:8][C:9](=[O])[C:10][C:11](=[O]))>>([C:1](=[O:12])[C:2]1=[C:3][C:4]=[C:5]([O:14])[C:6]2=[C:7]1[C:8]=[C:9][C:10]=[C:11]2)',
    '([C:1](=[O:12])[C:2][C:3](=[O:13])[C:4][C:5](=[O])[C:6][C:7](=[O])[C:8][C:9](=[O])[C:10][C:11](=[O]))>>([C:1](=[O:12])[C:2]1=[C:3]([O:13])[C:4]=[C:5][C:6]2=[C:7]1[C:8]=[C:9][C:10]=[C:11]2)',
    '([C:1](=[O:12])[C:2][C:3](=[O])[C:4][C:5](=[O])[C:6][C:7](=[O])[C:8][C:9](=[O])[C:10][C:11](=[O]))>>([C:1](=[O:12])[C:2]1=[C:3][C:4]=[C:5][C:6]2=[C:7]1[C:8]=[C:9][C:10]=[C:11]2)',
    '([C:1](=[O:8])[C:2][C:3](=[O:9])[C:4][C:5](=[O:10])[C:6][C:7](=[O]))>>([C:1](=[O:8])[C:2]1=[C:3]([O:9])[C:4]=[C:5]([O:10])[C:6]=[C:7]1)',
    '([C:1](=[O:8])[C:2][C:3](=[O])[C:4][C:5](=[O:10])[C:6][C:7](=[O]))>>([C:1](=[O:8])[C:2]1=[C:3][C:4]=[C:5]([O:10])[C:6]=[C:7]1)',
    '([C:1](=[O:8])[C:2][C:3](=[O:9])[C:4][C:5](=[O])[C:6][C:7](=[O]))>>([C:1](=[O:8])[C:2]1=[C:3]([O:9])[C:4]=[C:5][C:6]=[C:7]1)',
    '([C:1](=[O:8])[C:2][C:3](=[O])[C:4][C:5](=[O])[C:6][C:7](=[O]))>>([C:1](=[O:8])[C:2]1=[C:3][C:4]=[C:5][C:6]=[C:7]1)',
    '([C:1](=[O:12])[C:2][C:3](=[O:13])[C:4][C:5](=[O:14])[C:6][C:7](=[O])[C:8][C:9](=[O:16])[C:10][C:11](=[O]))>>([C:1]([O:12])1[C:2]2[C:3]([O:13])=[C:4][C:5]([O:14])=[C:6][C:7]=2[C:8]=[C:9]([O:16])[C:10]=1[C:11])',
    '([C:1](=[O:15])[C:2][C:3](=[O:16])[C:4][C:5](=[O:17])[C:6][C:7](=[O])[C:8][C:9](=[O:19])[C:10][C:11](=[O:20])[C:12][C:13](=[O:21])[C:14])>>([C:1]([O:15])1[C:2]2[C:3]([O:16])=[C:4][C:5]([O:17])=[C:6][C:7]=2[C:8]=[C:9]([O:19]3)[C:10]=1[C:11](=[O:20])[C:12][C:13]3([O:21])[C:14])'
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
    def check_terminal(self) -> bool:
        """
        Returns True if the node has no children/
        """
        return True

    @abstractmethod
    def reward(self, predictor) -> float:
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

    def __init__(self, exploration_weight: int = 0.5) -> None:
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
        if node.check_terminal():
            raise RuntimeError(f'Choose called on terminal node {node}!')

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # Avoid unseen moves.
            return self.Q[n] / self.N[n]  # Average reward.

        return max(self.children[node], key=score)

    def do_rollout(self, node: Node, predictor_model) -> None:
        """
        Make the tree one layer better. (Train for one iteration.)

        :param Node node: Node to do rollout from.
        """
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf, predictor_model)
        reward = self._simulate(leaf, predictor_model)
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

    def _expand(self, node: Node, predictor_model) -> None:
        """
        Update the `children` dict with the children of `node`.

        :param Node node: Node to expand.
        """
        if node in self.children:
            return  # Already expanded.

        self.children[node] = node.find_children(predictor_model)

    def _simulate(self, node: Node, predictor_model) -> float:
        """
        Returns the reward for a random simulation (to completion) of `node`.

        :param Node node: Node to simulate.
        :return: Reward for simulation.
        :rtype: float
        """
        while True:
            if node.check_terminal():
                reward = node.reward(predictor_model)
                return reward

            node = node.find_random_child(predictor_model)

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

    def find_children(mol, predictor_model) -> ty.Set["Molecule"]:
        """
        All possible successors of this molecule state.

        :param mol: Molecule: Molecule to find successors of.
        :return: Set of successor molecules.
        :rtype: ty.Set["Molecule"]
        """
        if mol.terminal:  # If the proc
            return set()

        children = set()
        #if mol had started
        if mol.SMILES :

            # make a progression using each of the extender unions
            for submol in extenders:
                children |= mol.make_progress(submol, cyc_reactions, predictor_model)

        #if mol had not started
        if not mol.SMILES :
            for submol in starters:
                children |= mol.make_progress(submol, cyc_reactions, predictor_model)

        return children


    def find_random_child(mol, predictor_model) -> "Molecule":
        """
        Random successor of this mol state (for more efficient simulation).

        :param mol: Molecule: Molecule to find random successor of.
        :return: Random successor molecule.
        :rtype: "Molecule"
        """
        # if the molecule do not have C(O)O  or it looks promising, no progress should be made
        if mol.terminal:
            return None

        # if the molecule has not started just use the starters
        if mol.SMILES is None:
            possible_sub_ads = starters
        # else use the extenders
        else:
            possible_sub_ads = extenders

        # otherwise, make a progress in the molecule object
        possible_children = mol.make_progress(choice(possible_sub_ads), cyc_reactions, predictor_model)
        #print(possible_children)
        possible_child = choice(list(possible_children))
        #print(possible_child)
        return possible_child

    def reward(mol, predictor) -> float:
        """
        Calculates the prediction value of a molecule

        :param Molecule mol: Molecule to predict the value from
        :param predictor: Predictor model used
        :rtype: float
        """

        # obtain the fp of mol
        str_SMILES = mol.SMILES
        if str_SMILES is None:
            return -1.0

        molec = Chem.MolFromSmiles(str_SMILES)
        fp = AllChem.GetMorganFingerprintAsBitVect(molec, 2, nBits=2048)
        mol_fingerprint = list(fp)

        # calculate predicitve value of the fp
        #make notice that proba returns the probs of each class: 0 (non-antibiotic), 1(antibiotic), and possible additional (non-relevant)
        predictive_value = predictor.predict_proba([mol_fingerprint])[0][1]
        return predictive_value



    def linear_add(mol, subunit: str) -> str:
        """
        Perform reaction between current molecule and another molecule

        :param mol: Molecule object to perform the linear addition
        :param subunit: Molecule in SMILES format to add
        :return: resulting Molecule in SMILES format.
        :rtype: str
        """
        # if the SMILES is None ( molecule generation has just started )
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

        #else:
            #print(Chem.MolToSmiles(mol1))
            #print(Chem.MolToSmiles(mol2))
            #raise ValueError("Linear addition could not happened.")
            #return None

    def cyclization(mol, reaction: str) -> ty.Set[str]:
        """
        Perform reactions within the current molecule atoms in order to obtain cycles

        :param mol: Molecule object to cycle
        :param reaction: chemical reaction
        :return: resulting Molecules in SMILES format
        :rtype: set(str)
        """
        smiles_mol = mol.SMILES
        inmol = Chem.MolFromSmiles(smiles_mol)
        unique_outputs = set()

        rxn = rdChemReactions.ReactionFromSmarts(reaction)
        try:
            # Run the reaction
            results = rxn.RunReactants([inmol])
            #iterate through each result and add to the
            for result in results:
                for molec in result:
                    Chem.SanitizeMol(molec)
                    unique_outputs.add(Chem.MolToSmiles(molec))

        except ValueError as e:
            None

        if unique_outputs != set():
            return unique_outputs


    def check_terminal(mol) -> bool:
        """
        Checks if a molecule is terminal or not
        :param Molecule mol: Molecule to check if it is terminal
        :return: True if the molecule is terminal, False otherwise
        :rtype: bool
        """

        if mol.pred_value > 0.9:
            return True

        if mol.num_adds >= 10:
            return True

        adds = mol.num_adds
        if adds > 1:
            smiles = Chem.MolFromSmiles(mol.SMILES)
            carboxylic_pattern = Chem.MolFromSmarts("C(=O)S")
            if not smiles.HasSubstructMatch(carboxylic_pattern):
                return True

        else:
            return False



    def make_progress(mol, subunit: str, reactions: ty.List[str], predictor_model) -> ty.Set["Molecule"]:

        """
        Return a set of mol instances with one addition made and cyclization events made.

        :param Molecule mol: molecule to make move on.
        :param subunit: molecule to add to the original molecule (mol)
        :param reactions: list of chemical reactions in SMILES format as strings
        :return: set["Molecule"], Molecule with new atributes
        :rtype: set["Molecule"]
        """
        all_newmol = set()

        new_SMILES = mol.linear_add(subunit)

        if mol.terminal:
            return

        if new_SMILES is not None:
            new_count = mol.num_adds + 1
            new_mol = Molecule(new_SMILES, mol.pred_value, mol.terminal, new_count)

            # calc the new pred value
            new_pred_value = new_mol.reward(predictor_model)

            # add just the linear version, no cyclizations
            just_linear_mol = Molecule(new_SMILES, new_pred_value, mol.terminal, new_count)
            check = just_linear_mol.check_terminal()

            just_linear_mol = Molecule(new_SMILES, new_pred_value, check, new_count)

            all_newmol.add(just_linear_mol)

            # if molecule has had ar least 4 additions, cyclate
            if new_count >= 4:
                #from new molecule, cyclate and obtain the new SMILES, pred values and terminal
                for cyc_rxn in reactions:

                    cyc_products = just_linear_mol.cyclization(cyc_rxn)

                    if cyc_products is None:
                        continue

                    for cyc_SMILES in cyc_products:

                        new_cyc_mol = Molecule(cyc_SMILES, None, None, new_count)
                        new_pred_value = new_cyc_mol.reward(predictor_model)

                        # add the counts and check terminal
                        cyc_mol = Molecule(cyc_SMILES, new_pred_value, mol.terminal, new_count)
                        check = cyc_mol.check_terminal()

                        cyc_mol = Molecule(cyc_SMILES, new_pred_value, check, new_count)
                        all_newmol.add(cyc_mol)


            return all_newmol
        #else:
        #    print("Linear addition could not happen.")
        #return all_newmol

def new_mol() -> Molecule:
    """
    Returns starting state of the molecule

    :return: Starting state of the molecule
    :rtype: Molecule
    """
    mol = Molecule(SMILES=None, pred_value=0.0, terminal=None, num_adds=0)
    return mol


def gen_molecule(predictor_model) -> Molecule:
    """
    Generates a molecule
    """
    tree = MCTS()
    mol = new_mol()

    while True:
        # train as we go, do X rollouts
        for i in range(5):
            # if the mol is terminal do not rollout
            if mol.terminal:
                break
            # if mol is not terminal do rollout
            else:
                tree.do_rollout(mol, predictor_model)


        mol = tree.choose(mol)
        if mol.terminal:
            break

    return mol


def main() -> None:
    """
    Main function to generate molecules until the desired number of molecules with pred_value = 1.0 is reached

    :param num_desired: The number of molecules with pred_value = 1.0 desired
    :type num_desired: int
    """

    # set the arguments from terminal line
    args = cli()

    predictor_model = joblib.load(args.model)
    output_file = args.output
    num_wanted = int(args.num_molecules)
    pred_limit = float(args.pred_limit)

    # turn warnings off.
    Chem.rdBase.DisableLog("rdApp.error")
    warnings.filterwarnings('ignore')
    RDLogger.DisableLog('rdApp.error')

    # generate molecules until the desired number of molecules with pred_value = 1.0 is reached
    with open(output_file, 'w') as generated_molecules:
        generated_molecules.write("ID\tSMILES\tpred_value\n")

        num_mols = 0

        while num_mols < num_wanted:
            mol = gen_molecule(predictor_model)
            if mol.pred_value >= pred_limit:
                print(mol)
                #write info in tsv format
                generated_molecules.write(f"{num_mols + 1}\t{mol.SMILES}\t{mol.pred_value}")
                num_mols += 1
                if num_mols < num_wanted:
                    generated_molecules.write('\n')

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"El script tardó {execution_time} segundos en ejecutarse.")

    print(f"{num_mols} molecules with pred_value >= {pred_limit} generated and saved to {output_file}")



if __name__ == "__main__":
    cProfile.run("main()", "profile_output.txt")
    stats = pstats.Stats("profile_output.txt")
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats()