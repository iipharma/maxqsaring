from maxqsaring.logging_helper import create_logger
from pandas import DataFrame
from typing import List, Dict
import hashlib
import random
import hmac
import os

from rdkit import Chem
from rdkit.Chem.Scaffolds import rdScaffoldNetwork, MurckoScaffold
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import PandasTools
import argparse
import json
import pandas as pd
import numpy as np
from rdkit import rdBase
from sklearn.model_selection import KFold, StratifiedKFold
from collections import defaultdict
from itertools import chain
rdBase.DisableLog("rdApp.*")


logger = create_logger('Splitting')

def scaffold_split_by_seed(df: DataFrame, seeds: List[int], frac: list, entity: str="Drug", add_no_mscaf=False) -> Dict:
    """create scaffold split. it first generates molecular scaffold for each molecule and then split based on scaffolds
    reference: https://github.com/chemprop/chemprop/blob/master/chemprop/data/scaffold.py

    Args:
        df (pd.DataFrame): dataset dataframe
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions
        entity (str): the column name for where molecule stores
    
    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test and values correspond to each dataframe
    """
    
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except:
        raise ImportError("Please install rdkit by 'conda install -c conda-forge rdkit'! ")
    from tqdm import tqdm
    from random import Random

    from collections import defaultdict
    

    s = df[entity].values
    scaffolds = defaultdict(set)
    idx2mol = dict(zip(list(range(len(s))),s))

    error_smiles = 0
    no_ms_index_sets =set()
    for i, smiles in tqdm(enumerate(s), total=len(s)):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol = Chem.MolFromSmiles(smiles), includeChirality = False)
            scaffolds[scaffold].add(i)
        except:
            logger.warning(f'{smiles} returns RDKit error and is thus omitted...')
            error_smiles += 1
            no_ms_index_sets.add(i)

    logger.info(f'{"="*5} Scaffold describe {"="*5}')
    tmp_counts = np.array([len(items) for key,items in scaffolds.items()])
    logger.info(f'There are {len(scaffolds)} murcko scaffolds.')
    logger.info(f'Mean: {np.mean(tmp_counts)}; Max: {np.max(tmp_counts)}; Min: {np.min(tmp_counts)}')
    logger.warning(f'There are {error_smiles} mols that have no murcko scaffolds.')

    train_size = int((len(df) - error_smiles) * frac[0])
    val_size = int((len(df) - error_smiles) * frac[1])
    test_size = (len(df) - error_smiles) - train_size - val_size
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    #index_sets = sorted(list(scaffolds.values()), key=lambda i: len(i), reverse=True)
    index_sets = list(scaffolds.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    
    split_ids_dict: dict = {}
    for seed in seeds:
        random = Random(seed)
        random.seed(seed)
        big_index_sets_copy=big_index_sets.copy()
        small_index_sets_copy = small_index_sets.copy()
        random.shuffle(big_index_sets_copy)
        random.shuffle(small_index_sets_copy)
        index_sets = big_index_sets_copy + small_index_sets_copy

        train, val, test = [], [], []

        if frac[2] == 0:
            for index_set in index_sets:
                if len(train) + len(index_set) <= train_size:
                    train += index_set
                    train_scaffold_count += 1
                else:
                    val += index_set
                    val_scaffold_count += 1
        else:
            for index_set in index_sets:
                if len(train) + len(index_set) <= train_size:
                    train += index_set
                    train_scaffold_count += 1
                elif len(val) + len(index_set) <= val_size:
                    val += index_set
                    val_scaffold_count += 1
                else:
                    test += index_set
                    test_scaffold_count += 1
        
        if add_no_mscaf:
            train += no_ms_index_sets

        split_ids_dict[seed]=[train, val, test]
    
    return split_ids_dict



class ScaffoldFoldAssign(object):
    """Code from https://github.com/melloddy/MELLODDY-TUNER
    """
    priority_cols = [
        "num_rings_delta",
        "has_macrocyle",
        "num_rbonds",
        "num_bridge",
        "num_spiro",
        "has_unusual_ring_size",
        "num_hrings",
        "num_arings",
        "node_smiles",
    ]
    priority_asc = [True, False, True, False, False, False, False, True, True]
    assert len(priority_cols) == len(
        priority_asc
    ), "priority_cols and priorty_asc must have same length"
    nrings_target = 3

    # rdScaffoldNetwork.ScaffoldNetworkParams are hardwired, since the heuristcs are not guaranteed to work with different setup here
    snparams = rdScaffoldNetwork.ScaffoldNetworkParams()
    snparams.flattenIsotopes = True
    snparams.flattenChirality = True
    snparams.includeGenericBondScaffolds = False
    snparams.includeGenericScaffolds = False
    snparams.includeScaffoldsWithAttachments = False  # this needs to be hardwired to False, as we start from Murcko, which has no attachment information
    snparams.includeScaffoldsWithoutAttachments = True  # this needs to hardwred to True,  as we start from Murcko, which has no attachment information
    snparams.pruneBeforeFragmenting = True

    # default constructor expecting all attributes passed as keyword arguments
    def __init__(self, secret, nfolds=5, verbosity=0):
        """Function to create and initialize a SaccolFoldAssign Calculator

        Args:
            secret: secret key used (for fold hashing)
            nfolds: desired number of folds
            verbosity: controlls verbosity
        """

        self.nfolds = nfolds

        self.secret = secret.encode()
        self.verbosity = verbosity

    @classmethod
    def from_param_dict(cls, secret, method_param_dict, verbosity=0):
        """Function to create and initialize a SaccolFoldAssign Calculator

        Args:
            secret: secret key used (for fold hashing)
            verbosity (int): controlls verbosity
            par_dict(dict): dictionary of method parameters
        """
        return cls(secret=secret, **method_param_dict, verbosity=verbosity)

    @staticmethod
    def murcko_scaff_smiles(mol_smiles):
        """Function to clauclate the Murcko scaffold, wrapper around rdkit MurckoScaffold.GetScaffoldForMol

        Args:
            mol_smiles(str): valid smiles of a molecule

        Returns:
            str: smiles string of the Murcko Scaffold

        """
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is not None:
            murcko_smiles = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
            if murcko_smiles == "":
                return None
            else:
                return murcko_smiles
        else:
            raise ValueError("could not parse smiles {}".format(mol_smiles))

    @staticmethod
    def has_unusual_ringsize(mol):
        """Function to check for ringsizes different than 5 or 6

        Args:
            mol(rdkit.Chem.rdchem.Mol): molecule

        Returns:
            bool: boolean indicating whether usnusally sized ring is present

        """

        return (
            len(
                [
                    len(x)
                    for x in mol.GetRingInfo().AtomRings()
                    if len(x) > 6 or len(x) < 5
                ]
            )
            > 0
        )

    @staticmethod
    def has_macrocycle(mol):
        """Function to check for macrocycles with rinsize > 9

        Args:
            mol(rdkit.Chem.rdchem.Mol): molecule

        Returns:
            bool: boolean indicating whether macrocycle is present

        """

        return len([len(x) for x in mol.GetRingInfo().AtomRings() if len(x) > 9]) > 0

    def sn_scaff_smiles(self, murcko_smiles):
        """Function to exctract the preferred scaffold based on Scaffold Tree rules from the scaffold network created from a Murcko scaffold

        Args:
            murcko_smiles(str): valdi smiles string of a Murcko scaffold

        Returns:
            str: smiles string of the preferred scaffold

        """

        if murcko_smiles is None:
            return None
        mol = Chem.MolFromSmiles(murcko_smiles)
        if mol is not None:
            # if the murcko scaffold has less or equal than the targeted number of rings, then the Murcko scaffold is already the sn_scaffold,
            # so no further decomposition is needed
            if Chem.rdMolDescriptors.CalcNumRings(mol) <= self.nrings_target:
                return murcko_smiles
            # otherwise start decomposition
            try:
                sn = rdScaffoldNetwork.CreateScaffoldNetwork([mol], self.snparams)
            except:
                raise ValueError(
                    "failed to calculate scaffold network for {}".format(murcko_smiles)
                )
            # create data fram with n ode smiles
            node_df = pd.DataFrame({"node_smiles": [str(n) for n in sn.nodes]})
            # print("="*10)
            # for i in list(node_df["node_smiles"]):
            #     print(i)
            PandasTools.AddMoleculeColumnToFrame(
                node_df, "node_smiles", "mol", includeFingerprints=False
            )
            node_df["num_rings"] = node_df["mol"].apply(
                Chem.rdMolDescriptors.CalcNumRings
            )
            node_df["num_rings_delta"] = (
                node_df["num_rings"] - self.nrings_target
            ).abs()
            node_df["num_rbonds"] = node_df["mol"].apply(
                Chem.rdMolDescriptors.CalcNumRotatableBonds, strict=False
            )
            node_df["num_hrings"] = node_df["mol"].apply(
                Chem.rdMolDescriptors.CalcNumHeterocycles
            )
            node_df["num_arings"] = node_df["mol"].apply(
                Chem.rdMolDescriptors.CalcNumAromaticRings
            )
            node_df["num_bridge"] = node_df["mol"].apply(
                Chem.rdMolDescriptors.CalcNumBridgeheadAtoms
            )
            node_df["num_spiro"] = node_df["mol"].apply(
                Chem.rdMolDescriptors.CalcNumSpiroAtoms
            )
            node_df["has_macrocyle"] = node_df["mol"].apply(self.has_macrocycle)
            node_df["has_unusual_ring_size"] = node_df["mol"].apply(
                self.has_unusual_ringsize
            )
            node_df.sort_values(
                self.priority_cols, ascending=self.priority_asc, inplace=True
            )
            # display(node_df)
            return node_df.iloc[0]["node_smiles"]
        else:
            raise ValueError(
                "murcko_smiles {} cannot be read by rdkit".format(murcko_smiles)
            )

    def hashed_fold_scaffold(self, sn_smiles):
        """applies hashing to assign scaffold sn_smiles to a fold

        Args:
            sn_smiles(str): smiles of the scaffold network scaffold

        Returns:
            int: fold id
        """
        scaff = str(sn_smiles).encode("ASCII")
        h = hmac.new(self.secret, msg=scaff, digestmod=hashlib.sha256)
        random.seed(h.digest(), version=2)
        return random.randint(0, self.nfolds - 1)

    # this function contaisn  the key functionality
    def calculate_single(self, smiles):
        """Function to calculate a sn_scaffold and fold_id from an individual smiles

        Args:
            smiles (str) : standardized smiles

        Returns:
            Tuple(str, str, int, bool, str) : a tuple of murcko_smiles, sn_scaffold_smiles, fold_id, Success_flag, error_message
        """
        try:
            
            murcko_smiles = self.murcko_scaff_smiles(smiles)
            
            sn_smiles = self.sn_scaff_smiles(murcko_smiles)
            # print(smiles)
            # print(murcko_smiles)
            # print(sn_smiles)
            
            fold_id = self.hashed_fold_scaffold(sn_smiles)
        except ValueError as err:
            return None, None, None, False, str(err)
        return murcko_smiles, sn_smiles, fold_id, True, None


def scaffold_bin_split(train_df, nfold, num_split,  key='intra-fold',entity='Drug'):
    sa = ScaffoldFoldAssign(nfolds=nfold, secret=key)
    df = train_df.copy()
    df[["murcko_smiles", "sn_smiles", "fold_id", "success", "error_message"]]=df.apply(lambda x: sa.calculate_single((x[entity])), axis=1, result_type='expand')

    df['fold_id'] = df['fold_id'].fillna(nfold)  # assign none value to a new fold
    # assert sum(frac) == 1
    kf = KFold(n_splits=num_split, random_state=42, shuffle=True)
    split_ids_dict: dict = {i: [[], [], []] for i in range(1,num_split+1) }

    for fold_i in range(nfold+1):
        sub_df = df.loc[df['fold_id']==fold_i]
        if fold_i == nfold:
            logger.warning(f"no assign fold id:\n{sub_df}")
            for idx in range(num_split):
                split_ids_dict[idx+1][0].extend(sub_df.index.values)  # no fold id into training set
        else:
            for idx, (train_index, valid_index) in enumerate(kf.split(sub_df)):
                split_ids_dict[idx+1][0].extend(sub_df.iloc[train_index].index.values)
                split_ids_dict[idx+1][1].extend(sub_df.iloc[valid_index].index.values)
    
    return split_ids_dict

def random_cv_split(train_df, num_split, random_seed= 42):
    kf = KFold(n_splits=num_split, random_state=random_seed, shuffle=True)
    split_ids_dict: dict = {i: [[], [], []] for i in range(1,num_split+1) }
    for idx, (train_index, valid_index) in enumerate(kf.split(train_df)):
        split_ids_dict[idx+1][0]=list(train_index)
        split_ids_dict[idx+1][1]=list(valid_index)

    return split_ids_dict

def stratified_cv_split(train_df: DataFrame, num_split, random_seed= 42, target_col = 'Y'):
    kf = StratifiedKFold(n_splits=num_split, random_state=random_seed, shuffle=True)
    X = np.arange(len(train_df.index)).reshape(-1,1)
    y = train_df[target_col].values
    split_ids_dict: dict = {i: [[], [], []] for i in range(1,num_split+1) }
    for idx, (train_index, valid_index) in enumerate(kf.split(X, y)):
        split_ids_dict[idx+1][0]=list(train_index)
        split_ids_dict[idx+1][1]=list(valid_index)
    return split_ids_dict


def cluster_cv_split(train_df: DataFrame, num_split, random_seed=42, target_col='Y'):
    cluster_col='ecfp0.5_no'
    kf = StratifiedKFold(n_splits=num_split, random_state=random_seed, shuffle=True)
    split_ids_dict: dict = {i: [[], [], []] for i in range(1,num_split+1) }
    other_dfs=[]
    df_groupby = train_df.groupby(by=cluster_col, as_index=False)
#     display(count_df)
    for idx, group_df in df_groupby:
        min_label_num=10
        for label in group_df[target_col].unique():
            value = np.sum(group_df[target_col]==label)
            if value < min_label_num:
                min_label_num = value
                
        if (len(group_df) > num_split*3) and (min_label_num >num_split):
            train_ids= np.array(group_df.index)
            train_y =  group_df[target_col].values
            for idx, (train_index, valid_index) in enumerate(kf.split(train_ids, train_y)):
                split_ids_dict[idx+1][0].extend(list(train_ids[train_index]))
                split_ids_dict[idx+1][1].extend(list(train_ids[valid_index]))

        else:
            other_dfs.append(group_df)
            
    for idx in range(1, num_split+1):
        print(len(split_ids_dict[idx][0]), len(split_ids_dict[idx][1]),len(split_ids_dict[idx][2]))
    
#     other_dfs = pd.concat(other_dfs, axis=0)
#     display(other_dfs)
    kf_other = KFold(n_splits=num_split, random_state=random_seed, shuffle=True)
    train_ids= np.arange(len(other_dfs)).reshape(-1,1)
    for idx, (train_index, valid_index) in enumerate(kf_other.split(train_ids)):
        split_ids_dict[idx+1][0].extend(list(chain(*[list(other_dfs[idx].index) for idx in train_index])))
        split_ids_dict[idx+1][1].extend(list(chain(*[list(other_dfs[idx].index) for idx in valid_index])))

    
    for idx in range(1, num_split+1):
        print(len(split_ids_dict[idx][0]), len(split_ids_dict[idx][1]),len(split_ids_dict[idx][2]))

    return split_ids_dict


def cluster_gcv_split(train_df: DataFrame, num_split, random_seed=42, target_col='Y'):
    cluster_col = 'ecfp_no'
    kf = StratifiedKFold(n_splits=num_split, random_state=random_seed, shuffle=True)
    split_ids_dict: dict = {i: [[], [], []] for i in range(1, num_split + 1)}
    other_dfs = []
    df_groupby = train_df.groupby(by=cluster_col, as_index=False)
    #     display(count_df)
    large_groups=[]
    small_groups=[]
    for idx, group_df in df_groupby:
        if len(group_df)>=3:
            large_groups.append(group_df)
        else:
            small_groups.append(group_df)


    for idx in range(1, num_split + 1):
        print(len(split_ids_dict[idx][0]), len(split_ids_dict[idx][1]), len(split_ids_dict[idx][2]))

    #     other_dfs = pd.concat(other_dfs, axis=0)
    #     display(other_dfs)
    kf_other = KFold(n_splits=num_split, random_state=random_seed, shuffle=True)
    train_ids = np.arange(len(other_dfs)).reshape(-1, 1)
    for idx, (train_index, valid_index) in enumerate(kf_other.split(train_ids)):
        split_ids_dict[idx + 1][0].extend(list(chain(*[list(other_dfs[idx].index) for idx in train_index])))
        split_ids_dict[idx + 1][1].extend(list(chain(*[list(other_dfs[idx].index) for idx in valid_index])))

    for idx in range(1, num_split + 1):
        print(len(split_ids_dict[idx][0]), len(split_ids_dict[idx][1]), len(split_ids_dict[idx][2]))

    return split_ids_dict


def cluster_mix_split(train_df: DataFrame, num_split, random_seed=42, target_col='Y'):
    cluster_col = 'fragfp_no'
    kf = StratifiedKFold(n_splits=num_split, random_state=random_seed, shuffle=True)
    split_ids_dict: dict = {i: [[], [], []] for i in range(1, num_split + 1)}
    other_dfs = []
    df_groupby = train_df.groupby(by=cluster_col, as_index=False)
    #     display(count_df)
    for idx, group_df in df_groupby:
        min_label_num = 10
        for label in group_df[target_col].unique():
            value = np.sum(group_df[target_col] == label)
            if value < min_label_num:
                min_label_num = value

        if (len(group_df) > num_split * 3) and (min_label_num > num_split):
            train_ids = np.array(group_df.index)
            train_y = group_df[target_col].values
            for idx, (train_index, valid_index) in enumerate(kf.split(train_ids, train_y)):
                split_ids_dict[idx + 1][0].extend(list(train_ids[train_index]))
                split_ids_dict[idx + 1][1].extend(list(train_ids[valid_index]))

        else:
            other_dfs.append(group_df)

    for idx in range(1, num_split + 1):
        print(len(split_ids_dict[idx][0]), len(split_ids_dict[idx][1]), len(split_ids_dict[idx][2]))

    #     other_dfs = pd.concat(other_dfs, axis=0)
    #     display(other_dfs)
    kf_other = KFold(n_splits=num_split, random_state=random_seed, shuffle=True)
    train_ids = np.arange(len(other_dfs)).reshape(-1, 1)
    for idx, (train_index, valid_index) in enumerate(kf_other.split(train_ids)):
        split_ids_dict[idx + 1][0].extend(list(chain(*[list(other_dfs[idx].index) for idx in train_index])))
        split_ids_dict[idx + 1][1].extend(list(chain(*[list(other_dfs[idx].index) for idx in valid_index])))

    for idx in range(1, num_split + 1):
        print(len(split_ids_dict[idx][0]), len(split_ids_dict[idx][1]), len(split_ids_dict[idx][2]))

    return split_ids_dict


def cluster_sample_split(train_df: DataFrame, num_split, sample_num=1, random_seed=42, target_col='Y'):
    cluster_col='fragfp_no'
    kf = StratifiedKFold(n_splits=num_split, random_state=random_seed, shuffle=True)
    split_ids_dict: dict = {i: [[], [], []] for i in range(1,num_split+1) }
    other_dfs=[]
    df_groupby = train_df.groupby(by=cluster_col, as_index=False)
#     display(count_df)
    for idx, group_df in df_groupby:
        for label in group_df[target_col].unique():
            tmp_df = group_df.query(f'{target_col} == {label}')
            for idx in range(num_split):
                if len(tmp_df.index)>sample_num:
                    sample_idx = tmp_df.sample(n=sample_num, random_state=idx).index
                    split_ids_dict[idx+1][0].extend(set(sample_idx))
                    split_ids_dict[idx+1][1].extend(set(tmp_df.index)-set(sample_idx))
                else:
                    split_ids_dict[idx+1][0].extend(set(tmp_df.index))

    return split_ids_dict