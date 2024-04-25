from torch.utils import data
import dgl
from rdkit import Chem
import numpy as np

def gin_collate_fn(batch):
    batch = dgl.batch(batch)
    return batch

def get_valid_smiles(smiList: list):
    valid_smiList = []
    valid_ids=[]
    for idx, smi in enumerate(smiList):
        mol = Chem.MolFromSmiles(smi)
        if mol and (mol.GetNumHeavyAtoms() > 0):
            valid_smiList.append(smi)
            valid_ids.append(idx)
    
    return valid_smiList, np.array(valid_ids)


class SmilesDataset(data.Dataset):
    def __init__(self, smiList: list, **kwargs):
        from dgllife.utils import smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
        self.node_featurizer = PretrainAtomFeaturizer()
        self.edge_featurizer = PretrainBondFeaturizer()
        from functools import partial
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        self.smiList, self.valid_ids = get_valid_smiles(smiList)

    def __len__(self): 
        return len(self.smiList)
    
    def __getitem__(self, index):
        X = self.smiList[index]
        X = self.fc(smiles=X, node_featurizer = self.node_featurizer, edge_featurizer = self.edge_featurizer)
        return X
       