from torch.utils import data
import dgl
from rdkit import Chem
import numpy as np
from maxqsaring.external.grover.data import MoleculeDatapoint, MolCollator
from argparse import Namespace


new_args= Namespace()
new_args.bond_drop_rate = 0
new_args.no_cache = True
grover_collate_fn= MolCollator(args=new_args, shared_dict={})

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
        self.smiList, self.valid_ids = get_valid_smiles(smiList)

    def __len__(self): 
        return len(self.smiList)
    
    def __getitem__(self, index):
        X = self.smiList[index]
        return MoleculeDatapoint(line=[X], args=None,
                features=None, use_compound_names=None)
       