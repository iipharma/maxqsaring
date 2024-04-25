
import re
from rdkit import Chem
import numpy as np
import torch
from torch.utils import data


pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
REGEX = re.compile(pattern)

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    tokens = [token for token in REGEX.findall(smi)]
    # assert smi == ''.join(tokens)
    # return ' '.join(tokens)
    return tokens

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_labels(atom, use_chirality=True):
    results = one_of_k_encoding(atom.GetDegree(),
                                [0, 1, 2, 3, 4, 5, 6]) + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()] \
              + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]
    atom_labels_list = np.array(results).tolist()
    atom_selected_index = [1, 2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 17, 19, 20, 21]
    atom_labels_selected = [atom_labels_list[x] for x in atom_selected_index]
    return atom_labels_selected

ATOM_TOKEN_LIST = ['c', 'C', 'O', 'N', 'n', '[C@H]', 'F', '[C@@H]', 'S', 'Cl', '[nH]', 's', 'o', '[C@]',
                    '[C@@]', '[O-]', '[N+]', 'Br', 'P', '[n+]', 'I', '[S+]',  '[N-]', '[Si]', 'B', '[Se]', '[other_atom]']
ALL_TOKEN_LIST = ['[PAD]', '[GLO]', 'c', 'C', '(', ')', 'O', '1', '2', '=', 'N', '3', 'n', '4', '[C@H]', 'F', '[C@@H]', '-', 'S', '/', 'Cl', '[nH]', 's', 'o', '5', '#', '[C@]', '[C@@]', '\\', '[O-]', '[N+]', 'Br', '6', 'P', '[n+]', '7', 'I', '[S+]', '8', '[N-]', '[Si]', 'B', '9', '[2H]', '[Se]', '[other_atom]', '[other_token]']

word2idx = {w: i for i, w in enumerate(ALL_TOKEN_LIST)}

def construct_input_from_smiles(smiles, max_len=200):
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    token_list = smi_tokenizer(smiles)
    # print("testsmiles:",smiles)
    padding_list = ['[PAD]' for _ in range(max_len-len(token_list))]
    tokens = ['[GLO]'] + token_list + padding_list
    mol = Chem.MolFromSmiles(smiles)
    atom_example = mol.GetAtomWithIdx(0)
    atom_labels_example = atom_labels(atom_example)
    atom_mask_labels = [2 for _ in range(len(atom_labels_example))]
    atom_labels_list = []
    atom_mask_list = []

    index = 0
    tokens_idx = []
    for i, token in enumerate(tokens):
        if token in ATOM_TOKEN_LIST:
            atom = mol.GetAtomWithIdx(index)
            an_atom_labels = atom_labels(atom)
            atom_labels_list.append(an_atom_labels)
            atom_mask_list.append(1)
            index = index + 1
            tokens_idx.append(word2idx[token])
        else:
            if token in ALL_TOKEN_LIST:
                atom_labels_list.append(atom_mask_labels)
                tokens_idx.append(word2idx[token])
                atom_mask_list.append(0)
            elif '[' in list(token):
                atom = mol.GetAtomWithIdx(index)
                tokens[i] = '[other_atom]'
                an_atom_labels = atom_labels(atom)
                atom_labels_list.append(an_atom_labels)
                atom_mask_list.append(1)
                index = index + 1
                tokens_idx.append(word2idx['[other_atom]'])
            else:
                tokens[i] = '[other_token]'
                atom_labels_list.append(atom_mask_labels)
                tokens_idx.append(word2idx['[other_token]'])
                atom_mask_list.append(0)

    tokens_idx = [word2idx[x] for x in tokens]
    if len(tokens_idx) == max_len + 1:
        return tokens_idx, atom_labels_list, atom_mask_list
    else:
        return [0], [0], [0]

def kmolbert_collate_fn(data):
    token_idx, global_label_list, mask = map(list, zip(*data))
    tokens_idx = torch.tensor(token_idx, dtype=torch.int32)
    global_label = torch.tensor(global_label_list)
    mask = torch.tensor(mask,dtype=torch.int32)
    return tokens_idx, global_label, mask


def get_valid_smiles(smiList: list):
    valid_smiList = []
    valid_ids = []
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
        smi = self.smiList[index]
        res = construct_input_from_smiles(smi)
        return res


if __name__ == '__main__':
    # smi=' NC1=NC2(CO1)c1cc(NC(=O)c3ncc(Cl)cc3F)ccc1OCC21CC1'
    smi=' [H]C1([H])OC(N)=NC12c1cc(-c3cncnc3)ccc1CC21CCC(OC)C(C)C1'
    res=construct_input_from_smiles(smi)
    print(res)