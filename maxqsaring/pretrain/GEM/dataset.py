from torch.utils import data
from pahelix.utils.compound_tools import mol_to_geognn_graph_data_MMFF3d
from rdkit.Chem import AllChem
from . import GEM1_ENCODER_CONFIG
import pgl
import numpy as np



class DownstreamCollateFn(object):
    """CollateFn for downstream model"""
    def __init__(self, 
            atom_names, 
            bond_names, 
            bond_float_names,
            bond_angle_float_names):
        self.atom_names = atom_names
        self.bond_names = bond_names
        self.bond_float_names = bond_float_names
        self.bond_angle_float_names = bond_angle_float_names

    def _flat_shapes(self, d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])
    
    def __call__(self, data_list):
        """
        Collate features about a sublist of graph data and return join_graph, 
        masked_node_indice and masked_node_labels.
        Args:
            data_list : the graph data in gen_features.for data in data_list,
            create node features and edge features according to pgl graph,and then 
            use graph wrapper to feed join graph, then the label can be arrayed to batch label.
        Returns:
            The batch data contains finetune label and valid,which are 
            collected from batch_label and batch_valid.  
        """
        atom_bond_graph_list = []
        bond_angle_graph_list = []
        for data in data_list:
            ab_g = pgl.Graph(
                    num_nodes=len(data[self.atom_names[0]]),
                    edges=data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names + self.bond_float_names})
            ba_g = pgl.Graph(
                    num_nodes=len(data['edges']),
                    edges=data['BondAngleGraph_edges'],
                    node_feat={},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_angle_float_names})
            atom_bond_graph_list.append(ab_g)
            bond_angle_graph_list.append(ba_g)

        atom_bond_graph = pgl.Graph.batch(atom_bond_graph_list)
        bond_angle_graph = pgl.Graph.batch(bond_angle_graph_list)
        # TODO: reshape due to pgl limitations on the shape
        self._flat_shapes(atom_bond_graph.node_feat)
        self._flat_shapes(atom_bond_graph.edge_feat)
        self._flat_shapes(bond_angle_graph.node_feat)
        self._flat_shapes(bond_angle_graph.edge_feat)
        return atom_bond_graph, bond_angle_graph
        
gem_collate_fn = DownstreamCollateFn(
    atom_names=GEM1_ENCODER_CONFIG['atom_names'],
    bond_names=GEM1_ENCODER_CONFIG['bond_names'],
    bond_float_names=GEM1_ENCODER_CONFIG['bond_float_names'],
    bond_angle_float_names = GEM1_ENCODER_CONFIG['bond_angle_float_names']
)


def get_valid_smiles(smiList: list):
    valid_smiList = []
    valid_ids=[]
    for idx, smi in enumerate(smiList):
        mol = AllChem.MolFromSmiles(smi)
        if mol and mol.GetNumAtoms()>0:
            valid_smiList.append(AllChem.MolToSmiles(mol))
            valid_ids.append(idx)
        else:
            print(f'Error smiles: {smi}')

    return valid_smiList, np.array(valid_ids)


class SmilesDataset(data.Dataset):
    def __init__(self, smiList: list, **kwargs):
        self.smiList, self.valid_ids= get_valid_smiles(smiList)
        
    def __len__(self): 
        return len(self.smiList)
    
    def __getitem__(self, index):
        smi = self.smiList[index]
        try:
            mol = AllChem.MolFromSmiles(smi)
            data = mol_to_geognn_graph_data_MMFF3d(mol)
            data['smiles'] = smi
        except:
            print(f'Error featuerized smiles: {index}: {smi}')
            data =None
        return data