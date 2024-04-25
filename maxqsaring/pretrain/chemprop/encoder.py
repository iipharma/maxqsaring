import torch
from torch import nn
import os
CUR_DIR = os.path.dirname(__file__)
from chemprop.data import  MoleculeDataset


class ChempropPretrainRepr(nn.Module):
    ## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/examples/property_prediction/moleculenet/utils.py#L76
    def __init__(self, model_name: str, *args):
        super(ChempropPretrainRepr, self).__init__()
        from chemprop.utils import load_checkpoint
        from chemprop.train.molecule_fingerprint import model_fingerprint
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.join(CUR_DIR, f'models/{model_name}.pt')
        self.model =load_checkpoint(model_path, device=self.device)
        self.generator = model_fingerprint

    def forward(self, batch):
        batch: MoleculeDataset
        batch_feats = batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_descriptors(), batch.bond_features()
        graph_feats = self.model.fingerprint(*batch_feats, 'last_FFN')
        return graph_feats