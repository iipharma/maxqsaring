import torch
from torch import nn
from maxqsaring.external.pahelix.model_zoo.gem_model import GeoGNNModel
import paddle
from . import GEM1_ENCODER_CONFIG, PRETRAIN_MODEL_CLS_FN, PRETRAIN_MODEL_REG_FN

class Gemv1PretrainRepr(nn.Module):
    def __init__(self, model_type: str):
        super(Gemv1PretrainRepr, self).__init__()
        
        self.gnn_model = GeoGNNModel(GEM1_ENCODER_CONFIG)
        
        if model_type.lower().startswith('gemv1_class'):
            self.gnn_model.set_state_dict(paddle.load(PRETRAIN_MODEL_CLS_FN))

        if model_type.lower().startswith('gemv1_reg'):
            self.gnn_model.set_state_dict(paddle.load(PRETRAIN_MODEL_REG_FN))

        self.gnn_model.eval()

    def forward(self, bg):
        atom_bond_graphs, bond_angle_graphs = bg
        atom_bond_graphs = atom_bond_graphs.tensor()
        bond_angle_graphs = bond_angle_graphs.tensor()
        node_repr, edge_repr, graph_repr = self.gnn_model(atom_bond_graphs, bond_angle_graphs)        
        return graph_repr