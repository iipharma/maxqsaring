import torch
from torch import nn
from dgl.nn.pytorch import Sequential
from dgllife.model.model_zoo import GINPredictor
import os
CUR_DIR = os.path.dirname(__file__)

# class DglGinPretrainLayer(nn.Module):
#     def __init__(self, model_name: str) -> None:
#         super(DglGinPretrainLayer, self).__init__()
#         from dgllife.model import load_pretrained
#         gin_model = load_pretrained(model_name)
        
#         self.gnn_layer = nn.Sequential(*[x[1] for x in gin_model.named_children() if x[0] not in ['readout', 'predict']])
#         # print(self.gnn_layer)

#     def forward(self, bg):
#         # node_feats = [
#         #     bg.ndata.pop('atomic_number'),
#         #     bg.ndata.pop('chirality_type')
#         # ]
#         # edge_feats = [
#         #     bg.edata.pop('bond_type'),
#         #     bg.edata.pop('bond_direction_type')
#         # ]
#         node_feats = self.gnn_layer(bg, node_feats, edge_feats)

#         return node_feats


# def DglGinPretrainLayer(model_name: str):
#     from dgllife.model import load_pretrained
#     gin_model = load_pretrained(model_name)
#     gnn_layer = Sequential(*[x[1] for x in gin_model.named_children() if x[0] not in ['readout', 'predict']])
#     return gnn_layer

def load_new_pretrain_model(model_path):
    if "infomax" in model_path:
        jk, readout='max', 'mean'
    elif "masking" in model_path:
        jk, readout='last', 'sum'
    elif "contextpred" in model_path:
        jk, readout = 'concat', 'sum'
    else:
        jk, readout = 'concat', 'sum'

    model = GINPredictor(
            num_node_emb_list=[120, 3],
            num_edge_emb_list=[6, 3],
            num_layers=5,
            emb_dim=300,
            JK=jk,
            dropout=0.5,
            readout=readout,
            n_tasks=1
        )
    model.gnn.JK = jk

    checkpoint = torch.load(model_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print('Pretrained model loaded')
    return model


class DglGinPretrainRepr(nn.Module):
    ## adapted from https://github.com/awslabs/dgl-lifesci/blob/2fbf5fd6aca92675b709b6f1c3bc3c6ad5434e96/examples/property_prediction/moleculenet/utils.py#L76
    def __init__(self, model_name: str):
        super(DglGinPretrainRepr, self).__init__()
        
        from dgl.nn.pytorch.glob import AvgPooling, SumPooling
        ## this is fixed hyperparameters as it is a pretrained model
        from dgllife.model import load_pretrained
        model_path = os.path.join(CUR_DIR, f'models/{model_name}.pth')
        if os.path.exists(model_path):
            self.gnn = load_new_pretrain_model(model_path)
        else:
            self.gnn = load_pretrained(model_name)

        if hasattr(self.gnn, 'gnn'):
            self.gnn= self.gnn.gnn
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.gnn = DglGinPretrainLayer(model_name)
        self.readout_avg = AvgPooling()
        self.readout_sum = SumPooling()

        self.gnn.to(self.device)
        self.gnn.eval()

    def forward(self, bg):
        bg = bg.to(self.device)
        node_feats = [
            bg.ndata.pop('atomic_number'),
            bg.ndata.pop('chirality_type')
        ]
        edge_feats = [
            bg.edata.pop('bond_type'),
            bg.edata.pop('bond_direction_type')
        ]
        node_feats = self.gnn(bg, node_feats, edge_feats)
        graph_feats = torch.cat([self.readout_avg(bg, node_feats), self.readout_sum(bg, node_feats)],dim=1)
        # graph_feats = self.readout_sum(bg, node_feats)
        return graph_feats