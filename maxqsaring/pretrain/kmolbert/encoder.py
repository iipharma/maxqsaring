import torch
from torch import nn
from maxqsaring.logging_helper import create_logger
from .base import BERT_atom_embedding_generator, load_model_weights
from . import MODEL_BASE_CKPT_PATH, MODEL_BASE_CKPT_PATH_WCL, MODEL_BASE_CKPT_PATH_CHIRAL, MODEL_BASE_CKPT_PATH_CHIRAL_RS

logger = create_logger(__name__)

def get_model_args(model_path):
    args = {}
    args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    args['metric_name'] = 'roc_auc'
    args['batch_size'] = 128
    args['num_epochs'] = 200
    args['d_model'] = 768
    args['n_layers'] = 6
    args['vocab_size'] = 47
    args['maxlen'] = 201
    args['d_k'] = 64
    args['d_v'] = 64
    args['d_ff'] = 768 * 4
    args['n_heads'] = 12
    args['global_labels_dim'] = 154
    args['atom_labels_dim'] = 15
    args['lr'] = 3e-5
    args['pretrain_layer'] = 'all_12layer'
    args['pretrain_model'] = model_path
    args['use_atom'] = False
    return args
    

class KMolBertPretrainRepr(nn.Module):
    def __init__(self, model_type, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # global_labels_dim=154
        if model_type == 'kBert_base':
            model_path = MODEL_BASE_CKPT_PATH
            global_labels_dim=154
        elif model_type == 'kBert_wcl':
            model_path = MODEL_BASE_CKPT_PATH_WCL
            global_labels_dim=135
        elif model_type == 'kBert_chiral':
            model_path = MODEL_BASE_CKPT_PATH_CHIRAL
            global_labels_dim = 154
        elif model_type == 'kBert_chiral_rs':
            model_path = MODEL_BASE_CKPT_PATH_CHIRAL_RS
            global_labels_dim = 1
        else:
            raise ValueError

        params = get_model_args(model_path)
        # global_labels_dim = args['global_labels_dim']
        self.model = BERT_atom_embedding_generator(d_model=params['d_model'],
                                                   n_layers=params['n_layers'],
                                                   vocab_size=params['vocab_size'],
                                                   maxlen=params['maxlen'],
                                                   d_k=params['d_k'],
                                                   d_v=params['d_v'],
                                                   n_heads=params['n_heads'],
                                                   d_ff=params['d_ff'],
                                                   global_label_dim=global_labels_dim,
                                                   atom_label_dim=params['atom_labels_dim'],
                                                   use_atom=params['use_atom'])
        self.model = load_model_weights(self.model, params)
        logger.info(f'Model weights loaded successfully.')
        self.model.to(params['device'])
        self.device= params['device']
        
    def forward(self, *inputs):
        # print(inputs)
        token_idx, atom_labels, atom_mask_index = inputs
        # print(token_idx)
        token_idx = token_idx.long().to(self.device)
        # print(token_idx)
        h_global, h_atom = self.model(token_idx, atom_mask_index)
        return h_global