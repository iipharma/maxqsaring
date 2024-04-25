import torch
from torch import nn
from maxqsaring.external.grover.util.utils import load_checkpoint, create_logger
from . import MODEL_BASE_CKPT_PATH, MODEL_LARGE_CKPT_PATH
from argparse import Namespace

logger = create_logger('GROVER', quiet=False)

class GroverPretrainRepr(nn.Module):
    def __init__(self, model_type: str, is_cuda=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        new_args = Namespace()
        new_args.parser_name = "fingerprint"
        new_args.fingerprint_source = 'both'
        new_args.cuda=is_cuda
        new_args.dropout = 0.0

        if model_type == 'grover_large':
            self.model = load_checkpoint(MODEL_LARGE_CKPT_PATH, current_args=new_args, cuda=is_cuda, logger= logger)
        elif model_type == 'grover_base':
            self.model = load_checkpoint(MODEL_BASE_CKPT_PATH, current_args=new_args, cuda=is_cuda, logger= logger)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model.to(self.device)
        # self.model.eval()

        
    def forward(self, *args):
        graph_repr = self.model(*args)
        return graph_repr

