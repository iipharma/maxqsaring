import os

CUR_DIR = os.path.dirname(__file__)

MODEL_BASE_CKPT_PATH = os.path.join(CUR_DIR,'models/pretrain_k_bert_epoch_7.pth')
MODEL_BASE_CKPT_PATH_WCL = os.path.join(CUR_DIR,'models/pretrain_k_bert_wcl_epoch_7.pth')
MODEL_BASE_CKPT_PATH_CHIRAL = os.path.join(CUR_DIR,'models/pretrain_k_bert_chirality_epoch_7.pth')
MODEL_BASE_CKPT_PATH_CHIRAL_RS = os.path.join(CUR_DIR,'models/pretrain_k_bert_chirality_R_S_epoch_7.pth')
