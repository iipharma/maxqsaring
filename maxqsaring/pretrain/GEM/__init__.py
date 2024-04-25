import os
CUR_DIR = os.path.dirname(__file__)

GEM1_ENCODER_CONFIG={    
    "atom_names": ["atomic_num", "formal_charge", "degree", 
        "chiral_tag", "total_numHs", "is_aromatic", 
        "hybridization"],
    "bond_names": ["bond_dir", "bond_type", "is_in_ring"],
    "bond_float_names": ["bond_length"],
    "bond_angle_float_names": ["bond_angle"],
    "embed_dim": 32,
    "dropout_rate": 0.5,
    "layer_num": 8,
    "readout": "sum"
}
PRETRAIN_MODEL_CLS_FN = os.path.join(CUR_DIR,'models/class.pdparams')
PRETRAIN_MODEL_REG_FN = os.path.join(CUR_DIR,'models/regr.pdparams')