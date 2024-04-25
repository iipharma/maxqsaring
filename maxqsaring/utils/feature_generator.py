import numpy as np
from maxqsaring.logging_helper import create_logger
from pandas import DataFrame
import pandas as pd
from pathlib import Path
logger = create_logger('FeatGenerator')

from maxqsaring.featurizer import MolFPEncoder
from concurrent.futures import ProcessPoolExecutor
from maxqsaring.feat_enum import FeatNameToKey


FP_ENCODER={ }

def smiles_to_fps(fp_name: str, smi):
    fps = FP_ENCODER[fp_name].featurize(smi)
    return fps

def parallel_smiles_to_fps(fp_name, smiList):
    fp_res = []
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(smiles_to_fps, fp_name, smi) for smi in smiList]
        for future in futures:
            fp_res.append(future.result())
    
    return fp_res

def gen_features(df: DataFrame, processed_dir: Path, fp_names:list, smiles_col='Drug', target_col : str = 'Y', mode='train'):
    df_total = df.copy()

    for fp_name in fp_names:
        feat_fn = processed_dir / f'features/{mode}feats-{fp_name}.pkl'
        feat_fn.parent.mkdir(parents=True, exist_ok=True)
        if feat_fn.exists() and (not mode.startswith('temp')):
            # df_total.append(pd.read_pickle(feat_fn))
            logger.info(f'{feat_fn} is found.')
            df_total[fp_name] = pd.read_pickle(feat_fn)[fp_name]
        else:
            logger.info(f'Calc features: {fp_name}...')
            if 'pretrain_' in fp_name[:11]: # need to model for inference
                fpInference = MolFPEncoder(fp_name=fp_name)
                df_total[fp_name] = fpInference.calc(df[smiles_col].values)
                # del fpInference
            else:
                if '_rad' in fp_name[-7:]:
                    # FPEncoder.calc = FPEncoder._load_func(fp_name=fp_name, radius=int(fp_name[-1]))
                    FP_ENCODER[fp_name] = MolFPEncoder(fp_name=fp_name, radius=int(fp_name[-1]))
                else: 
                    FP_ENCODER[fp_name] = MolFPEncoder(fp_name=fp_name)

                df_total[fp_name] = parallel_smiles_to_fps(fp_name, df_total[smiles_col].values)
                del FP_ENCODER[fp_name]
            
            # save features
            df_total.to_pickle(str(feat_fn))
            logger.info(f'Saved into {feat_fn}')
    
    key_names = fp_names.copy()
    if target_col:
        key_names += [target_col]
    return df_total[key_names]
    