import os
import sys
sys.path.append('.')
sys.path.append('./maxqsaring/external')
import click
from tdc.benchmark_group import admet_group
from enum import Enum
from pathlib import Path
from maxqsaring.modelling import TaskModelling
from maxqsaring.utils.util import select_bestfeat_and_test, run_one_generation
from maxqsaring.utils.metrics import get_metric
from maxqsaring.utils.mol_cleaner import clean_smi
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

from maxqsaring.logging_helper import create_logger
logger = create_logger(__name__)
# DATA_ROOT= '/workspace/projects/chemautoxgb-dev/data/'
DATA_ROOT= os.environ.get('DATA_ROOT_DIR', None)
assert DATA_ROOT is not None

class TaskFeatSet(Enum):
    """Feature combinations for 22 tasks from TDC benchmark

    Args:
        Enum (_type_): _description_
    """
    caco2_wang = 'rdkit2d_deepchem,morganfp_featureCounts_rdkit_rad2'
    hia_hou = 'property,pretrain_chemprop_supervised_scaffold_calclipo,autocorr'
    pgp_broccatelli = 'rdkit2dnorm_chemprop,rdkitfp,pretrain_chemprop_supervised_scaffold_explipo'
    bioavailability_ma= 'fragment,pretrain_grover_large,rdkit2dnorm_chemprop,property'
    lipophilicity_astrazeneca= 'rdkit2d_deepchem,pretrain_chemprop_supervised_scaffold_explipo,property,pharmafp_erg'
    solubility_aqsoldb = 'mordredfp_2d_deepchem,avalonfp'
    ppbr_az = 'rdkit2d_deepchem,pretrain_chemprop_supervised_random_explipo,morganfp_chiral_rdkit_rad1'
    vdss_lombardo = 'circlefp_deepchem_rad1,fragment,morganfp_featureCounts_rdkit_rad3,pharmafp_erg,rdkit2dnorm_chemprop'
    cyp2c9_veith = 'pretrain_gin_supervised_infomax,pretrain_gin_supervised_contextpred,property,pretrain_gin_supervised_masking,mold2'
    cyp2d6_veith = 'pretrain_gin_supervised_masking,pretrain_gin_supervised_infomax'
    cyp3a4_veith = 'pretrain_gin_supervised_infomax,pretrain_gin_supervised_masking,atompairfp'
    cyp2c9_substrate_carbonmangels = 'pretrain_gin_supervised_contextpred,morganfp_chiralCounts_rdkit_rad1,property'
    cyp2d6_substrate_carbonmangels = 'estatefp,constitution'
    cyp3a4_substrate_carbonmangels = 'morganfp_feature_rdkit_rad3,charges,circlefp_feature_deepchem_rad1'
    half_life_obach = 'pretrain_gin_supervised_edgepred,pretrain_chemprop_supervised_scaffold_cyp2c9sub,pharmafp_base_short,autocorr'
    clearance_hepatocyte_az = 'pretrain_gin_supervised_edgepred,pharmafp_erg,charges'
    clearance_microsome_az = 'pretrain_gin_supervised_contextpred,rdkit2d_chemprop'
    ld50_zhu = 'pretrain_chemprop_supervised_scaffold_ld50,maccsfp_deepchem,charges'
    herg = 'morganfp_featureCounts_rdkit_rad2,charges'
    ames = 'mordredfp_2d_deepchem,morganfp_chiralCounts_rdkit_rad2,mold2'
    dili = 'pretrain_gin_supervised_edgepred,estatefp,rdkit2d_deepchem,constitution'
    herg_kraim={
        "default": 'mordredfp_2d_deepchem,pubchemfp,mold2',
        'scaffold': 'mordredfp_2d_deepchem,pubchemfp,mold2',
        'random-cv': 'atompairfp,rdkit2d_chemprop,morganfp_chiralCounts_rdkit_rad3,morganfp_basic_rdkit_rad3'
    }

class TaskMetaInfo(Enum):
    """Meta infromation of 22 tasks among TDC benckmark

    Args:
        Enum (_type_): _description_
    """
    caco2_wang = 'regression', 'mae', None
    hia_hou = 'classification', 'auc', None
    pgp_broccatelli = 'classification', 'auc', None
    bioavailability_ma = 'classification','auc', None
    lipophilicity_astrazeneca = 'regression', 'mae', None
    logd_new = 'regression', 'mae', None
    solubility_aqsoldb = 'regression', 'mae', None
    bbb_martins = 'classification','auc', None
    ppbr_az = 'regression', 'mae', None
    vdss_lombardo = 'regression', 'mae', 'log10'
    cyp2c9_veith = 'classification','auc', None
    cyp2d6_veith = 'classification','auc', None
    cyp3a4_veith = 'classification','auc', None
    cyp2c9_substrate_carbonmangels =  'classification', 'auc', None
    cyp2d6_substrate_carbonmangels = 'classification', 'auc', None
    cyp3a4_substrate_carbonmangels = 'classification', 'auc', None
    half_life_obach = 'regression', 'mae', 'log10'
    clearance_hepatocyte_az = 'regression', 'mae', 'logExp_zscore'
    clearance_microsome_az = 'regression', 'mae', 'log10'
    ld50_zhu = 'regression', 'mae', None
    herg = 'classification', 'auc', None
    ames = 'classification', 'auc', None
    dili = 'classification', 'auc', None
    
@click.group()
def cli1():
    pass
        
@cli1.command('train')
@click.option("--task_name", '-tn', type = str, default='cyp2c9_veith')
@click.option("--split", '-s', type = str, default='scaffold')
@click.option("--tmp_dir", '-d', type = str,  default=None)
@click.option("--num_gen", '-ng', type = int,  default=10)
def train(**kwargs):
    task_name = kwargs['task_name']
    split_method = kwargs['split']
    num_gen = kwargs['num_gen']
    assert num_gen >= 1
    meta_info = list(getattr(TaskMetaInfo, task_name).value) + [split_method]  # task_class, metric_name, norm_y, split_method
    group = admet_group(path = DATA_ROOT)
    benchmark = group.get(task_name)
    train_val, test = benchmark['train_val'], benchmark['test']
    logger.info(f'train_num: {len(train_val.index)}; test num: {len(test.index)}')

    # build models
    modelapi = TaskModelling(task_name, *meta_info, tmp_dir = kwargs['tmp_dir'])
    best_feat_name = select_bestfeat_and_test(modelapi, train_val, num_gen, task_name, *meta_info)

    # eval models
    modelapi.eval_testset(test, best_feat_name)

@click.group()
def cli3():
    pass

@cli3.command('eval')
@click.option("--task_name", '-tn', type = str, default='cyp2c9_veith')
@click.option("--version_tag", '-tag', type = str, default='default')
@click.option("--split", '-s', type = str, default='scaffold')
@click.option("--tmp_dir", '-d', type = str,  default=None)
@click.option("--restart", '-rs', type = bool,  default=False)
def eval(**kwargs):
    """Evaluation for test data from TDC ADMET group
    """
    task_name = kwargs['task_name']
    split = kwargs['split']
    meta_info= getattr(TaskMetaInfo, task_name).value
    fp_names = getattr(TaskFeatSet, task_name).value

    if isinstance(fp_names, dict):
        fp_names = fp_names[kwargs['version_tag']]

    group = admet_group(path = DATA_ROOT)
    benchmark = group.get(task_name)
    train_val, test = benchmark['train_val'], benchmark['test']
    logger.info(f'train_num: {len(train_val.index)}; test num: {len(test.index)}')
    modelapi = TaskModelling(task_name, *meta_info, tmp_dir = kwargs['tmp_dir'])

    try:
        assert kwargs['restart']==False
        test_preds, drop_ids = modelapi.eval_testset(test, fp_names)
    except:
        logger.info(f'{"="*20} Re-Modelling...')
        modelapi.split(train_val, split)
        modelapi.gen_featsets(fp_names)  # generate features
        modelapi.eval_featsets(fp_names, restart=kwargs['restart']) # build model with features
        test_preds, drop_ids = modelapi.eval_testset(test, fp_names) # evaluate the test data with built models

    if len(drop_ids) >0:
        test.drop(drop_ids, axis=0, inplace=True)
    pred_names = ['mean', 'std']+ [f'M{i+1}' for i in range(test_preds.shape[1]-2)]
    test[pred_names] = test_preds

    output_fn = f'{DATA_ROOT}/admet_group/{task_name}/test_predict.csv'
    test.to_csv(output_fn, index=False)
    logger.info(f'Saved into {output_fn}')

    logger.info(f'{"="*20} Evaluation is finished. Good luck!')


@click.group()
def cli2():
    pass

@cli2.command('predict')
@click.option("--task_name", '-tn', type = str, default='cyp2c9_veith')
@click.option("--version_tag", '-tag', type = str, default='default')
@click.option("--tmp_dir", '-d', type = str,  default=None)
@click.option("--test_file", '-tf', type = str, default='tmp/test.csv')
@click.option("--smiles_col", '-sc', type = str, default='smiles')
@click.option("--target_col", '-tc', type = str, default=None)
@click.option("--eval_rank_col", '-erc', type = str, default=None)
@click.option("--eval_class_col", '-ecc', type = str, default=None)
@click.option("--clean_smi", '-cs', type = bool, default=False)
@click.option("--mode_prefix", '-mp', type = str, default='temp')
def predict(**kwargs):
    """Predict externel files 
    """
    task_name = kwargs['task_name']
    meta_info= getattr(TaskMetaInfo, task_name).value
    fp_names = getattr(TaskFeatSet, task_name).value
    mode_prefix =kwargs['mode_prefix']

    if isinstance(fp_names, dict):
        fp_names = fp_names[kwargs['version_tag']]

    modelapi = TaskModelling(task_name, *meta_info, tmp_dir = kwargs['tmp_dir'])
    test_fn = Path(kwargs['test_file'])
    test_df = pd.read_csv(test_fn)
    smiles_col = kwargs["smiles_col"]
    if kwargs['clean_smi']:
        test_df['std_smiles']=test_df[smiles_col].apply(clean_smi)
    else:
        test_df['std_smiles']=test_df[smiles_col]
    
    if 'Drug' in test_df.columns:
        del test_df['Drug']
    
    if kwargs.get('target_col'):
        target_col = kwargs["target_col"]
        test_df.rename(columns={'std_smiles': 'Drug', target_col: 'Y'}, inplace=True)
        # print(test_df['Y'].unique())
        test_preds, drop_ids = modelapi.eval_testset(test_df, fp_names, mode=f'{mode_prefix}Test')
    else:
        test_df.rename(columns={'std_smiles': 'Drug'}, inplace=True)
        test_preds, drop_ids = modelapi.predict(test_df, fp_names, mode=f'{mode_prefix}Test')

    if len(drop_ids) >0:
        test_df.drop(drop_ids, axis=0, inplace=True)
    pred_names = [f'{task_name}_mean', f'{task_name}_std']+ [f'{task_name}_M{i+1}' for i in range(test_preds.shape[1]-2)]
    test_df[pred_names] = test_preds

    suffix = test_fn.suffix
    output_fn = str(test_fn).replace(suffix, "_predict.csv")
    if kwargs.get('eval_rank_col'):
        targets =test_df[kwargs['eval_rank_col']].values.reshape(-1,1)
        names = [f'{task_name}_M{i+1}' for i in range(test_preds.shape[1]-2)]+ [f'{task_name}_mean']
        results = [get_metric('spearmanr', targets, test_df[name_].values.reshape(-1,1)).item() for name_ in names ]
        logger.info([(x,y) for x,y in zip(names, results)])

    if kwargs.get('eval_class_col'):
        targets =test_df[kwargs['eval_class_col']].values.reshape(-1,1)
        names = [f'{task_name}_M{i+1}' for i in range(test_preds.shape[1]-2)]+ [f'{task_name}_mean']
        results = [get_metric('auc', targets, test_df[name_].values.reshape(-1,1)).item() for name_ in names ]
        logger.info(f'auc: {results}')
        results = [get_metric('mcc', targets, test_df[name_].values.reshape(-1,1)).item() for name_ in names ]
        logger.info(f'mcc: {results}')

        
    test_df.to_csv(output_fn, index=False)
    logger.info(f'Saved into {output_fn}')


@click.group()
def cli4():
    pass

@cli4.command('stepboost')
@click.option("--task_name", '-tn', type = str, default='cyp2c9_veith')
@click.option("--version_tag", '-tag', type = str, default='default')
@click.option("--split", '-s', type = str, default='scaffold')
@click.option("--tmp_dir", '-d', type = str,  default=None)
@click.option("--test_file", '-tf', type = str, default='tmp/test.csv')
def stepboost(**kwargs):
    task_name = kwargs['task_name']
    split_method = kwargs['split']
    meta_info = list(getattr(TaskMetaInfo, task_name).value) + [split_method]  # task_class, metric_name, norm_y, split_method
    group = admet_group(path = DATA_ROOT)
    benchmark = group.get(task_name)
    train_val, test = benchmark['train_val'], benchmark['test']
    new_df= pd.read_csv(kwargs['test_file'])
    # build models
    modelapi = TaskModelling(task_name, *meta_info, tmp_dir = kwargs['tmp_dir'])
    
    from maxqsaring.feat_enum import FeatNameToKey
    tmp_feats=FeatNameToKey.copy()

    cur_feat = getattr(TaskFeatSet, task_name).value
    # set current feature version
    if isinstance(cur_feat, dict):
        cur_feat = cur_feat[kwargs['version_tag']]

    if len(cur_feat) >0:
        for ele in cur_feat.split(','):
            del tmp_feats[ele]
        df_perf = pd.DataFrame([f'{cur_feat},{keystr}' for keystr in tmp_feats.keys()], columns=['featSet'])
    else:
        df_perf = pd.DataFrame(tmp_feats.keys(), columns=['featSet'])

    output_dir = Path(f'./{modelapi.processed_dir}/summary/')
    output_dir.mkdir(parents=True, exist_ok=True) 

    modelapi.split(train_val, split_method)
    print(df_perf)
    for fp_set in df_perf['featSet'].values:
        modelapi.gen_featsets(fp_set)
        modelapi.eval_featsets()
        modelapi.eval_testset(new_df, fp_set, mode='validTest')

cli = click.CommandCollection(sources=[cli1, cli2, cli3, cli4])

if __name__ == '__main__':
    cli()