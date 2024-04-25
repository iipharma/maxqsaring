
from pathlib import Path
import numpy as np
from maxqsaring.logging_helper import create_logger
from maxqsaring.utils import feature_generator, split
from maxqsaring.feat_enum import FeatNameToKey
import json
from maxqsaring.utils.model_builder import update_pos_weight
logger = create_logger(__name__)

# class NormalizeY:
#     def __init__(self, ):

#     def norm(self, x):
#         return (x-3)/2


class TaskModelling:
    def __init__(self, task_name: str, *args, **kwargs) -> None:
        self.task_name = task_name
        tmp_dir = kwargs.get('tmp_dir')
        self.processed_dir = Path(tmp_dir) / task_name if tmp_dir else Path(f'./tempdata/{task_name}')
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.split_ids :dict = None
        self.smiles_col=kwargs.get('smiles_col', 'Drug')
        self.target_col=kwargs.get('target_col', 'Y')
        self.model_type = args[0]
        self.key_metric = args[1]
        self.norm_y = args[2]
        self.fp_names = None
        logger.info(f'meta info: {task_name}, {args}')

    def split(self, train_df, method='scaffold'):
        self.train_df = train_df
        logger.info(f'{method} split ...')
        if method == 'scaffold':
            seeds = [1,2,3,4,5]
            frac = [0.875, 0.125, 0.0]
            # frac = [0.9, 0.1, 0.0]

            self.split_ids = split.scaffold_split_by_seed(self.train_df, seeds, frac, entity= 'Drug')

        elif method == 'scaffold-bin':
            self.split_ids = split.scaffold_bin_split(self.train_df, 10, 5)
        
        elif method == 'random-cv':
            self.split_ids = split.random_cv_split(self.train_df, 5)
        elif method == 'random-stratified-cv':
            self.split_ids = split.stratified_cv_split(self.train_df, 5)
        elif method == 'cluster-cv':
            self.split_ids = split.cluster_cv_split(self.train_df, 5)    
        elif method == 'cluster-sample':
            self.split_ids = split.cluster_sample_split(self.train_df, 5)
        else:
            raise ValueError(f'{method} split is not allowed!')
        logger.info(f'split_ids: {[(key, len(value[0]), len(value[1]) ) for key,value in self.split_ids.items()]}')


    def gen_featsets(self, fp_names: str or list, mode='train'):
        logger.info(f'{"="*20} Start to run task: {self.task_name} task...')
        logger.info(f'{"="*10} Generating features...')
        logger.info(f'Current Features: {fp_names}')
        self.fp_names = fp_names if isinstance(fp_names, list) else fp_names.split(',')
        self.trainfeat_df = feature_generator.gen_features(self.train_df, self.processed_dir, self.fp_names, self.smiles_col, self.target_col, mode=mode)
        logger.info(f'Finished Features: {fp_names}')

    def search_best_params(self):
        logger.info(f'{"="*10} Search model params...')
        suffix = FeatNameToKey[self.fp_names[0]] # record the first feature name
        params_fn= self.processed_dir / f'hyperparameter/bestparams_{suffix}.json'
        params_fn.parent.mkdir(parents=True, exist_ok=True)
        if params_fn.exists():
            self.best_params = json.load(open(params_fn, 'r'))
        else:
            from maxqsaring.utils.model_builder import select_best_params_perfs
            self.best_params = select_best_params_perfs(
                            self.trainfeat_df, 
                            self.model_type,
                            self.split_ids,
                            self.key_metric,
                            self.processed_dir / 'hyperparameter',
                            suffix,
                            self.norm_y
                        )
            logger.info(f'Save params into {params_fn}')

        
        update_pos_weight(self.task_name, self.best_params)
        json.dump(self.best_params, open(params_fn, 'w'))
        logger.info(f'Best params:\n{self.best_params}')

    def eval_featsets(self, new_fp_names:list or str =None, restart=False):
        logger.info(f'{"="*10} Evaluating features...')
        fp_names =new_fp_names if new_fp_names else self.fp_names
        fp_names = fp_names if isinstance(fp_names, list) else fp_names.split(',')
        full_suffix = [FeatNameToKey[x] for x in fp_names]
        suffix = full_suffix[0]
        full_suffix = "-".join(full_suffix)
        self.search_best_params()
        update_pos_weight(self.task_name, self.best_params)
        logger.info(f'Current features: {fp_names}; key names: {full_suffix}')
        logger.info(f'Current best params: {self.best_params}')
        if restart:
            import shutil
            tmp_dir= self.processed_dir / f'models/{full_suffix}'
            if tmp_dir.exists(): shutil.rmtree(tmp_dir)

        from maxqsaring.utils.model_builder import build_model_and_eval_featsets
        perfs = build_model_and_eval_featsets(
                    self.trainfeat_df, 
                    self.model_type, 
                    self.split_ids, 
                    self.processed_dir / 'models',
                    full_suffix, 
                    self.norm_y,
                    **self.best_params)
        return perfs

    def eval_testset(self, test_df, sel_fp_names, mode='test') -> np.ndarray:
        logger.info(f'{"="*20} Start to run test task: {self.task_name} ...')
        logger.info(f'Current Features: {sel_fp_names}')
        sel_fp_names = sel_fp_names if isinstance(sel_fp_names, list) else sel_fp_names.split(',')
        full_suffix = "-".join([FeatNameToKey[x] for x in sel_fp_names])
        model_dir = self.processed_dir / f'models/{full_suffix}'
        if not (model_dir/ 'performance.csv').exists():
            raise ValueError(f'Models not found from {model_dir}')
        logger.info(f'{"="*10} Generating features...')
        testfeat_df = feature_generator.gen_features(test_df, self.processed_dir, sel_fp_names, self.smiles_col, self.target_col, mode=mode)
        logger.info(f'model dir: {model_dir}')
        from maxqsaring.utils.model_builder import eval_testdata
        test_perfs, test_preds, drop_ids = eval_testdata(testfeat_df, self.model_type, model_dir)
        out_fn = model_dir / f'{mode}_performance.csv'
        test_perfs.to_csv(out_fn)
        logger.info(f'Save into {out_fn}')
        return test_preds, drop_ids
        
        
    def predict(self, test_df, sel_fp_names, mode='tmpTest'):
        logger.info(f'{"="*20} Start to run test task: {self.task_name} ...')
        logger.info(f'Current Features: {sel_fp_names}')
        sel_fp_names = sel_fp_names if isinstance(sel_fp_names, list) else sel_fp_names.split(',')
        full_suffix = "-".join([FeatNameToKey[x] for x in sel_fp_names])
        model_dir = self.processed_dir / f'models/{full_suffix}'
        assert (model_dir/ 'performance.csv').exists()
        logger.info(f'{"="*10} Generating features...')
        testfeat_df = feature_generator.gen_features(test_df, self.processed_dir, sel_fp_names, self.smiles_col, target_col=None, mode=mode)
        logger.info(f'model dir: {model_dir}')
        from maxqsaring.utils.model_builder import predict
        test_preds, drop_ids= predict(testfeat_df, self.model_type, model_dir)
        return test_preds, drop_ids

    def explain(self, sel_fp_names):
        logger.info(f'{"="*20} Start to explain features ...')
        logger.info(f'Current Features: {sel_fp_names}')
        sel_fp_names = sel_fp_names if isinstance(sel_fp_names, list) else sel_fp_names.split(',')
        
        
        pass