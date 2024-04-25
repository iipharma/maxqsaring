import xgboost
from typing import Dict, List
from xgboost import XGBModel
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
import pandas as pd
import numpy as np
from pathlib import Path
import joblib, json

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from maxqsaring.logging_helper import create_logger
from maxqsaring.utils.metrics import get_metric
from maxqsaring.feat_enum import FeatNameToKey

logger = create_logger('ModelBuilder')

class BaseTransform:
    def __init__(self) -> None:
        self.mean: float = None
        self.std: float =None

    def transform(self, x):
        pass

    def inverse_transform(self, inv_x):
        pass


class log10_zscore(BaseTransform):    
    def transform(self, x):
        log_x= np.log10(x)
        self.mean =  np.mean(log_x)
        self.std = np.std(log_x)
        return (log_x-self.mean) / self.std
    
    def inverse_transform(self, inv_x):
        return 10**((inv_x*self.std)+ self.mean)
    
class logExp_zscore(BaseTransform):    
    def transform(self, x):
        log_x= np.log(x)
        self.mean =  np.mean(log_x)
        self.std = np.std(log_x)
        return (log_x-self.mean) / self.std
    
    def inverse_transform(self, inv_x):
        return np.exp((inv_x*self.std)+ self.mean)
    

class log10(BaseTransform):
    def transform(self, x):
        return np.log10(x)
    
    def inverse_transform(self, inv_x):
        return 10**inv_x

class logExp(BaseTransform):
    def transform(self, x):
        return np.log(x)
    
    def inverse_transform(self, inv_x):
        return np.exp(inv_x)


def transform_y(name: str=None):
    if name is None:
        return None
    
    if name.lower().startswith('minmax'):
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler()
    elif name.lower() in ['zscore', 'standardscaler']:
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    else: 
        from sklearn.preprocessing import FunctionTransformer
        call_func: BaseTransform = eval(name)()
        return FunctionTransformer(call_func.transform, inverse_func=call_func.inverse_transform, check_inverse=True)  
    
def update_pos_weight(task_name, params: dict):
    if task_name == 'cyp2c9_substrate_carbonmangels':
        params.update({'scale_pos_weight': 4})   # neg /pos = 4
    elif task_name == 'cyp2d6_substrate_carbonmangels':
        params.update({'scale_pos_weight': 2.5})
    elif task_name == 'bbb_martins':
        params.update({'scale_pos_weight': 0.33})
    # elif task_name == 'hepg2_pbctg':
    #     params.update({'scale_pos_weight': 2})
    else:
        pass

MODEL_SEARCH_PARAMS: Dict[str, List] = {
    "n_estimators": [50, 100, 200, 300, 400, 500, 600],
    # "scale_pos_weight": [0.2, 0.4,0.6, 0.8,1],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.1, 1, 5, 10],
    "reg_lambda": [0, 0.1, 1, 5, 10],
    "min_child_weight": [1, 3, 5]
}

MODEL_TYPE_FUNCS = {
    'classification': xgboost.XGBClassifier,
    'regression': xgboost.XGBRegressor,
}

EVAL_METRICS={
    # 'classification': ['acc', 'f1', 'recall', 'specificity', 'ccr', 'mcc', 'auc', 'prc-auc'],
    'classification': ['mcc', 'npv', 'acc', 'ppv', 'spe', 'sen', 'ccr', 'f1', 'auc', 'prc-auc'],
    'regression': ['rmse', 'mae', 'r2', 'pearsonr', 'spearmanr'],
}

MERTIC_ASCEND_IS_BETTER={
    'acc': False, #
    'mcc': False,
    'auc': False,
    'f1': False,
    'prc-auc': False,
    'rmse': True,
    'mae': True,
    'r2': False, 
    'pearsonr': False,
    'spearmanr': False,
 }

MODEL_PRED_FUNC={
    'classification': lambda model, x: model.predict_proba(x)[:, 1],
    'regression': lambda model, x: model.predict(x),
}




def reduce_features(train_data: np.ndarray, valid_data: np.ndarray, model_type:str):
    if model_type.startswith('class'):
        var, corr = 0.01, 0.95
    else:
        var, corr = 0.00, 1.00

    data= np.concatenate([train_data, valid_data], axis=0)
    numcol = data.shape[1]
    data = data[:,:-1]    
    kept_col_ids = np.arange(data.shape[1])

    if var >= 0.00:
        logger.info(f'Removing low variance features...')
        variance = np.var(data, axis = 0)
        # print(variance)
        kept_col_ids = np.where(variance > var)[0]
        # print(kept_col_ids)
        logger.info(f'Remove variance (<={var}): {data.shape[1]} -> {kept_col_ids.shape[0]}')
    
    # remove the same columns
    _, index = np.unique(data[:, kept_col_ids], axis=1, return_index=True)
    kept_col_ids = kept_col_ids[index]
    logger.info(f'Remove duplicate cols: {data.shape[1]} -> {kept_col_ids.shape[0]}')

    if corr < 1.0 and kept_col_ids.shape[0]>1:
        logger.info(f'Removing high correlated features...')
        corr_mat = np.abs(np.corrcoef(data[:, kept_col_ids].T))
        rm_col_ids = np.unique(np.argwhere(np.triu(corr_mat, k=1) > corr)[:,1])
        kept_col_ids= np.delete(kept_col_ids, rm_col_ids)    
        logger.info(f'Remove corr (>{corr}): {data.shape[1]} -> {kept_col_ids.shape[0]}')
    
    kept_col_ids = np.append(kept_col_ids, numcol-1)
    return kept_col_ids


def gen_params_list(num):
    params_space= list(ParameterSampler(MODEL_SEARCH_PARAMS, n_iter=num, random_state=0))    
    return params_space

def build_one_model(model_type: str, train_data, valid_data, params, return_model=False, norm_y: str = None):
    params['tree_method']='gpu_hist'
    # params['device']=f'CUDA:{os.environ["CUDA_VISIBLE_DEVICES"]}'
    params['early_stopping_rounds'] = 10
    # params['num_boost_round'] = 1000

    # params['importance_type'] = 'cover'
    # params['objective'] = 'reg:linear'
    model: XGBModel = MODEL_TYPE_FUNCS[model_type](**params)
    X_train, y_train = train_data[:, :-1], train_data[:, -1:]
    X_test, y_test = valid_data[:,:-1], valid_data[:,-1:]

    if norm_y:
        scaler = transform_y(norm_y)
        scaler.fit(np.concatenate([y_train, y_test], axis = 0))
        model.fit(X_train, scaler.transform(y_train), eval_set=[(X_test, scaler.transform(y_test))], verbose=0)
        valid_preds = MODEL_PRED_FUNC[model_type](model, X_test).reshape(-1,1)
        valid_preds = scaler.inverse_transform(valid_preds)
        model.target_scaler = scaler  # save this attr
    else:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
        valid_preds = MODEL_PRED_FUNC[model_type](model, X_test).reshape(-1,1)

    metrics= [get_metric(item, y_test, valid_preds).squeeze() for item in  EVAL_METRICS[model_type]]
    if not return_model:
        del model
        return metrics
    return metrics, model

def build_seed_params(model_type, train_data, valid_data, params_list, norm_y, max_workers=10):
    # XGBModel = MODEL_TYPE_FUNCS[model_type]
    perfs=[]
    # note. it must use thread pool, instead process pool,
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(build_one_model, model_type, train_data, valid_data, param, False, norm_y) for param in params_list]
        for idx, future in enumerate(futures):
            perfs.append(future.result())
    perfs= pd.DataFrame(perfs, columns= EVAL_METRICS[model_type])
    return perfs


def rank_params_perfs(perfs: pd.DataFrame, metric_name:str):
    key_names= [x for x in perfs.columns.values if x.startswith(metric_name)]
    perfs[f'mean_{metric_name}'] = perfs[key_names].mean(axis=1)
    perfs[f'std_{metric_name}'] = perfs[key_names].std(axis=1)
    perfs.sort_values(by=[f'mean_{metric_name}', f'std_{metric_name}'], ascending=[MERTIC_ASCEND_IS_BETTER[metric_name], True], inplace=True)
    return perfs


def process_data(df:pd.DataFrame, valid_ids=None):
    if valid_ids:
        data = df.iloc[valid_ids]
    else:
        data = df.copy()
    data.dropna(inplace=True)
    drop_ids = df.index.difference(data.index).tolist()
    logger.info({x: data[x].iloc[0].shape for x in data.columns.values})
    data = np.concatenate([np.vstack(data[x].values) for x in data.columns.values],axis=1)
    return data, drop_ids


def select_best_params_perfs(train_df: pd.DataFrame, model_type, split_ids: dict, metric_name, processed_dir: Path, suffix: str, norm_y: str=None):
    num_params= 100
    params_list = gen_params_list(num_params)
    feat_ids_fn = processed_dir / f'keptfeatids_{suffix}.npy'
    params_perfs_fn = processed_dir / f'paramsperfs{num_params}_{suffix}.csv'
    bestparam_fn = processed_dir / f'bestparams_{suffix}.json'
    task_name = processed_dir.parent.stem
    
    if (not params_perfs_fn.exists()) or (not bestparam_fn.exists()):
        kept_col_ids = None
        if feat_ids_fn.exists():
            kept_col_ids = np.load(feat_ids_fn)

        all_perfs=[]
        for seed, (train_ids, valid_ids, _) in split_ids.items():
            logger.info(f'{"="*5} Seed {seed}')
            # train_data, valid_data = process_data(train_df, train_ids, valid_ids)
            train_data, _ = process_data(train_df, train_ids)
            valid_data, _ = process_data(train_df, valid_ids)
            # calc reduced features
            if kept_col_ids is None:
                kept_col_ids = reduce_features(train_data, valid_data, model_type)
                np.save(feat_ids_fn, kept_col_ids)
                logger.info(f'Saved into {feat_ids_fn}')

            if  kept_col_ids.shape[0] <= 2:
                logger.warning(f'the feature dim is less than 1; skip to reduce features')
            else:
                train_data=train_data[:, kept_col_ids]
                valid_data=valid_data[:, kept_col_ids]

            [update_pos_weight(task_name, x)  for x in params_list]# add task weight, update scale_pos_weight
            perfs = build_seed_params(model_type, train_data, valid_data, params_list, norm_y)
            perfs.rename(columns={key: f'{key}_seed{seed}' for key in perfs.columns.values}, inplace=True)
            logger.info(f'\n{perfs}\n{"="*30}')
            all_perfs.append(perfs)

        all_perfs = pd.concat(all_perfs, axis=1)
    else:
        all_perfs=pd.read_csv(params_perfs_fn)
           
    all_perfs = rank_params_perfs(all_perfs, metric_name)
    all_perfs.to_csv(params_perfs_fn)
    logger.info(f'Save params performance into {params_perfs_fn}')
    tmp = all_perfs[[f'mean_{metric_name}', f'std_{metric_name}']].head(5)
    logger.info(f'Top 5 performance: \n{tmp}')

    return params_list[all_perfs.index[0]]

def get_stats(df:pd.DataFrame): # add three row: mean, std, mean +/- std
    df = pd.concat([df, df.apply(['mean', 'std'])])
    df.loc[len(df.index)] = [f'{x:.3f} ± {y:.3f}' for x,y in zip(df.loc["mean"].values, df.loc["std"].values)]
    return df

def build_model_and_eval_featsets(train_df: pd.DataFrame, model_type:str, split_ids: dict, processed_dir: Path, full_suffix, norm_y, **model_params):

    model_dir: Path = processed_dir / f'{full_suffix}'
    model_dir.mkdir(parents=True, exist_ok=True)
    json.dump(model_params, open(model_dir / 'bestparams.json', 'w'))

    perf_fn = model_dir / 'performance.csv'
    if not perf_fn.exists():
        feat_ids_fn = model_dir / f'keptfeatids.npy'
        kept_col_ids=None
        if feat_ids_fn.exists():
            kept_col_ids = np.load(feat_ids_fn)

        all_perfs = []
        for seed, (train_ids, valid_ids, _) in split_ids.items():
            logger.info(f'{"="*5} build seed {seed}...')
            # train_data, valid_data = process_data(train_df, train_ids, valid_ids)
            train_data, _ = process_data(train_df, train_ids)
            valid_data, _ = process_data(train_df, valid_ids)
            # calc reduced features
            if kept_col_ids is None:
                kept_col_ids = reduce_features(train_data, valid_data, model_type)
                np.save(feat_ids_fn, kept_col_ids)
                logger.info(f'Saved into {feat_ids_fn}')

            if  kept_col_ids.shape[0] <=2:
                logger.warning(f'the feature dim is less than ; skip to reduce features')
            else:
                train_data=train_data[:, kept_col_ids]
                valid_data=valid_data[:, kept_col_ids]

            val_perfs, model = build_one_model(model_type, train_data, valid_data, model_params, return_model=True, norm_y=norm_y)
            tmp_pref = pd.DataFrame([val_perfs], columns=EVAL_METRICS[model_type])
            logger.info(f'val-seed{seed} performance:\n{tmp_pref}')
            joblib.dump(model, model_dir / f'seed{seed}.joblib')
            all_perfs.append(val_perfs)
        
        all_perfs = pd.DataFrame(all_perfs, columns=EVAL_METRICS[model_type])
        all_perfs = get_stats(all_perfs)
        all_perfs.to_csv(perf_fn)
    else:
        all_perfs = pd.read_csv(perf_fn, index_col=0)
    logger.info(f'{"="*5}: \n{all_perfs}')
    # metric_data = all_perfs[metric_name].values 
    return list(map(float, all_perfs.loc["mean"].values)), list(map(float, np.array(all_perfs.loc["std"].values)))

def eval_testdata(test_df: pd.DataFrame, model_type:str, model_dir: Path):
    model_fns = list(model_dir.glob('seed*.joblib'))
    test_data, drop_ids = process_data(test_df)
    kept_col_ids = np.load(model_dir / f'keptfeatids.npy')
    test_data = test_data[:, kept_col_ids]
    perfs = []
    test_preds=[]
    for model_fn in sorted(model_fns):
        model: XGBModel = joblib.load(model_fn)
        test_pred = MODEL_PRED_FUNC[model_type](model, test_data[:,:-1]).reshape(-1,1)
        if hasattr(model, 'target_scaler'):
            test_pred = model.target_scaler.inverse_transform(test_pred)
        test_metrics = [get_metric(item, test_data[:, -1:], test_pred).squeeze() for item in  EVAL_METRICS[model_type]]
        perfs.append(test_metrics)
        test_preds.append(test_pred)

    test_preds= np.concatenate(test_preds,axis=1)
    test_pred_mean = np.mean(test_preds,axis=1).reshape(-1,1)
    test_pred_std = np.std(test_preds,axis=1).reshape(-1,1)

    mean_metrics= [get_metric(item, test_data[:, -1:], test_pred_mean).squeeze() for item in  EVAL_METRICS[model_type]]
    perfs = pd.DataFrame(perfs, columns=EVAL_METRICS[model_type])
    perfs = get_stats(perfs)
    perfs.loc[len(perfs.index)]=mean_metrics  # add the mean testing
    logger.info(f'test performance:\n{perfs}')
    return perfs, np.concatenate([test_pred_mean, test_pred_std, test_preds], axis =1), drop_ids


def predict(test_df: pd.DataFrame, model_type:str, model_dir: Path):
    model_fns = list(model_dir.glob('seed*.joblib'))
    test_data, drop_ids = process_data(test_df)
    kept_col_ids = np.load(model_dir / f'keptfeatids.npy')
    test_data = test_data[:, kept_col_ids[:-1]]
    test_preds=[]
    for model_fn in sorted(model_fns):
        model: XGBModel = joblib.load(model_fn)
        # convert model to json
        model.save_model(model_fn.with_suffix('.json'))

        test_pred = MODEL_PRED_FUNC[model_type](model, test_data).reshape(-1,1)
        if hasattr(model, 'target_scaler'):
            test_pred = model.target_scaler.inverse_transform(test_pred)
        test_preds.append(test_pred)

    test_preds= np.concatenate(test_preds,axis=1)
    test_pred_mean = np.mean(test_preds,axis=1).reshape(-1,1)
    test_pred_std = np.std(test_preds,axis=1).reshape(-1,1)
    return np.concatenate([test_pred_mean, test_pred_std, test_preds], axis =1), drop_ids


def explain(test_df: pd.DataFrame, model_type: str, fp_names: list):
    pass