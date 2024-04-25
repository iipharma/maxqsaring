import os
from pathlib import Path
from maxqsaring.modelling import TaskModelling
from maxqsaring.feat_enum import FeatNameToKey
from maxqsaring.utils.model_builder import EVAL_METRICS, MERTIC_ASCEND_IS_BETTER
from pandas import DataFrame
import pandas as pd
from maxqsaring.logging_helper import create_logger
logger = create_logger('Util')

def compare_better(metric_name, old, new):
    flag = new > old
    if not MERTIC_ASCEND_IS_BETTER[metric_name]: # larger is better
        return flag
    else:
        return not flag

def run_one_generation(modelapi:TaskModelling, df_perf:DataFrame, output_dir, *meta_info):
    means = []
    stds = []
    task_name, model_type, metric_name, norm_y, split_method = meta_info
    for fp_set in df_perf['featSet'].values:
        # generate features with y
        modelapi.gen_featsets(fp_set)
        # build models
        modelapi.search_best_params()
        mean, std = modelapi.eval_featsets()
        means.append(mean)
        stds.append(std)
    
    mean_names = [f'mean_{x}' for x in EVAL_METRICS[model_type]]
    std_names = [f'std_{x}' for x in EVAL_METRICS[model_type]]
    df_perf[mean_names] = means
    df_perf[std_names] =stds
    df_perf.sort_values(by=[f'mean_{metric_name}', f'std_{metric_name}'], ascending=[MERTIC_ASCEND_IS_BETTER[metric_name], True], inplace=True)
    logger.info(f'Top 5 performance: \n{df_perf.head(5)}')
    df_perf.to_csv(output_dir / 'performance.csv', index=False)

    feat_name, best_mean, best_std = df_perf[[
                                    'featSet', 
                                    f'mean_{metric_name}', 
                                    f'std_{metric_name}'
                                    ]].iloc[0].values
    return feat_name, best_mean, best_std

def select_bestfeat_and_test(modelapi: TaskModelling, train_df, num_gen, *meta_info):
    task_name, model_type, metric_name, norm_y, split_method = meta_info
    modelapi.split(train_df, split_method)
    perf_list = []
    df_perf_one = pd.DataFrame(FeatNameToKey.keys(), columns=['featSet'])
    for gen_idx in range(1, num_gen+1):
        logger.info(f'{"#"*20} Gen {gen_idx} {"#"*20}')
        output_dir = Path(f'./{modelapi.processed_dir}/summary/gen_{gen_idx}')
        output_dir.mkdir(parents=True, exist_ok=True) 
        df_perf_one['genearation'] = gen_idx
        feat_name, best_mean, best_std = run_one_generation(modelapi, df_perf_one, output_dir, *meta_info)
        if gen_idx >1:
            if not compare_better(metric_name, perf_list[-1][2], best_mean): # compare the mean values
                logger.info(f'Gen {gen_idx} ({metric_name}:{best_mean:.5f}) is not better than Gen {gen_idx-1} ({metric_name}:{perf_list[-1][2]:.5f}).')
                return perf_list[-1][1]
        perf_list.append([gen_idx, feat_name, best_mean, best_std])
        fp_names = df_perf_one['featSet'].values
        fp_list = [f'{fp_names[0]},{add_fp.split(",")[-1]}' for add_fp in fp_names[1:]]
        df_perf_one = pd.DataFrame(fp_list, columns=['featSet'])
    
    return feat_name