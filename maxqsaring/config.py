import xgboost
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, auc, precision_recall_curve
from scipy.stats import spearmanr, pearsonr
import numpy as np
MODEL_PARAMS={
    'hia_hou': {"subsample": 1.0, "reg_lambda": 1, "reg_alpha": 0, "n_estimators": 300, "min_child_weight": 1, "max_depth": 4, "learning_rate": 0.2, "colsample_bytree": 0.6},
    'pgp_broccatelli': {"subsample": 0.6, "reg_lambda": 5, "reg_alpha": 0.1, "n_estimators": 200, "min_child_weight": 1, "max_depth": 6, "learning_rate": 0.2, "colsample_bytree": 0.9},
    'bioavailability_ma': {"subsample": 0.9, "reg_lambda": 10, "reg_alpha": 1, "n_estimators": 500, "min_child_weight": 1, "max_depth": 5, "learning_rate": 0.1, "colsample_bytree": 0.7},
    'bbb_martins': {"subsample": 1.0, "reg_lambda": 10, "reg_alpha": 0, "n_estimators": 500, "min_child_weight": 1, "max_depth": 7, "learning_rate": 0.01, "colsample_bytree": 0.6},
    'cyp3a4_substrate_carbonmangels': {"subsample": 0.6, "reg_lambda": 5, "reg_alpha": 1, "n_estimators": 300, "min_child_weight": 1, "max_depth": 5, "learning_rate": 0.01, "colsample_bytree": 0.7},
    'herg': {"subsample": 0.7, "reg_lambda": 0.1, "reg_alpha": 1, "n_estimators": 200, "min_child_weight": 1, "max_depth": 6, "learning_rate": 0.05, "colsample_bytree": 0.5},
    'ames': {"subsample": 0.9, "reg_lambda": 0, "reg_alpha": 0, "n_estimators": 300, "min_child_weight": 1, "max_depth": 5, "learning_rate": 0.1, "colsample_bytree": 1.0},
    'dili': {"subsample": 0.5, "reg_lambda": 0, "reg_alpha": 1, "n_estimators": 400, "min_child_weight": 3, "max_depth": 4, "learning_rate": 0.05, "colsample_bytree": 0.7},

    'cyp2d6_veith': {"subsample": 0.5, "reg_lambda": 5, "reg_alpha": 0.1, "n_estimators": 600, "min_child_weight": 1, "max_depth": 5, "learning_rate": 0.05, "colsample_bytree": 0.8},
    'cyp3a4_veith': {"subsample": 0.8, "reg_lambda": 10, "reg_alpha": 5, "n_estimators": 500, "min_child_weight": 3, "max_depth": 6, "learning_rate": 0.1, "colsample_bytree": 0.8},
    'cyp2c9_veith': {"subsample": 0.8, "reg_lambda": 10, "reg_alpha": 5, "n_estimators": 400, "min_child_weight": 5, "max_depth": 7, "learning_rate": 0.05, "colsample_bytree": 0.6},
    'cyp2d6_substrate_carbonmangels':{"subsample": 0.5, "reg_lambda": 0.1, "reg_alpha": 0, "n_estimators": 400, "min_child_weight": 3, "max_depth": 5, "learning_rate": 0.01, "colsample_bytree": 0.8},
    'cyp2c9_substrate_carbonmangels': {"subsample": 0.9, "reg_lambda": 10, "reg_alpha": 1, "n_estimators": 100, "min_child_weight": 3, "max_depth": 3, "learning_rate": 0.05, "colsample_bytree": 0.6},

    
    'caco2_wang':  {"subsample": 0.5, "reg_lambda": 0, "reg_alpha": 1, "n_estimators": 400, "min_child_weight": 3, "max_depth": 4, "learning_rate": 0.05, "colsample_bytree": 0.7},
    'lipophilicity_astrazeneca': {"subsample": 0.8, "reg_lambda": 10, "reg_alpha": 5, "n_estimators": 400, "min_child_weight": 3, "max_depth": 5, "learning_rate": 0.1, "colsample_bytree": 0.7},
    'solubility_aqsoldb': {"subsample": 0.5, "reg_lambda": 5, "reg_alpha": 0.1, "n_estimators": 600, "min_child_weight": 1, "max_depth": 5, "learning_rate": 0.05, "colsample_bytree": 0.8},
    'ppbr_az':{"subsample": 0.5, "reg_lambda": 0, "reg_alpha": 1, "n_estimators": 400, "min_child_weight": 3, "max_depth": 4, "learning_rate": 0.05, "colsample_bytree": 0.7}, 
    'ld50_zhu': {"subsample": 0.5, "reg_lambda": 5, "reg_alpha": 0.1, "n_estimators": 600, "min_child_weight": 1, "max_depth": 5, "learning_rate": 0.05, "colsample_bytree": 0.8},
    # 'ld50_zhu':{"subsample": 0.8, "reg_lambda": 10, "reg_alpha": 5, "n_estimators": 400, "min_child_weight": 5, "max_depth": 7, "learning_rate": 0.05, "colsample_bytree": 0.6},


    'vdss_lombardo': {"subsample": 0.6, "reg_lambda": 5, "reg_alpha": 1, "n_estimators": 300, "min_child_weight": 1, "max_depth": 5, "learning_rate": 0.01, "colsample_bytree": 0.7},
    # 'half_life_obach':{"subsample": 0.5, "reg_lambda": 10, "reg_alpha": 10, "n_estimators": 200, "min_child_weight": 1, "max_depth": 4, "learning_rate": 0.01, "colsample_bytree": 0.6},
    'half_life_obach':{"subsample": 0.7, "reg_lambda": 0.1, "reg_alpha": 1, "n_estimators": 200, "min_child_weight": 1, "max_depth": 6, "learning_rate": 0.05, "colsample_bytree": 0.5},
    
    'clearance_microsome_az': {"subsample": 0.7, "reg_lambda": 5, "reg_alpha": 10, "n_estimators": 600, "min_child_weight": 3, "max_depth": 6, "learning_rate": 0.01, "colsample_bytree": 0.9},
    'clearance_hepatocyte_az':{"subsample": 0.6, "reg_lambda": 5, "reg_alpha": 1, "n_estimators": 300, "min_child_weight": 1, "max_depth": 5, "learning_rate": 0.01, "colsample_bytree": 0.7}
}

class_models = []

MODEL_FUNCS={
    'hia_hou': xgboost.XGBClassifier,
    'pgp_broccatelli': xgboost.XGBClassifier,
    'bioavailability_ma': xgboost.XGBClassifier,
    'bbb_martins': xgboost.XGBClassifier,
    'cyp3a4_substrate_carbonmangels': xgboost.XGBClassifier,
    'herg': xgboost.XGBClassifier,
    'ames': xgboost.XGBClassifier,
    'dili':xgboost.XGBClassifier,

    'cyp2d6_veith':xgboost.XGBClassifier,
    'cyp3a4_veith':xgboost.XGBClassifier,
    'cyp2c9_veith':xgboost.XGBClassifier,
    'cyp2d6_substrate_carbonmangels':xgboost.XGBClassifier,
    'cyp2c9_substrate_carbonmangels':xgboost.XGBClassifier,

    'caco2_wang':  xgboost.XGBRegressor,
    'lipophilicity_astrazeneca': xgboost.XGBRegressor,
    'solubility_aqsoldb':xgboost.XGBRegressor,
    'ppbr_az':xgboost.XGBRegressor, 
    'ld50_zhu':xgboost.XGBRegressor,

    'vdss_lombardo':xgboost.XGBRegressor,
    'half_life_obach':xgboost.XGBRegressor,
    'clearance_microsome_az':xgboost.XGBRegressor,
    'clearance_hepatocyte_az':xgboost.XGBRegressor
}

def prc_auc_score(targets, preds) -> float:
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)

# METRIC_FUNCS={
#     "caco2_wang": [mean_absolute_error, mean_squared_error],
#     'half_life_obach': [lambda x, y: spearmanr(x, y)[0], lambda x,y: np.sqrt(mean_squared_error(x,y))],
#     'hia_hou': [auc, prc_auc],
#     'pgp_broccatelli': [auc, prc_auc],
# }

METRIC_FUNCS={
    'hia_hou': [roc_auc_score, prc_auc_score],
    'pgp_broccatelli': [roc_auc_score, prc_auc_score],
    'bioavailability_ma': [roc_auc_score, prc_auc_score],
    'bbb_martins': [roc_auc_score, prc_auc_score],
    'cyp3a4_substrate_carbonmangels': [roc_auc_score, prc_auc_score],
    'herg': [roc_auc_score, prc_auc_score],
    'ames': [roc_auc_score, prc_auc_score],
    'dili':[roc_auc_score, prc_auc_score],

    'cyp2d6_veith':[prc_auc_score, roc_auc_score],
    'cyp3a4_veith':[prc_auc_score, roc_auc_score],
    'cyp2c9_veith':[prc_auc_score, roc_auc_score],
    'cyp2d6_substrate_carbonmangels':[prc_auc_score, roc_auc_score],
    'cyp2c9_substrate_carbonmangels':[prc_auc_score, roc_auc_score],

    'caco2_wang':  [mean_absolute_error, lambda x,y: np.sqrt(mean_squared_error(x,y))],
    'lipophilicity_astrazeneca': [mean_absolute_error, lambda x,y: np.sqrt(mean_squared_error(x,y))],
    'solubility_aqsoldb':[mean_absolute_error, lambda x,y: np.sqrt(mean_squared_error(x,y))],
    'ppbr_az':[mean_absolute_error, lambda x,y: np.sqrt(mean_squared_error(x,y))], 
    'ld50_zhu':[mean_absolute_error, lambda x,y: np.sqrt(mean_squared_error(x,y))],

    'vdss_lombardo':[lambda x, y: spearmanr(x, y)[0], lambda x,y: np.sqrt(mean_squared_error(x,y))],
    'half_life_obach':[lambda x, y: spearmanr(x, y)[0], lambda x,y: np.sqrt(mean_squared_error(x,y))],
    'clearance_microsome_az':[lambda x, y: spearmanr(x, y)[0], lambda x,y: np.sqrt(mean_squared_error(x,y))],
    'clearance_hepatocyte_az':[lambda x, y: spearmanr(x, y)[0], lambda x,y: np.sqrt(mean_squared_error(x,y))]

}
