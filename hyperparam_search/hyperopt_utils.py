#import required packages
import gc
import numpy as np
import pandas as pd
import numpy.random
import lightgbm as lgb

from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample
#optional but advised
import warnings
warnings.filterwarnings('ignore')

#GLOBAL HYPEROPT PARAMETERS
NUM_EVALS = 1000 #number of hyperopt evaluation rounds
N_FOLDS = 5 #number of cross-validation folds on data in each evaluation round

#LIGHTGBM PARAMETERS
LGBM_MAX_LEAVES = 2**10 #maximum number of leaves per tree for LightGBM
LGBM_MAX_DEPTH = 25 #maximum tree depth for LightGBM

def quick_hyperopt(data, labels, package='lgbm', num_evals=NUM_EVALS, eval_metric='mae', diagnostic=False):
    
    # LightGBM
    if package=='lgbm':
        print(f'Running {num_evals} rounds of LightGBM parameter optimisation:')
        #clear space
        gc.collect()
        
        integer_params = ['max_depth', 'num_leaves', 'min_data_in_leaf', 'bagging_freq']
        if eval_metric=='mae':
            metric = 'mae'
        elif eval_metric=='mse':
            metric = 'mse'
        else:
            print(f'Metric {eval_metric} not found. Falling back to mae.')
            metric = 'mae'
            
        def objective(space_params):
            
            #cast integer params from float to int
            for param in integer_params:
                space_params[param] = int(space_params[param])
            
            cv_results = lgb.cv(space_params,
                                train,
                                nfold=N_FOLDS,
                                stratified=False,
                                early_stopping_rounds=200,
                                metrics=metric,
                                seed=42)
            if metric=='mae':
                best_loss = cv_results['l1-mean'][-1]
            elif metric=='mse':
                best_loss = cv_results['l2-mean'][-1]

            return{'loss':best_loss, 'status': STATUS_OK }
        
        train = lgb.Dataset(data, labels)
                
        #integer and string parameters, used with hp.choice()
        objective_list = ['huber', 'gamma', 'fair']
        space ={
                'boosting' : 'gbdt',
                'num_leaves' : hp.quniform('num_leaves', 8, 92, 4),
                'max_depth': hp.quniform('max_depth', -1, 16, 1),
                'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 100, 1),
                'reg_alpha' : hp.uniform('reg_alpha', 0.1, 0.95),
                'reg_lambda' : hp.uniform('reg_lambda', 0.1, 0.95),
                'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
                'metric' : 'mae',
                'objective' : hp.choice('objective', objective_list),
                'bagging_fraction' : hp.uniform('bagging_fraction', 0.5, 0.95),
                'bagging_freq': hp.quniform('bagging_freq', 3, 7, 1)
            }
        
        #optional: activate GPU for LightGBM
        #follow compilation steps here:
        #https://www.kaggle.com/vinhnguyen/gpu-acceleration-for-lightgbm/
        #then uncomment lines below:
        #space['device'] = 'gpu'
        #space['gpu_platform_id'] = 0,
        #space['gpu_device_id'] =  0

        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals, 
                    trials=trials)
                
        #fmin() will return the index of values chosen from the lists/arrays in 'space'
        #to obtain actual values, index values are used to subset the original lists/arrays
        best['objective'] = objective_list[best['objective']]
                
        #cast floats of integer params to int
        for param in integer_params:
            best[param] = int(best[param])
        
        print('{' + '\n'.join('{}: {}'.format(k, v) for k, v in best.items()) + '}')
        if diagnostic:
            return(best, trials)
        else:
            return(best)
    
    else:
        print('Package not recognised. Please use "lgbm" for LightGBM, "xgb" for XGBoost or "cb" for CatBoost.')             