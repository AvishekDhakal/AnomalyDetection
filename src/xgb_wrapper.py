# # xgb_wrapper.py

# from xgboost import XGBClassifier
# from sklearn.base import BaseEstimator, ClassifierMixin

# class XGBClassifierWrapper(XGBClassifier, BaseEstimator, ClassifierMixin):
#     def __init__(
#         self,
#         objective='binary:logistic',
#         learning_rate=0.1,
#         n_estimators=100,
#         max_depth=3,
#         min_child_weight=1,
#         gamma=0,
#         subsample=1,
#         colsample_bytree=1,
#         reg_alpha=0,
#         reg_lambda=1,
#         scale_pos_weight=1,
#         use_label_encoder=False,
#         eval_metric='logloss',
#         random_state=42,
#         verbosity=0
#     ):
#         super().__init__(
#             objective=objective,
#             learning_rate=learning_rate,
#             n_estimators=n_estimators,
#             max_depth=max_depth,
#             min_child_weight=min_child_weight,
#             gamma=gamma,
#             subsample=subsample,
#             colsample_bytree=colsample_bytree,
#             reg_alpha=reg_alpha,
#             reg_lambda=reg_lambda,
#             scale_pos_weight=scale_pos_weight,
#             use_label_encoder=use_label_encoder,
#             eval_metric=eval_metric,
#             random_state=random_state,
#             verbosity=verbosity
#         )
        
#         # Explicitly set as instance attributes
#         self.objective = objective
#         self.learning_rate = learning_rate
#         self.n_estimators = n_estimators
#         self.max_depth = max_depth
#         self.min_child_weight = min_child_weight
#         self.gamma = gamma
#         self.subsample = subsample
#         self.colsample_bytree = colsample_bytree
#         self.reg_alpha = reg_alpha
#         self.reg_lambda = reg_lambda
#         self.scale_pos_weight = scale_pos_weight
#         self.use_label_encoder = use_label_encoder
#         self.eval_metric = eval_metric
#         self.random_state = random_state
#         self.verbosity = verbosity
    
#     def __sklearn_tags__(self):
#         return {
#             'requires_fit': True,
#             'requires_predict': True,
#             'requires_predict_proba': True,
#             'requires_y': True,
#             'non_deterministic': False,
#             'multilabel': False,
#             'multiclass': False,
#             'single_output': True,
#             'no_validation': False
#         }


# xgb_wrapper.py

from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class XGBClassifierWrapper(XGBClassifier, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        objective='binary:logistic',
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=1,
        colsample_bytree=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    ):
        super().__init__(
            objective=objective,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=use_label_encoder,
            eval_metric=eval_metric,
            random_state=random_state,
            verbosity=verbosity
        )
    
    def __sklearn_tags__(self):
        return {
            'requires_fit': True,
            'requires_predict': True,
            'requires_predict_proba': True,
            'requires_y': True,
            'non_deterministic': False,
            'multilabel': False,
            'multiclass': False,
            'single_output': True,
            'no_validation': False
        }
