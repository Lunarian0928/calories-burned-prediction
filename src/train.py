"""
train.py
--------
모델 학습 모듈

Optuna 기반 하이퍼파라미터 최적화와 스태킹 앙상블 학습을 담당합니다.
최종 채택된 아키텍처: XGBoost + LightGBM + CatBoost + RandomForest → RidgeCV 메타 모델

핵심 설계:
- Keytel 공식(물리 기반 칼로리 추정)을 베이스라인으로 사용
- 잔차(실제값 - Keytel 예측치)를 학습 타겟으로 설정하여 모델이 물리 공식의 오차를 학습
- K-Means 군집화로 운동 패턴을 세분화하여 파생 변수(Cluster_ID)로 활용
"""

import os
import numpy as np
import pandas as pd
import joblib
import optuna

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.features import add_poly_and_ratios
from src.utils import get_preprocessor, seed_everything

seed_everything(42)

MODEL_DIR = './models'
CLUSTER_COLS = ['Exercise_Duration', 'BPM', 'Body_Temperature(F)']
N_CLUSTERS = 5
N_TRIALS = 30
KEYTEL_COL = 'Session_Keytel_Calories'


class StackingWrapper(BaseEstimator, RegressorMixin):
    """
    스태킹 앙상블 래퍼 클래스

    전처리(K-Means 군집화, ColumnTransformer) → 잔차 예측 → Keytel 합산의
    전체 파이프라인을 하나의 객체로 캡슐화합니다.
    predict() 호출만으로 최종 칼로리 예측값을 반환합니다.
    """

    def __init__(self, stacking_model, kmeans, scaler, preprocessor, keytel_col, cluster_cols):
        self.stacking_model = stacking_model
        self.kmeans = kmeans
        self.scaler = scaler
        self.preprocessor = preprocessor
        self.keytel_col = keytel_col
        self.cluster_cols = cluster_cols

    def fit(self, X, y):
        return self

    def predict(self, X):
        X = X.copy()

        # K-Means 군집 ID 파생
        X_scaled = self.scaler.transform(X[self.cluster_cols])
        X['Cluster_ID'] = self.kmeans.predict(X_scaled)

        # Keytel 베이스라인 분리
        keytel = X[self.keytel_col].values

        # 전처리 후 잔차 예측
        X_ready = self.preprocessor.transform(X)
        if hasattr(X_ready, 'toarray'):
            X_ready = X_ready.toarray()

        resid_pred = self.stacking_model.predict(X_ready)

        # 잔차 + Keytel 합산으로 최종 칼로리 복원
        return resid_pred + keytel


def _objective(trial, X, y, cv):
    """Optuna 목적 함수 — 스태킹 V3 하이퍼파라미터 탐색"""

    param_xgb = {
        'n_estimators': trial.suggest_int('xgb_n', 500, 1500),
        'max_depth': trial.suggest_int('xgb_depth', 3, 10),
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1),
        'subsample': trial.suggest_float('xgb_sub', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_col', 0.6, 1.0),
        'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
    }
    param_lgbm = {
        'n_estimators': trial.suggest_int('lgbm_n', 500, 1500),
        'max_depth': trial.suggest_int('lgbm_depth', 3, 15),
        'learning_rate': trial.suggest_float('lgbm_lr', 0.01, 0.1),
        'num_leaves': trial.suggest_int('lgbm_leaves', 20, 100),
        'random_state': 42, 'n_jobs': -1, 'verbose': -1,
    }
    param_cat = {
        'iterations': trial.suggest_int('cat_iterations', 500, 1500),
        'depth': trial.suggest_int('cat_depth', 4, 10),
        'learning_rate': trial.suggest_float('cat_lr', 0.01, 0.1),
        'random_seed': 42, 'verbose': 0, 'allow_writing_files': False,
    }
    param_rf = {
        'n_estimators': trial.suggest_int('rf_n', 100, 300),
        'max_depth': trial.suggest_int('rf_depth', 5, 20),
        'random_state': 42, 'n_jobs': -1,
    }

    estimators = [
        ('xgb',  XGBRegressor(**param_xgb)),
        ('lgbm', LGBMRegressor(**param_lgbm)),
        ('cat',  CatBoostRegressor(**param_cat)),
        ('rf',   RandomForestRegressor(**param_rf)),
    ]

    reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=3,
        passthrough=True,
        n_jobs=-1,
    )

    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # K-Means 군집 ID 생성
        scaler_tmp = StandardScaler()
        X_tr['Cluster_ID'] = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(
            scaler_tmp.fit_transform(X_tr[CLUSTER_COLS])
        )
        X_val['Cluster_ID'] = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit_predict(
            scaler_tmp.transform(X_val[CLUSTER_COLS])
        )

        # 잔차 타겟 생성
        y_tr_resid = y_tr - X_tr[KEYTEL_COL]
        keytel_val = X_val[KEYTEL_COL]

        # 전처리
        preprocessor = get_preprocessor(X_tr)
        X_tr_ready = preprocessor.fit_transform(X_tr)
        X_val_ready = preprocessor.transform(X_val)
        if hasattr(X_tr_ready, 'toarray'):
            X_tr_ready = X_tr_ready.toarray()
            X_val_ready = X_val_ready.toarray()

        reg.fit(X_tr_ready, y_tr_resid)
        pred_resid = reg.predict(X_val_ready)
        final_pred = pred_resid + keytel_val

        scores.append(np.sqrt(mean_squared_error(y_val, final_pred)))

    return np.mean(scores)


def train(X_train: pd.DataFrame, y_train: pd.Series, model_name: str, cv) -> StackingWrapper:
    """
    스태킹 앙상블 모델을 학습하고 저장합니다.

    저장된 모델이 있으면 로드하고, 없으면 Optuna로 최적화 후 전체 데이터로 재학습합니다.

    Parameters
    ----------
    X_train   : 학습 피처 데이터프레임 (add_poly_and_ratios 적용 후)
    y_train   : 학습 타겟 시리즈
    model_name: 모델 저장 파일명 (확장자 제외)
    cv        : KFold 객체

    Returns
    -------
    학습된 StackingWrapper 객체
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, f'{model_name}.pkl')

    if os.path.exists(save_path):
        print(f'기존 모델 로드: {save_path}')
        return joblib.load(save_path)

    print('Optuna 하이퍼파라미터 최적화 시작...')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, cv),
        n_trials=N_TRIALS,
    )
    print(f'최적 RMSE: {study.best_value:.4f}')

    bp = study.best_params
    p_xgb  = {'n_estimators': bp['xgb_n'],  'max_depth': bp['xgb_depth'],  'learning_rate': bp['xgb_lr'],
               'subsample': bp['xgb_sub'],    'colsample_bytree': bp['xgb_col'],
               'random_state': 42, 'n_jobs': -1, 'verbosity': 0}
    p_lgbm = {'n_estimators': bp['lgbm_n'], 'max_depth': bp['lgbm_depth'], 'learning_rate': bp['lgbm_lr'],
               'num_leaves': bp['lgbm_leaves'], 'random_state': 42, 'n_jobs': -1, 'verbose': -1}
    p_cat  = {'iterations': bp['cat_iterations'], 'depth': bp['cat_depth'], 'learning_rate': bp['cat_lr'],
               'random_seed': 42, 'verbose': 0, 'allow_writing_files': False}
    p_rf   = {'n_estimators': bp['rf_n'],   'max_depth': bp['rf_depth'],   'random_state': 42, 'n_jobs': -1}

    print('전체 데이터로 최종 재학습 중...')

    # K-Means 군집화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train[CLUSTER_COLS])
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    X_final = X_train.copy()
    X_final['Cluster_ID'] = kmeans.fit_predict(X_scaled)

    # 전처리
    preprocessor = get_preprocessor(X_final)
    X_ready = preprocessor.fit_transform(X_final)
    if hasattr(X_ready, 'toarray'):
        X_ready = X_ready.toarray()

    # 잔차 타겟
    y_resid = y_train - X_train[KEYTEL_COL]

    estimators = [
        ('xgb',  XGBRegressor(**p_xgb)),
        ('lgbm', LGBMRegressor(**p_lgbm)),
        ('cat',  CatBoostRegressor(**p_cat)),
        ('rf',   RandomForestRegressor(**p_rf)),
    ]

    reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=5,
        passthrough=True,
        n_jobs=-1,
    )
    reg.fit(X_ready, y_resid)

    model = StackingWrapper(
        stacking_model=reg,
        kmeans=kmeans,
        scaler=scaler,
        preprocessor=preprocessor,
        keytel_col=KEYTEL_COL,
        cluster_cols=CLUSTER_COLS,
    )

    joblib.dump(model, save_path)
    print(f'모델 저장 완료: {save_path}')

    return model
