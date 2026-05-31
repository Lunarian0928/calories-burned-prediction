"""
utils.py
--------
공통 유틸리티 모듈

- seed_everything  : 재현성을 위한 랜덤 시드 고정
- get_preprocessor : 범주형/수치형 피처 전처리기 생성
"""

import os
import random
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


# 순서형 변수 인코딩 순서 정의
AGE_ORDER       = ['20s', '30s', '40s', '50s', '60s', '70s+']
INTENSITY_ORDER = ['Low', 'Moderate', 'High', 'Extreme']
DURATION_ORDER  = ['Short', 'Medium', 'Long']
SESSION_ORDER   = ['WarmUp', 'Power_Short', 'Endurance_Long', 'Athletic_Long', 'General']
WEIGHT_ORDER    = ['Normal Weight', 'Overweight', 'Obese']

ORDINAL_FEATURES = ['Age_Range', 'Intensity_Level', 'Duration_Category', 'Session_Type', 'Weight_Status']
NOMINAL_FEATURES = ['Gender']

ORDER_MAP = {
    'Age_Range':        AGE_ORDER,
    'Intensity_Level':  INTENSITY_ORDER,
    'Duration_Category': DURATION_ORDER,
    'Session_Type':     SESSION_ORDER,
    'Weight_Status':    WEIGHT_ORDER,
}


def seed_everything(seed: int = 42) -> None:
    """재현성을 위해 Python, NumPy, 환경변수 랜덤 시드를 고정합니다."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def get_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    데이터프레임의 컬럼 구성에 따라 전처리기를 생성합니다.

    - 순서형(Ordinal) 범주: OrdinalEncoder (도메인 지식 기반 순서 지정)
    - 명목형(Nominal) 범주: OneHotEncoder
    - 수치형(Numeric)     : StandardScaler

    Parameters
    ----------
    X_train : 학습 데이터프레임

    Returns
    -------
    fit되지 않은 ColumnTransformer 객체
    """
    ord_cols = [c for c in ORDINAL_FEATURES if c in X_train.columns]
    nom_cols = [c for c in NOMINAL_FEATURES if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in ord_cols + nom_cols]

    categories = [ORDER_MAP[c] for c in ord_cols]

    return ColumnTransformer(
        transformers=[
            ('ord_cat', OrdinalEncoder(categories=categories), ord_cols),
            ('nom_cat', OneHotEncoder(drop='first', handle_unknown='ignore'), nom_cols),
            ('num',     StandardScaler(), num_cols),
        ]
    )
