"""
features.py
-----------
피처 엔지니어링 모듈

노트북에서 실험한 피처 생성 전략 3가지를 함수로 정리합니다.
- add_interactions      : 2차 교호작용 파생 변수
- add_poly_features     : 교호작용 + 다항식(2차, 3차) 파생 변수
- add_poly_and_ratios   : 교호작용 + 다항식 + 도메인 기반 비율 파생 변수 (최종 채택)
"""

from itertools import combinations
import pandas as pd


# 교호작용 및 다항식 대상 연속형 변수
INTERACTION_COLS = [
    'Exercise_Duration',
    'BPM',
    'Body_Temperature(F)',
    'Weight(kg)',
    'Age',
    'Height(cm)',
]

POLY_COLS = ['Exercise_Duration', 'BPM', 'Body_Temperature(F)']


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    연속형 변수 간 2차 교호작용(곱셈) 파생 변수를 생성합니다.
    물리적 관계(운동 시간 × 심박수 × 체중)를 반영한 3차 교호작용도 추가합니다.

    Parameters
    ----------
    df : 입력 데이터프레임

    Returns
    -------
    파생 변수가 추가된 데이터프레임 (원본 보존)
    """
    df = df.copy()

    for col1, col2 in combinations(INTERACTION_COLS, 2):
        name1 = col1.split('(')[0]
        name2 = col2.split('(')[0]
        df[f'Inter_{name1}_x_{name2}'] = df[col1] * df[col2]

    # 칼로리 소모 물리 공식 기반 3차 교호작용
    df['Inter_Full_Physics'] = (
        df['Exercise_Duration'] * df['BPM'] * df['Weight(kg)']
    )

    return df


def add_poly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    교호작용 파생 변수에 다항식(2차, 3차) 파생 변수를 추가합니다.

    Parameters
    ----------
    df : 입력 데이터프레임

    Returns
    -------
    파생 변수가 추가된 데이터프레임 (원본 보존)
    """
    df = add_interactions(df)

    for col in POLY_COLS:
        col_name = col.split('(')[0]
        df[f'Poly_{col_name}_Squared'] = df[col] ** 2
        df[f'Poly_{col_name}_Cubed'] = df[col] ** 3

    return df


def add_poly_and_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    교호작용 + 다항식 + 도메인 기반 비율 파생 변수를 생성합니다. (최종 채택 전략)

    도메인 비율 변수 설명:
    - HR_Load_Rate     : 연령 기반 최대 심박수 대비 현재 심박수 부하율
    - BPM_per_Weight   : 단위 체중당 심박수
    - Duration_per_Age : 연령 대비 운동 지속 능력 지표
    - Heating_Rate     : 운동 시간당 체온 상승률 (발열 속도)

    Parameters
    ----------
    df : 입력 데이터프레임

    Returns
    -------
    파생 변수가 추가된 데이터프레임 (원본 보존)
    """
    df = add_poly_features(df)

    df['HR_Load_Rate']     = df['BPM'] / (220 - df['Age'])
    df['BPM_per_Weight']   = df['BPM'] / (df['Weight(kg)'] + 1e-5)
    df['Duration_per_Age'] = df['Exercise_Duration'] / (df['Age'] + 1e-5)
    df['Heating_Rate']     = df['Body_Temperature(F)'] / (df['Exercise_Duration'] + 1e-5)

    return df