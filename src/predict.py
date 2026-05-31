"""
predict.py
----------
추론 및 제출 파일 생성 모듈

학습된 모델로 테스트 데이터를 추론하고 제출용 CSV를 생성합니다.
"""

import os
import numpy as np
import pandas as pd
import joblib

MODEL_DIR      = './models'
SUBMISSION_DIR = './data/submission'


def predict(test_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    학습된 모델로 테스트 데이터를 추론하고 제출용 CSV를 생성합니다.

    Parameters
    ----------
    test_df    : 테스트 데이터프레임 (ID 컬럼 포함, 피처 엔지니어링 적용 후)
    model_name : 모델 파일명 (확장자 제외)

    Returns
    -------
    제출용 데이터프레임 (ID, Calories_Burned)
    """
    model_path = os.path.join(MODEL_DIR, f'{model_name}.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'모델 파일이 없습니다: {model_path}')

    print(f'모델 로드 중: {model_path}')
    model = joblib.load(model_path)

    submission_ids = test_df['ID']
    test_x = test_df.drop(columns=['ID'], errors='ignore')

    print('테스트 데이터 추론 중...')
    preds = model.predict(test_x)

    submission = pd.DataFrame({
        'ID': submission_ids,
        'Calories_Burned': preds,
    })

    # 음수 칼로리 하한값 보정 및 정수 반올림
    submission['Calories_Burned'] = np.round(
        submission['Calories_Burned'].clip(lower=0)
    ).astype(int)

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    save_path = os.path.join(SUBMISSION_DIR, f'submission_{model_name}.csv')
    submission.to_csv(save_path, index=False)

    print(f'제출 파일 저장 완료: {save_path}')
    print(submission.head())

    return submission