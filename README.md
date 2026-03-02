## 🏃‍♂️ 칼로리 소모량 예측 AI 모델 개발
### 프로젝트 개요 (Project Overview)
- **주제**: 체중, 성별, 운동 시간 심박 수 등 생체 데이터를 활용한 칼로리 소모량 예측 (회귀 알고리즘)
- **주최/주관**: 데이콘 (Dacon)
- **참가 대상**: [초격차] AI 헬스케어 머신러닝 트랙
- **목표**: 제공된 생체 데이터를 기반으로 개인의 건강과 라이프스타일 이해를 높일 수 있는 정교한 예측 머신러닝 모델 구축

---

### 기술 스택 (Tech Stack)
- **언어**: Python
- **데이터 분석 및 시각화**: Pandas, NumPy, Matplotlib, Seaborn
- **머신러닝 모델**: Scikit-Learn, XGBoost, LightGBM, CatBoost
- **최적화 도구**: Optuna

---

### 프로젝트 구조 (Project Structure)
```plaintext
├── data/
│   ├── train.csv                 # 학습용 데이터
│   ├── test.csv                  # 추론용 데이터
│   └── submission/               # 최종 제출 결과물 폴더
├── CaloriesBurned_Predict_김창현.ipynb  # 메인 모델링 및 분석 코드
└── README.md
```

---

### 데이터셋 상세 (Dataset Description)
본 프로젝트에서 사용한 데이터는 데이콘(Dacon)에서 제공한 개인별 생체 및 운동 기록 데이터입니다.  
총 15,000개(학습용 7,500개, 추론용 7,500개)의 샘플로 구성되어 있으며,  
각 샘플은 신체 정보와 운동 환경을 나타내는 10개의 독립 변수(Feature)와 1개의 종속 변수(Target)를 포함하고 있습니다.

#### 📊 데이터 구성 (Data Structure)
- **`train.csv`**: 학습용 데이터 (7,500 rows) - 독립 변수 및 목표 예측값 (`Calories_Burned` ) 포함
- **`test.csv`**: 추론용 데이터 (7,500 rows) - 모델 평가를 위한 블라인드 데이터
- **`sample_submission.csv`**: 최종 제출 양식 (`ID` 및 예측된 `Calories_Burned`)

#### 📝 변수 상세 (Feature Details)
데이터의 성격과 모델링 활용 목적에 따라 전체 변수를 4가지 논리적 그룹으로 분류했습니다. 
| 분류 (Category) | 변수명 (Column) | 설명 (Description) | 비고 (Note) |
| :--- | :--- | :--- | :--- |
| **식별자** | `ID` | 샘플 별 고유 식별자 | 모델 학습 시 제외 |
| **운동 지표** | `Exercise_Duration` | 운동 시간 (분 단위) | 연속형 수치 데이터 |
| | `BPM` | 심박수 | 운동 강도 파생 변수 활용 가능 |
| **신체 지표** | `Body_Temperature(F)` | 체온 (화씨) | 섭씨(℃) 변환 전처리 필요 |
| | `Height(Feet)` | 키 (피트 단위) | 인치와 결합하여 cm 단위 변환 필요 |
| | `Height(Remainder_Inches)` | 키 (나머지 인치) | 피트와 결합하여 cm 단위 변환 필요 |
| | `Weight(lb)` | 몸무게 (파운드 단위) | kg 단위 변환 전처리 필요 |
| | `Weight_Status` | 체중 상태 | 범주형 데이터 (Categorical) |
| **인적 사항** | `Gender` | 성별 | 범주형 데이터 (Categorical) |
| | `Age` | 나이 | 연속형 수치 데이터 |
| **타겟 (Target)**| `Calories_Burned` | **칼로리 소모량** | **최종 예측해야 할 종속 변수 (회귀)** |
---

### 탐색적 데이터 분석 및 피처 엔지니어링 (EDA & Feature Engineering)
운동 생리학 및 열역학 도메인 지식을 활용하여, 
칼로리 소모의 근본적인 메커니즘을 반영한 다수의 파생 변수를 기획하고 구축하였습니다.

#### 📏 물리 지표 표준화 (Physical Metrics Standardization)
- **단위 변환**: `Weight(lb)`를 kg으로, 분리된 `Height(Remainder_Inches)`를 cm 및 m로 통합 변환하여 데이터 스케일을 직관적으로 통일

#### 🧬 도메인 지식 기반 파생 변수 생성 (Domain Specific Feature Engineering)
- **기초 대사량 및 신체 지표 (BMR/RMR)**:
  - **`BMI (체질량지수)`**: 물리 단위가 통합된 키와 몸무게를 바탕으로 산출
  - **`BMR_ Harris` & `RMR_Mifflin`**: 성별, 나이, 체중, 키를 복합적으로 고려하는 대표적인 기초 대사량 추정 방식인 Harris-Benedict 및 Mifflin-St Jeor 공식을 각각 적용하여 개인별 고유 대사량을 수치화
    
- **생리학적 칼로리 소모 모델링 (Keytel & Theoretical Calories)**:
  - **`BPM_Multiplier`**: 훈련 데이터의 심박수 분위수(Quantile)을 4구간으로 나누어, 상대적인 운동 가중치를 부여
  - **`Session_Keytel_Calories`**: 심박수, 체중, 나이, 성별을 모두 고려하여 운동 중 소모되는 칼로리를 정교하게 추정하는 Keytel 공식을 적용하여, 모델에 생리학적 기준점 제공
  - **`Session_Harris_Calories`, `Session_Mifflin_Calories`**: 기초대사량 수치에 운동 시간과 강도 가중치를 곱하여 이론적 소모량을 다각도로 산출
    
- **구간화 및 세션 프로파일링 (Binning & Session Profiling)**:
  - **`Age_Range`**: 10년 단위로 분할하여 연령대별 비선형 패턴 범주화
  - **`Intensity_Level`**: 훈련 데이터의 심박수 분위수를 기준으로 운동 강도를 4등분하여 범주화
  - **`Duration_Category`**: 운동 시간을 3등분하여 구간별 특성 범주화
  - **`Session_Type`**: 운동 시간(15분 기준)과 심박수(95 BPM 기준)를 교차 결합하여, 세션의 성격을 4가지(`WarmUp`, `Power_Short`, `Endurance_Long`, `Athletic_Long`)로 프로파일링
    
- **열역학 지표 (Thermal Dynamics)**:
  - **`Temp_Excess`**: 정상 체온(98.6°F)을 기준으로 초과 발열량 도출
  - **`Thermal_Intensity`**: 운동 시간당 체온 상승 속도를 나타내는 열 생성률 산출
  - **`Heat_Energy_Proxy`**: 체중과 초과 체온을 곱하여 체질량 열에너지 산출
  - **`Is_Overheat`**: 104°F 이상의 고열 상태를 나타내는 이진(Binary) 변수를 추가하여 극단적 신체 부하 상태 명시
 
#### 🧮 수학적 및 통계적 특성 확장 (Mathmatical & Statistical Expansion)
- **교호작용 변수군**: 핵심 6개 연속형 변수간의 모든 2차 결합(2-way) 및 운동 시간 × 심박수 × 체중을 곱한 3차 교호작용 변수를 추가하여 물리적 메커니즘 수식화 
- **제곱, 세제곱 변수군**: 가속화되는 비선형 증가 추세를 모델이 온전히 학습할 수 있도록, 핵심 변수(`Exercise_Duration`, `BPM`, `Body_Temperature`)에 대한 2차항 및 3차항 확장
- **비율 파생 변수군**: 연령 대비 심박수 부하율, 단위 체중당 심박수, 연령 대비 운동 지속 능력 나누기 연산 기반의 비율 지수 산출 
