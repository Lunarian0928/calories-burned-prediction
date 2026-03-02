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

### 탐색적 데이터 분석 (Exploratory Data Analysis)
제공된 생체 및 운동 데이터의 시각적 분포를 확인하여 각 변수의 통계적 특성을 파악하고자 하였습니다.

#### 데이터 분포 (Data Distribution)
- **`Age` (연령)**: 20대에 가장 많은 빈도를 보이며, 연령이 증가할수록 빈도가 서서히 감소하는 우측 꼬리(Right-skewed) 형태 분포 확인

  <img width="590" height="390" alt="age_hist" src="https://github.com/user-attachments/assets/ee3d0bd8-b829-4e8b-b046-976f01777c10" />

- **`Body_Temperature` (체온)**: 104°F~106°F 구간에 데이터가 집중된 좌편향(Left-skewed) 분포 양상

  <img width="589" height="390" alt="body_temperature_hist" src="https://github.com/user-attachments/assets/118e0998-c0e1-4e62-af8c-73ef5aa043e5" />
  
- **`BPM` (심박수)**: 90~100 구간을 중심으로 하는 정규분포(Normal Distribution) 형태

  <img width="589" height="390" alt="bpm_hist" src="https://github.com/user-attachments/assets/866cfdcb-58fb-44c1-bf1b-c353f6bd5990" />

- **`Calories_Burned` (칼로리 소모량)**: 50 이하 저수치 구간에 밀집되어 있으며, 값이 커질수록 빈도가 감소하는 우측 꼬리(Right-skewed) 분포 형상
  
  <img width="590" height="390" alt="calories_burned_hist" src="https://github.com/user-attachments/assets/31fef6d0-0536-4a54-9040-22ae1114ba6b" />

- **`Exercise_Duration` (운동 시간)**: 1분부터 30분까지 특정 구간에 편향되지 않고 고르게 분포된 균등 분포(Uniform Distribution) 양상 확인
  
  <img width="589" height="390" alt="exercise_duration_hist" src="https://github.com/user-attachments/assets/eadbedfe-9551-4f33-9ff2-621ce7e3dc70" />

- **`Height` (키)**: 패트(`Feet`) 단위는 5피트와 6피트에 집중된 이산적 분포 형태이며, 나머지 인치(`Remainder_Inches`) 데이터는 0~12 사이에 넓게 퍼진 분포 확인

  <img width="590" height="390" alt="height_hist" src="https://github.com/user-attachments/assets/a8258218-aa20-4d46-8e28-2cbf547ffee4" />

- **`Weight` (체중)**: 140lb(약 63kg) 및 190lb(약 86kg) 부근에서 두 개의 정점을 갖는 쌍봉형(Bimodal) 분포 확인

  <img width="589" height="390" alt="weight_hist" src="https://github.com/user-attachments/assets/92d40ea4-5e6d-45dd-9a67-ec737d1ffbdc" />

#### 범주형 변수에 따른 관계 분석 (Relationship Analysis by Categorical Variables)
- **성별(`Gender`)에 따른 생체 및 운동 지표**: 연령 및 운동 시간의 중앙값은 남녀가 동일(각 39세, 15분)하지만, 체중과 키 등 물리적 지표는 남성이 여성보다 확연히 높은 수치 기록
  <table>
    <tr>
      <td align="center"><img src="https://github.com/user-attachments/assets/6eb8c4db-33f5-4053-90a5-cd796784d43f" alt="age_by_gender" width="100%" /></td>
      <td align="center"><img src="https://github.com/user-attachments/assets/eacd0303-3711-439a-a49d-54ffb371f75f" alt="exercise_duration_by_gender" width="100%" /></td>
    </tr>
    <tr>
      <td align="center"><img src="https://github.com/user-attachments/assets/6e07d613-7289-4431-a960-11dfe5f95ba1" alt="height_by_gender" width="100%" /></td>
      <td align="center"><img src="https://github.com/user-attachments/assets/4340ec37-f46b-4cd0-8003-b7416c1c0651" alt="weight_by_gender" width="100%" /></td>
    </tr>
  </table>

- **성별(`Gender`)에 따른 칼로리 소모량**: 신체 지표의 뚜렷한 차이에도 불구하고 칼로리 소모량의 중앙값은 여성(78.00)이 남성(76.00)보다 근소하게 높은 수치 유지
  <br>
  <img width="590" height="390" alt="calories_burned_by_gender" src="https://github.com/user-attachments/assets/af6d4927-b4e4-4612-b215-e8360369e3d1" />

- **체중 상태(`Weight_Status`)에 따른 심박수 및 소모량**: 체중 상태와 관계없이 심박수 중앙값은 94~96 구간에서 유사하게 형성되며, 칼로리 소모량은 과체중(Overweight) 그룹이 중앙값 81.00으로 가장 높은 수치 확인
  <table>
    <tr>
      <td align="center"><img src="https://github.com/user-attachments/assets/679ec414-926e-4880-a9af-a926872a8521" alt="bpm_by_weight_status" width="100%" /></td>
      <td align="center"><img src="https://github.com/user-attachments/assets/12450b8e-1ae2-43d9-a2a9-8596a0a714be" alt="calories_burned_by_weight_status" width="100%" /></td>
    </tr>
  </table>

- **체중 상태(`Weight_Status`)에 따른 키 분포**: 과체중(Overweight) 및 비만(Obese) 그룹의 키(피트) 중앙값이 정상 체중(Normal Weight) 그룹보다 상대적으로 높게 형성된 경향 파악
  <table>
    <tr>
      <td align="center"><img src="https://github.com/user-attachments/assets/e71e6eec-8ce8-4149-ae97-677e52904325" alt="height_by_weight_status" width="100%" /></td>
      <td align="center"><img src="https://github.com/user-attachments/assets/83fd2b5b-d43f-4a8b-b39c-fedbd236a402" alt="remainder_inches_by_weight_status" width="100%" /></td>
    </tr>
  </table>

---

### 의료 도메인 지식 및 분석 
칼로리 소모 메커니즘을 규명하기 위해, 전문 의학 및 보건학 문헌을 바탕으로 가설을 수립하고 파생 변수 생성의 근거로 사용하였습니다.

- **칼로리 소모와 체중의 관계**: 체중 변화와 에너지 소모량은 단순 비례하지 않으며, 신체의 대사 적응 및 제지방량 변화에 의해 조절됩니다.  
이를 반영하여 기초대사량 기반의 `BMR_Harris`, `RMR_Mifflin` 파생 변수를 도출하였습니다.

  - 🔗 **Reference**: Manfred J. Müller et al., *Changes in Energy Expenditure with Weight Gain and Weight Loss in Humans* [[Link](https://d-nb.info/1120450403/34)]

- **칼로리 소모와 체온의 관계**: 신체의 에너지 전환은 열을 발생시키며, 체온은 신진대사율과 연동하여 변환합니다.  
골격근 대사에 의한 초과 발열량을 포착하기 위해 `Temp_Excess` 및 `Thermal_Intensity`를 산출하였습니다.

  - 🔗 **Reference**: Bolt Pharmacy, *Metabolic Rate and Body Temperature: Understanding Their Vital Relationship* [[Link](https://www.boltpharmacy.co.uk/guide/what-is-the-relationship-between-metabolic-rate-and-body-temperature)]

- **칼로리 소모와 심박수(BPM)의 관계**: 심박수는 신체의 활동량과 운동 강도를 객관적으로 수치화하는 핵심 생리 지표입니다.  
이를 근거로 `BPM_Multiplier` 가중치를 부여하고 `Session_Keytel_Calories` 기준점을 설정하였습니다.

  - 🔗 **Reference**: Cleveland Clinic, *What To Know About Exercise and Heart Rate Zones* [[Link](https://health.clevelandclinic.org/exercise-heart-rate-zones-explained)]
    
- **칼로리 소모와 연령의 관계**: 연령이 증가하면 제지방량 및 주요 장기 질량이 감소하여 기초 대사율이 하락합니다.  
연령에 따른 근본적인 대사 차이를 반영하기 위해 `Age_Range`를 범주화하고 비율 지수를 산출하였습니다.

  - 🔗 **Reference**: PubMed Central, *Energy Expenditure and Aging* [[Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC2818133/)]
    
- **칼로리 소모와 성별의 관계**: 성별은 제지방량과 체지방량 구성비의 차이를 결정짓는 주요 생물학적 요인입니다.  
성별 기반의 총 에너지 소비량 격차를 반영하기 위해 관련 수식을 분리하여 적용하였습니다.

  - 🔗 **Reference**: PubMed Central, *Sex Differences in Measures of Energy Expenditure and Body Composition in Young, Middle-Aged, and Older Adults* [[Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC12830153/)]

---

### 피처 엔지니어링 (Feature Engineering)
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

---

### 모델링 및 하이퍼파라미터 최적화 (Modeling & Hyperparameter Optimization)
생리학적 공식과 머신러닝 알고리즘을 결합한 하이브리드 아키텍처를 설계하고, 스태킹 앙상블을 통해
최종 예측 오차(RMSE)를 최소화했습니다.

#### 🔨 핵심 모델링 구축 과정 (Core Modeling Construction Process)
- **베이스라인 모델 구축 및 검증**: XGBoost를 기반으로 Optuna를 활용해 하이퍼파라미터를 탐색하고,  
K-Fold 교차 검증을 도입하여 단일 검증의 과적합 위험성 방지 및 모델 일반화 성능 확보
- **잔차 집중 학습 아키텍처 도입**: 독립 변수를 통한 타겟 직접 추정 방식 대신, 도메인 지식(Keytel 공식) 예측값을 뺀 잔차(Residual)만을  
트리 모델이 학습하도록 설게하여 비선형 패턴 포착 능력 극대화
- **이기종 앙상블 및 동시 최적화**: 단일 알고리즘의 한계를 극복하기 위해 부스팅 계열(XGBoost)과 배깅 계열(Random Forest)을 결합하고,  
두 알고리즘의 하이퍼파라미터 동시 튜닝을 통해 상호 보완적 아키텍처 구축
- **하이브리드 스태킹 파이프라인 완성**: 파이프라인 내부에서 K-Means 군집화 피처를 동적 생성하고,  
다중 부스팅 모델(XGBoost, LightGBM, CatBoost)과 Random Forest를 결합한 스태킹 앙상블 래퍼 구축
- **수학적 파생 변수 확장을 통한 고도화**: 스태킹 모델의 설명력을 끌어올리기 위해 피처 간 교호작용,  
다항식, 비율 파생 변수를 순차적으로 추가하여 잔차 예측 정밀도 최종 고도화

#### 📈 실험 단계별 성능 추이 (Performance Trends by Experimental Stage)
모델 아키텍처 및 파생 변수 확장 단계에 따른 검증 데이터(Validation) 및 데이콘 리더보드 실제 제출(Test) RMSE 변화를 기록했습니다.

---

### 최종 모델 상세 
단일 모델의 한계를 극복하고 예측의 정밀도를 한계까지 끌어올리기 위해,  
도메인 지식과 머신러닝의 강점을 결합한 **군집 기반 잔차 선행 학습 스태킹(Stacking) 아키텍처**를 최종 모델로 구축했습니다.

#### ✏️ 모델 구조 및 학습 메커니즘
- **데이터 분할 및 군집화 (Data Splitting & Clustering)**:
  - **교차 검증 분할**: 5-Fold 교차 검증을 통해 OOF 방식으로 데이터를 분할해 일반화 성능을 확보했습니다.
  - **K-Means 군집화**: 운동 시간(`Exercise_Duration`), 심박수(`BPM`), 체온(`Body_Temperature(F)`) 변수를 스케일링(StandardScaler)한 후,  
  K-Means 알고리즘(`n_cluster=5`)을 적용하여 운동 생리적 특성을 5단계로 세분화한 `Cluster_ID` 피처를 생성했습니다.

- **1차 모델(Base Models)의 잔차 선행 학습**:
  - **학습 목표**: 1차 모델인 XGBoost, LightGBM, CatBoost, Random Forest는 타겟값을 직접 예측하지 않습니다.  
  기존 도메인 공식(Keytel)이 설명하지 못하는 비선형적 오차인 잔차($y_{residual}$)만을 타겟으로 설정하여 선행 학습을 수행했습니다. 
  - **잔차 산출 수식**: $y_{residual} = Acutal \ Calories - Keytel \ Calories$
    
- **2차 모델(Meta Model)의 통합 학습 및 잔차 보정**:
  - **메타 피처 통합**: 1차 모델들의 예측 결과(Meta Features)만 사용하는 일반적인 스태킹과 달리,  
  passthrough=True` 속성을 적용하여 원본 피처(Original Features)와 군집화 정보(Cluster Info)를 2차 메타 모델(RidgeCV)에 통합 전달했습니다. 
  - **정밀 보정**: 메타 모델은 각 1차 모델이 어떤 군집 및 피처 조건에서 오차를 발생시키는지 파악하여 잔차를 정밀하게 보정합니다.
  - **최종 보정 수식**: $y_{residual} = RidgeCV(Meta\ Features + Cluster\ Info + Original\ Features)$
    
- **앙상블 가중치 역할 분담 및 최종 칼로리 산출**:
  - **메타 모델(RidgeCV)**이 최적화한 앙상블 가중치는 각 1차 모델에 고유한 구조적 역할을 부여합니다.
    - **Random Forest (가중치 0.8872)**: 다수 파생 변수로 인한 과대적합을 배깅 방식으로 억제하여 안정적인 예측 기준선을 제공합니다. 
    - **LightGBM (가중치 0.0580) & CatBoost (가중치 0.0565)**: 기준선 위에서 깊은 트리를 통해 군집별 미세한 잔차 패턴을 정밀 보정하고 노이즈를 제어합니다.
    - **XGBoost (가중치 -0.0081)**: 특정 비선형 구간의 과대적합 경향을 음수 가중치로 억제하여 오차 편차를 방어합니다. 
 - **최종 칼로리 산출** : 메타 모델이 최종 예측한 잔차에 기본 공식(Keytel) 예측값을 더하여 최종 칼로리 소모량을 복원했습니다.  
 $Predicted\ Calories = y_{residual} + Keytel\ Calories$

#### 🔣 1차 모델별 하이퍼파라미터 세부 최적화 결과 및 구조적 의미 (Base Models Deatailed Hyperparameter Optimization)
Optuna를 활용하여 4개의 1차 모델이 가진 고유한 하이퍼파라미터 탐색 공간을 설정하고,  
교차 검증을 통해 파생 변수의 과적합을 제어하며, 잔차를 정밀하게 학습할 수 있는 최적의 조합을 도출했습니다. 

- **XGBoost (피처 의존성 분산 및 비선형 패턴 매핑)**:

  - **`n_estimators`**: 500~1500 범위 중 **1401**을 도출했습니다.
  충분한 반복 학습을 통해 군집 및 잔차의 미세한 비선형 패턴을 매핑합니다.

  - **`max_depth`**: 3~10 범위 중 **10**을 도출했습니다.  
트리 깊이를 최대치로 허용하여 복잡한 교호작용 변수 간의 다차원적 관계를 포착합니다.

  - **`learning_rate`**: 0.01~0.1 범위 중 **0.0886**을 도출했습니다.  
  비교적 높은 학습률을 설정하여 깊은 트리의 학습 속도 및 수렴 과정을 최적화합니다.

  - **`subsample`**: 0.6~1.0 범위 중 **0.9573**을 도출했습니다.  
  매 트리마다 데이터의 약 95.7%를 샘플링하여 데이터 손실을 최소화하면서도 미세한 과적합을 방어합니다.

  - **`colsample_bytree`**: 0.6~1.0 범위 중 **0.6733**을 도출했습니다.  
  매 트리마다 약 67.3%의 피처만 참조하게 하여,  
  특정 강력한 파생 변수에 대한 모델의 과도한 의존성을 억제하고 트리의 다양성을 확보합니다.

- **LightGBM (잔차 정밀 학습 및 트리 복잡도 통제)**: 

  - **`n_estiamtors`**: 500~1500 범위 중 **1480**을 도출했습니다.  
  다수의 반복 학습을 통해 잔차 보정의 정밀도 한계치를 넘었습니다.

  - **`max_depth`**: 3~15 범위 중 **13**을 도출했습니다.    
  깊은 트리를 형성하여 군집별로 발생하는 세부적인 오차 패턴을 정밀하게 학습합니다.

  - **`learning_rate`**: 0.01~0.1 범위 중 **0.0399**를 도출했습니다.  
  상대적으로 낮은 학습률을 적용하여 깊은 트리 구조에서 발생할 수 있는 과적합을 통제하며 안정적인 수렴을 유도합니다.

  - **`num_leaves`**: 20~100 범위 중 **34**를 도출했습니다.  
  최대 깊이가 13임에도 리프 노드 수를 34로 강하게 제한하여,  
  트리의 복잡도를 통제하고 모델의 일반화 성능을 극대화합니다.
    
- **CatBoost (대칭 트리 기반 예측 편향 방어)**:
  
  - **`iterators`**: 500~1500 범위 중 **1404**를 도출했습니다.  
  충분한 반복을 통해 데이터 내 잔여 노이즈를 깊이 있게 학습하고 예측을 세밀화합니다.

  - **`depth`**: 4~10 범위 중 **9**를 도출했습니다.    
  대칭 트리(Oblivious Tree) 구조에서 9의 깊이를 가짐으로써 데이터의 다차원적 상호작용을 균형 있게 포착합니다.

  - **`learning_rate`**: 0.01~0.1 범위 중 **0.0334**를 도출했습니다.  
  낮은 학습률을 통해 대칭 트리의 점진적이고 안정적인 잔차 보정을 유도합니다.
    
- **Random Forest (안정적 기준선 형성 및 노이즈 차단)**:
  
  - **`n_estimators`**: 100~300 범위 중 **219**를 도출했습니다.  
  219개의 독립적인 트리를 배깅(Bagging)하여 예측의 분산을 감소시키고 메인 기준선의 안정성을 확보합니다. 
  - **`max_depth`**: 5~20 범위 중 **8**을 도출했습니다.   
  부스팅 계열과 달리 트리 깊이를 8로 얕게 통제하여,  
  수많은 파생 변수 환경에서 발생할 수 있는 노이즈 암기를 원천 차단하고  
  가장 강건한 1차 예측치를 생성합니다. 
