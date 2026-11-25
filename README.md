> **이 저장소는 비교용(구버전) 사이트예요.** 신버전([Annyang](https://github.com/byeolgog/Annyang))과 같은 코드·모델(LightGBM + 지연특성, AWS로 학습)이고, 예보는 **전부 단기예보**를 써요 (`config.USE_ULTRA_NEAR = False`).
> 신버전은 가까운 6시간에 초단기예보로 학습한 새 모델을 쓰고, 두 사이트의 '2시간 전 예측' 표로 병행 비교해요.
> 이전 랜덤포레스트 사이트 코드는 `legacy-rf-site` 브랜치에 그대로 보관돼 있어요.

# 오늘춥나?? — 안양대 캠퍼스 체감온도 AI

### [서비스 바로 보기](https://byeolgog.github.io/perceived-temperature-forAyu/)

2026 안양대학교 AI 경진대회 프로젝트(AI26-016, 팀 환전)입니다.

안양시 만안구 날씨를 안양대학교 캠퍼스의 기온·습도·풍속으로 보정하고, 기상청 체감온도 공식과 개인 추위 민감도를 적용해 캠퍼스에 머무는 동안 필요한 옷차림을 추천합니다.

## 현재 학습·평가 기준

| 구분 | 현재 기준 |
|---|---|
| 비교 모델 | RF, RF + 지연특성, LightGBM + 지연특성, LSTM, TSMixer |
| 학습 | 2022~2023년 |
| 검증 및 최종 모델 선택 | 2024년 |
| 회고적 테스트 | 2025년, 모든 후보 모델 평가 |
| 추가 외부 테스트 | 2026년, 안양대 격자 정답이 준비된 기간만 별도 평가 |
| 최종 선택 원칙 | 2024년 검증 종합점수 1위로 고정. 2025·2026년 결과로 다시 선택하지 않음 |
| 배포 | 같은 최고 모델을 Flask 로컬호스트와 `docs/` 정적 사이트에 반영 |

2025년 결과는 개발 중 이미 여러 번 확인했으므로 엄밀한 미공개 테스트가 아니라 **회고적 백테스트**로 표기합니다. 2026년은 현재 AWS 입력만 있고 캠퍼스 격자 정답이 없어 아직 모델 평가에 포함되지 않습니다.

> 기존 운영 RF의 참고 성능은 2023~2024년 학습 → 2025년 평가에서 체감온도 ±1°C 72.0%, ±2°C 90.8%였습니다. 이는 새 2022~2023년 학습 → 2024년 검증 비교 실험과 데이터 분할이 다르므로 새 모델 순위와 직접 섞지 않습니다.

새 다섯 모델의 최종 수치는 전체 Colab 학습 후 `validation_comparison.csv`와 `comparison_2025.csv`에 생성됩니다. 재학습이 끝나기 전에는 위 기존 RF 수치를 새 최고 모델의 결과로 해석하지 않습니다.

## 서비스 동작 방식

1. 기상청 만안구 초단기실황과 시간별 단기예보를 가져옵니다. 실패하면 Open-Meteo 또는 데모 데이터를 사용합니다.
2. 2024년 검증으로 선택한 최고 모델이 만안구 예보를 안양대 캠퍼스 기온·습도·풍속으로 보정합니다.
3. 최근 안양대 500m 격자 실측이 있으면 현재 오차를 앞으로의 예측에 `exp(-경과시간/4시간)`만큼 감쇠해 반영합니다.
4. 기상청 체감온도 공식과 사용자 추위 민감도(±2°C)를 적용합니다.
5. 캠퍼스 체류 시간 중 가장 불리한 체감온도를 기준으로 옷차림을 추천합니다.
6. GitHub Actions가 실측·예보·예측 기록을 갱신하고 GitHub Pages를 배포합니다.

체감온도는 10~4월에는 저온·바람 공식을, 5~9월에는 여름철 습도 공식을 사용합니다. 겨울에 풍속이 1.3m/s 기준선 부근이면 체감온도가 급변할 수 있어 화면에 범위로 표시합니다.

## 데이터 구성

| 경로 | 내용 | 현재 상태 |
|---|---|---|
| `data/train_data.csv` | 만안 AWS와 안양대 격자 정답을 합친 기존 자료 | 2023~2025년 |
| `data/22train.csv` | 2022년 만안 AWS 원자료 | 입력만 있음 |
| `data/grid/grid_2022_01.csv`~`grid_2022_12.csv` | 2022년 안양대 500m 격자 정답 | 12개월 준비됨 |
| `data/26data.csv` | 2026년 만안 AWS 원자료 | 2026-01-01~2026-11-21 부분 자료, 정답 없음 |
| `data/prepared/model_data.csv` | 위 자료를 학습 형식으로 합친 생성 파일 | 현재 2022~2025년, Git 제외 |
| `data/prepared/quality_report.json` | 연도별 기간·누락·결측·병합 상태 | 전처리할 때 갱신 |

`prepare_training_data.py`는 원본 CSV를 수정하지 않고 `data/prepared/`에 새 파일을 만듭니다. 2026년을 평가하려면 같은 기간의 안양대 격자 정답을 `data/grid/`에 추가한 뒤 다시 전처리해야 합니다. 2026년 자료가 1년 미만이어도 평가할 수 있지만 결과에는 실제 기간과 행 수를 함께 기록합니다.

## 비교 모델

| 모델 | 입력 | 체크포인트 | 학습 장치 |
|---|---|---|---|
| RF | 현재 시각 기온·습도·풍속, 시각·월 | `best.joblib` | CPU |
| RF + 지연특성 | 현재값, 1~24시간 지연값, 변화량, 이동통계 | `best.joblib` | CPU |
| LightGBM + 지연특성 | RF + 지연특성과 같은 입력 | `best.joblib` | CPU |
| LSTM | 최근 24시간 시퀀스 | `best.pt` | GPU 권장 |
| TSMixer | 최근 24시간 시퀀스 | `best.pt` | GPU 권장 |

모든 모델은 안양대와 만안구의 기온·습도·풍속 차이인 보정량을 예측합니다. RF와 LightGBM은 모두 결정트리 계열이지만, RF는 여러 독립 트리의 평균을 사용하고 LightGBM은 앞선 트리의 오차를 다음 트리가 순차적으로 보완합니다.

## Colab에서 전체 실행하기 — 권장

전체 실행 노트북은 프로젝트 폴더의 한 단계 위에 있습니다.

```text
C:\Users\prese\myvsc\perceived_temperature_colab.ipynb
```

1. `perceived-temperature-forAyu` 폴더와 `perceived_temperature_colab.ipynb`를 Google Drive에 올립니다.
2. Colab에서 노트북을 열고 **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다.
3. 첫 셀부터 순서대로 실행합니다. 노트북은 Drive 안의 프로젝트 폴더를 자동으로 찾습니다.
4. 동일한 이름의 프로젝트가 여러 개면 첫 코드 셀의 `PROJECT_DIR`에 정확한 Drive 경로를 입력합니다.

노트북은 다음 과정을 순서대로 수행합니다.

1. Drive 연결과 프로젝트 경로 확인
2. `requirements-train.txt` 설치와 GPU 확인
3. 2022년 격자 12개 확인, 없을 때만 수집
4. `data/prepared/model_data.csv` 생성과 연도 확인
5. 다섯 모델을 2022~2023년으로 학습하고 2024년으로 검증
6. 전체·세부·계절별 검증 비교표 출력
7. 2024년 종합점수 1위를 `selection.json`에 저장
8. 선택을 고정한 채 모든 모델을 2025년과 가능한 2026년에 평가
9. 최고 모델을 Flask와 정적 사이트에 배포하고 예측 일치 여부 확인

현재 저장소에는 2022년 격자 파일이 있으므로 API 키 없이 해당 단계를 건너뜁니다. 다시 수집해야 한다면 Colab 왼쪽의 **보안 비밀(Secrets)**에 `KMA_API_KEY`를 만들고 API허브 인증키를 넣습니다. 인증키를 코드·노트북·README에 직접 적거나 Git에 올리지 마세요.

## 로컬에서 학습·평가하기

Python 3.9 이상을 사용합니다.

### 1. 환경 설치와 데이터 준비

```powershell
git clone https://github.com/byeolgog/perceived-temperature-forAyu.git
cd perceived-temperature-forAyu
python -m pip install -r requirements-train.txt
python prepare_training_data.py --grid-dir data/grid
```

생성 결과에서 2022~2025년이 모두 보여야 합니다.

```powershell
python -c "import pandas as pd; d=pd.read_csv('data/prepared/model_data.csv'); print(pd.to_datetime(d.tm).dt.year.value_counts().sort_index())"
```

### 2. 모델별 학습과 2024년 검증

```powershell
python train_model.py --model rf       --data data/prepared/model_data.csv
python train_model.py --model rf-lag   --data data/prepared/model_data.csv
python train_model.py --model lightgbm --data data/prepared/model_data.csv
python train_model.py --model lstm     --data data/prepared/model_data.csv --seeds 3 --device auto
python train_model.py --model tsmixer  --data data/prepared/model_data.csv --seeds 3 --device auto
```

`--device auto`는 CUDA가 있으면 GPU를 사용하고 없으면 CPU를 사용합니다. 신경망 기본값은 최근 24시간, 시드 3개, 최대 60에폭, early stopping patience 8입니다.

### 3. 2024년 최고 모델 선택

```powershell
python compare_checkpoints.py --select-only --models rf rf-lag lightgbm lstm tsmixer
```

이 명령은 2025·2026년 데이터를 읽지 않고 2024년 검증 결과만 사용해 `compare_runs/checkpoints/selection.json`을 만듭니다.

### 4. 선택을 고정한 뒤 모든 모델 테스트

```powershell
python compare_checkpoints.py --data data/prepared/model_data.csv --test-year 2025 --device auto --models rf rf-lag lightgbm lstm tsmixer
```

2026년 안양대 격자 정답까지 합쳐진 경우에만 다음 명령을 추가로 실행합니다.

```powershell
python compare_checkpoints.py --data data/prepared/model_data.csv --test-year 2026 --device auto --models rf rf-lag lightgbm lstm tsmixer
```

테스트 표의 순위와 관계없이 최종 선택은 2024년에 만든 `selection.json` 그대로 유지됩니다.

## 로그와 결과 파일

| 경로 | 내용 |
|---|---|
| `compare_runs/logs/<모델>/train_<시각>.log` | 실행별 학습 로그. 재학습해도 이전 로그 유지 |
| `compare_runs/checkpoints/<모델>/best.*` | 모델별 최고 체크포인트 |
| `compare_runs/checkpoints/<모델>/metadata.json` | 설정, 데이터 해시, 검증 지표, 실행 환경 |
| `compare_runs/checkpoints/validation_comparison.csv` | 2024년 전체 검증 비교 |
| `compare_runs/checkpoints/validation_season_comparison.csv` | 2024년 계절별 검증 비교 |
| `compare_runs/checkpoints/selection.json` | 2024년 검증으로 고정한 최종 모델 |
| `compare_runs/checkpoints/comparison_<연도>.csv` | 해당 테스트 연도의 모든 모델 비교 |
| `compare_runs/checkpoints/comparison_<연도>_season_metrics.csv` | 해당 테스트 연도의 계절별 지표 |

학습 중에도 성능이 출력됩니다. RF·RF-lag는 25개 트리마다 종합점수·체감 MAE·±1°C 적중률을, LightGBM은 50회 부스팅마다 검증 오차를, LSTM·TSMixer는 매 에폭 검증 loss·현재 best·patience·학습률을 기록합니다.

## 종합점수

종합점수는 기상청 공식 지표가 아니라 후보 모델을 같은 기준으로 선택하기 위한 이 프로젝트의 0~100점 지표입니다. 점수 버전은 `overall_v2_season_balanced`입니다.

```text
MAE 점수 = 100 × clip(1 - 체감온도 MAE / 3°C, 0, 1)

종합점수 = 0.40  × 연중 ±1°C 적중률
         + 0.075 × 봄(3~5월) ±1°C 적중률
         + 0.075 × 여름(6~8월) ±1°C 적중률
         + 0.075 × 가을(9~11월) ±1°C 적중률
         + 0.075 × 겨울(12~2월) ±1°C 적중률
         + 0.15  × 연중 ±2°C 적중률
         + 0.15  × MAE 점수
```

| 항목 | 비중 | 목적 |
|---|---:|---|
| 연중 ±1°C 적중률 | 40% | 사용자가 체감하는 작은 오차를 가장 중요하게 반영 |
| 사계절 ±1°C 적중률 | 계절별 7.5%, 합계 30% | 표본이 많은 계절에 결과가 치우치지 않도록 보정 |
| 연중 ±2°C 적중률 | 15% | 큰 오차가 자주 나는 모델에 불이익 부여 |
| 체감온도 MAE 점수 | 15% | 임계값 밖의 연속적인 오차 크기 반영 |

10~4월 겨울 체감온도 공식 기간의 적중률도 별도로 출력하지만 종합점수에는 넣지 않습니다. 종합점수에는 일반 계절 구분인 겨울 12~2월 지표가 들어갑니다.

## 최고 모델 배포

2024년 최고 모델 선택과 테스트가 끝난 뒤 실행합니다.

```powershell
python deploy_best_model.py --data data/prepared/model_data.csv
python web_export.py
python verify_static_export.py --manifest data/deployed_campus_model/manifest.json --model-js docs/model.js --data data/prepared/model_data.csv
```

- `deploy_best_model.py`는 선택 체크포인트와 RF fallback을 `data/deployed_campus_model/`에 복사하고 72시간 추론을 검사합니다.
- `web_export.py`는 같은 최고 모델을 브라우저용 `docs/model.js`로 변환합니다.
- `verify_static_export.py`는 Python 체크포인트와 정적 모델의 예측 차이를 검사합니다.
- RF, RF-lag, LightGBM, LSTM, TSMixer 모두 Flask와 정적 사이트에서 사용할 수 있습니다.
- 지연특성·신경망 모델은 과거 문맥이 부족한 첫 구간만 함께 배포된 RF로 보완합니다.

### Flask 로컬호스트

```powershell
python server.py
```

- 서비스: `http://localhost:5000`
- 연결 모델 확인: `http://localhost:5000/api/model-status`

### 정적 사이트 로컬 확인

```powershell
python -m http.server 8000 --directory docs
```

- 정적 사이트: `http://localhost:8000`
- GitHub Pages에는 생성된 `docs/model.js`와 수정된 사이트 코드를 커밋해야 새 최고 모델이 반영됩니다.

## API 키와 보안

키가 없어도 Open-Meteo와 데모 데이터로 사이트를 실행할 수 있습니다. 실시간 기상청 자료를 사용하려면 `.env.example`을 `.env`로 복사하고 필요한 값만 설정합니다.

| 이름 | 사용 위치 | 용도 |
|---|---|---|
| `KMA_API_KEY` | Colab Secret 또는 로컬 `.env` | Colab에서는 API허브 과거 격자 수집 키, 로컬에서는 공공데이터포털 단기예보 키 |
| `KMA_APIHUB_KEY` | 로컬 `.env`, GitHub Actions Secret | 안양대 실측과 기상청 예보 갱신 |
| `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` | 로컬 `.env` | 추천 옷 검색 |
| `OPENAI_API_KEY` | 로컬 `.env` | 선택적 AI 문장 생성 |

`.env`, API 키, Colab Secret 값은 저장소·노트북 출력·README에 올리지 않습니다. 노출된 키는 API허브에서 재발급합니다.

## GitHub Pages 배포

`.github/workflows/pages.yml`은 `main` 브랜치 push, 수동 실행, 정기 실행 시 다음 작업을 수행합니다.

1. `KMA_APIHUB_KEY`로 안양대 실측과 만안구 실황·예보 수집
2. `docs/campus_now.json`, `docs/forecast.json`, `docs/history.json` 갱신
3. `docs/`를 GitHub Pages에 배포

처음 한 번 설정합니다.

1. GitHub 저장소의 **Settings → Secrets and variables → Actions**에서 `KMA_APIHUB_KEY`를 만듭니다.
2. **Settings → Pages → Source**를 **GitHub Actions**로 선택합니다.
3. **Actions → Pages (캠퍼스 실측 갱신) → Run workflow**를 실행합니다.

사이트 파일에는 인증키가 포함되지 않습니다. 실측 갱신에 실패하면 직전 공개 데이터를 유지하고, 데이터가 오래되면 화면은 AI 예측으로 전환합니다.

## 주요 파일

| 경로 | 역할 |
|---|---|
| `prepare_training_data.py` | 원본 AWS 정리, 안양대 격자 병합, 품질 보고서 생성 |
| `train_model.py` | 비교 모델 하나를 학습하고 최고 체크포인트·로그 저장 |
| `compare_models.py` | 공통 특성·모델·채점·종합점수 코드 |
| `compare_checkpoints.py` | 2024년 모델 선택과 연도별 전체 모델 테스트 |
| `deploy_best_model.py` | `selection.json`의 최고 모델을 Flask용으로 배포 |
| `deployed_model.py` | 다섯 모델의 공통 Flask 추론기 |
| `web_export.py` | 최고 모델을 `docs/model.js`로 변환 |
| `verify_static_export.py` | Python 모델과 정적 모델 예측 일치 검사 |
| `static_neural.py` | LSTM·TSMixer 정적 추론의 NumPy 기준 구현 |
| `service.py`, `server.py` | Flask 서비스와 API |
| `feels_like.py` | 캠퍼스 보정, 기상청 체감온도, 개인 민감도 계산 |
| `weather.py` | 기상청·Open-Meteo·데모 예보와 실측 수집 |
| `update_live_weather.py` | GitHub Pages용 실시간 JSON 생성 |
| `site_model.py` | 정적 브라우저 모델과 같은 방식의 기록용 추론 |
| `docs/index.html` | 정적 사이트 화면 |
| `docs/standalone.js` | 정적 사이트 로직과 다섯 모델 브라우저 추론 |
| `docs/model.js` | `web_export.py`로 생성한 정적 모델·설정 |
| `requirements-train.txt` | 서비스·LightGBM·PyTorch를 포함한 전체 의존성 |

## 노트북

| 노트북 | 용도 |
|---|---|
| `../perceived_temperature_colab.ipynb` | 준비·학습·비교·테스트·배포 전체 자동 실행 |
| `notebooks/1_campus_grid_collect.ipynb` | API허브 안양대 500m 격자 수집 |
| `notebooks/2_aws_377_clean.ipynb` | 안양만안 AWS(377) 원자료 정리 |
| `notebooks/3_merge_train_data.ipynb` | AWS와 격자 정답 병합 |
| `notebooks/4_train_campus_model.ipynb` | 기존 단일 RF 실험 |
| `notebooks/5_compare_models.ipynb` | 비교 모델 실험용 노트북 |

## 문제 해결

| 증상 | 확인할 내용 |
|---|---|
| `model_data.csv`가 없다고 나옴 | `python prepare_training_data.py --grid-dir data/grid`를 먼저 실행 |
| 2022년 학습 데이터가 없다고 나옴 | `data/grid/grid_2022_01.csv`~`grid_2022_12.csv` 존재 여부 확인 |
| 2026년 테스트 데이터가 없다고 나옴 | `26data.csv`만으로는 부족하며 같은 기간의 안양대 격자 정답 필요 |
| CUDA가 `False` | Colab 런타임을 T4 GPU로 변경 후 환경 설치 셀부터 다시 실행 |
| 정적 사이트가 예전 모델을 표시 | `deploy_best_model.py` → `web_export.py` 재실행 후 브라우저 강력 새로고침 |
| `/api/model-status`가 `legacy_rf` | 최고 모델 배포 명령을 실행해 `data/deployed_campus_model/manifest.json` 생성 |
| `Address already in use` | 기존 서버를 종료하거나 이미 열린 `http://localhost:5000` 사용 |

## 시연 팁

- 주소 끝에 `#analyze`를 붙이면 분석 화면부터 시작합니다.
- 하교 시간 알림은 브라우저 탭이 열려 있어야 작동합니다.
- 로그인과 사용자 식별 정보 수집은 사용하지 않습니다.
