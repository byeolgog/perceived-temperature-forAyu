# 오늘춥나?? — 안양대 캠퍼스 체감온도 & 옷차림 추천

2026 안양대학교 AI 경진대회 (AI26-016, 팀 환전)

날씨앱 예보(안양시 만안구 기준)를 **AI 모델로 안양대 캠퍼스 날씨**로 보정해 체감온도를 계산하고,
캠퍼스에 머무는 시간대 중 가장 추운 때를 기준으로 **옷차림을 추천**합니다.

## 다운로드하고 실행하기

### 0. 준비물 (처음 한 번)

- **Python 3.9 이상**: [python.org](https://www.python.org/downloads/)에서 설치
  - 설치 첫 화면에서 **"Add python.exe to PATH"를 꼭 체크**하세요.
  - 확인: 터미널(PowerShell)에서 `python --version` 입력 → `Python 3.x.x`가 나오면 OK
- **Git**(선택): [git-scm.com](https://git-scm.com/)
  - Git이 없으면 아래 '방법 B'처럼 ZIP으로 받으면 됩니다.

### 1. 코드 내려받기

**방법 A — Git 사용**
```powershell
git clone https://github.com/HIsabelle/onulchupna.git
cd onulchupna
```

**방법 B — ZIP 다운로드**
1. 깃허브 저장소 페이지에서 초록색 **`<> Code`** 버튼 → **Download ZIP**
2. 압축을 풀고, 풀린 폴더에서 빈 곳을 **우클릭 → "터미널에서 열기"** (또는 VSCode로 폴더를 열고 상단 메뉴 **터미널 → 새 터미널**)

### 2. 필요한 라이브러리 설치 (처음 한 번, 1~3분)

```powershell
pip install -r requirements.txt
```

### 3. AI 모델 학습 (처음 한 번, 약 20초)

```powershell
python train_campus.py
python train.py
```

- `train_campus.py`: 캠퍼스 보정 모델. 2025년으로 채점한 적중률을 보여주고 `data/campus_model.joblib`을 만듭니다.
- `train.py`: 옷차림 모델. `정확도 … 판단 기준 7/10 (70%) → 통과`가 나오면 완료입니다.

모델 파일(`data/*.joblib`)은 용량이 커서 깃허브에 올리지 않으므로, 받은 컴퓨터마다 한 번씩 실행해야 합니다.
(캠퍼스 모델은 없으면 서버가 켜질 때 `data/train_data.csv`로 자동 학습합니다.)

### 4. 실행

```powershell
python server.py
```

`오늘춥나?? → http://localhost:5000`이 나오면 브라우저에서 **http://localhost:5000** 을 여세요.

- **끄기**: 터미널에서 `Ctrl + C`. 실행 중에는 그 터미널 창을 닫지 마세요.
- **다음부터는** 4번(`python server.py`)만 하면 됩니다.

### 자주 나는 문제

| 증상 | 해결 |
|---|---|
| `python`을 찾을 수 없음 | Python 설치 때 PATH 체크를 안 한 경우 → 재설치하며 체크, 또는 `py server.py` |
| `No module named 'flask'` 등 | 2번 `pip install -r requirements.txt`를 다시 실행 |
| `Address already in use` | 이미 서버가 켜져 있음 → 그냥 브라우저에서 열기 |
| 화면이 예전 그대로 | 서버를 껐다 켜고 브라우저에서 `Ctrl + F5` |
| 상단에 "기상청 API 키가 없어…" | 정상 동작입니다(Open-Meteo 사용). 기상청 값을 쓰려면 아래 API 키 설정 |

## 설치 없이 보기 — HTML 버전

파이썬 설치 없이 **`docs/index.html`을 더블클릭**하면 브라우저에서 바로 열립니다.
서버가 없으면 자동으로 '웹 체험판'으로 동작해요.

| 기능 | 서버 버전 (`python server.py`) | HTML 버전 (파일·GitHub Pages) |
|---|---|---|
| 날씨 | 기상청(키 있을 때) → Open-Meteo | Open-Meteo |
| 체감온도 계산 | ✅ | ✅ 같은 계산식 |
| 캠퍼스 AI 보정 | Random Forest 100그루 × 3 | Random Forest 20그루 × 3 축소판 |
| 지금 캠퍼스 체감온도 (홈 카드) | 500m 격자 실측(키 있을 때) → 없으면 AI 예측 | GitHub Actions가 매시간 받은 실측 → 없으면 AI 예측 |
| AI 옷차림 모델 | Random Forest 200그루 | Random Forest 30그루 축소판 (정확도 비슷) |
| 쇼핑몰 예시 이미지 | 네이버 API 키 있으면 사진 | 검색 링크만 |
| 학생 체감 제보 | 모두의 제보를 모아 통계·재학습 | 이 기기 브라우저에만 저장 |

### GitHub Pages로 인터넷에 공개하기 (+ 캠퍼스 실측 매시간 갱신)

`.github/workflows/pages.yml`이 main에 올라올 때마다, 그리고 **매시간** 사이트를 배포합니다.
배포할 때 기상청 API허브에서 안양대 500m 격자 최신값을 받아 `campus_now.json`으로 같이 올려서,
홈 화면 '지금 캠퍼스 체감온도'에 실측값이 나옵니다.

처음 한 번만 설정하세요.

1. **인증키를 비밀값으로 저장**: Settings → Secrets and variables → Actions → **New repository secret**
   - Name: `KMA_APIHUB_KEY` / Secret: 기상청 API허브 인증키
   - 비밀값은 저장한 사람도 다시 볼 수 없고, 사이트 파일에는 키가 들어가지 않습니다.
2. **배포 방식 바꾸기**: Settings → Pages → Build and deployment → Source를 **GitHub Actions**로 선택
3. **첫 실행**: Actions 탭 → 'Pages (캠퍼스 실측 갱신)' → **Run workflow**
4. 1~2분 뒤 `https://<아이디>.github.io/<저장소 이름>/` 에서 누구나 접속 가능

- 실측값 받기에 실패하면 직전 값을 그대로 두고 다음 시간에 다시 시도합니다. 실측이 6시간 넘게 갱신되지 않으면 화면은 AI 예측으로 바뀝니다.
- GitHub은 60일 동안 저장소에 변경이 없으면 매시간 실행을 멈춥니다. Actions 탭에서 다시 켤 수 있어요.
- 파일을 더블클릭해서 열 때는 실측 파일을 읽을 수 없어 AI 예측이 나옵니다.

## API 키 설정 (선택)

키가 없어도 실행됩니다. 기상청 실데이터·옷 예시 이미지를 쓰려면:

1. 폴더의 `.env.example`을 복사해 이름을 **`.env`** 로 바꾸기
2. 발급받은 키를 `=` 뒤에 붙여넣기 (예: `KMA_API_KEY=abcd1234...`)
3. 서버를 껐다 다시 켜기

> ⚠️ `.env`에는 비밀 키가 들어 있으므로 깃허브에 올리거나 메신저로 공유하지 마세요. (`.gitignore`로 업로드 제외되어 있음)


| 키 | 용도 | 발급처 | 없으면 |
|---|---|---|---|
| `KMA_API_KEY` | 기상청 단기예보 + 초단기실황 | data.go.kr '기상청_단기예보 조회서비스' (Decoding 키) | Open-Meteo 사용 |
| `KMA_APIHUB_KEY` | 위와 동일 (API허브 경로) + **안양대 500m 격자 실측값** (홈 화면 '캠퍼스 체감온도') | apihub.kma.go.kr (고해상도 격자자료 활용신청 필요) | 캠퍼스 실측값 표시 안 함 |
| `NAVER_CLIENT_ID/SECRET` | 추천 옷 예시 이미지 | developers.naver.com 검색 API | 쇼핑몰 검색 링크만 표시 |
| `OPENAI_API_KEY` | 'AI 한마디' 문장 | platform.openai.com | 템플릿 문장 |

키가 없거나 틀리면 화면 상단에 이유가 표시되고 자동으로 대체 소스를 씁니다.
인터넷이 아예 없으면 데모 데이터로 동작하므로 시연이 끊기지 않습니다.

## 구조

| 파일 | 역할 |
|---|---|
| `config.py` | 캠퍼스·날씨앱 기준 좌표, 모델·데이터 경로, 민감도 보정 등 설정 |
| `weather.py` | 기상청 / Open-Meteo / 데모 예보 수집 + 안양대 500m 격자 실측값 |
| `campus_model.py` | **캠퍼스 AI 모델**(랜덤포레스트, 만안 → 안양대 기온·습도·풍속) + 기상청 체감온도 공식 |
| `train_campus.py` | 캠퍼스 모델 학습·채점: `python train_campus.py [--data CSV] [--test-year 2025]` |
| `data/train_data.csv` | 캠퍼스 모델 학습 데이터 (안양만안 AWS 377 ↔ 안양대 500m 격자, 2023~2025년 시간별 26,150줄) |
| `feels_like.py` | 캠퍼스 체감 계산: 지난 시각은 실측, 남은 시각은 AI 예측 → 기상청 공식 → 개인 민감도 |
| `outfit.py` | 기온별 옷차림 8단계 × 상의·하의·신발·의류 잡화 |
| `model.py` | Random Forest / XGBoost 옷차림 분류, 학생 제보 반영 |
| `train.py` | 학습: `python train.py [--aws AWS.csv] [--model xgb]` |
| `service.py` | 화면용 계산 묶음 (분석 결과·제보 통계) |
| `server.py` | Flask 웹 서버 + API (`/api/home`, `/api/analyze`, `/api/feedback`, `/api/products`) |
| `docs/index.html` | 예선 자료 화면 구상 그대로: 시작 → AI 분석 중 → 결과 (서버·HTML 버전 공용) |
| `docs/standalone.js` | 서버 없이 브라우저에서 계산하는 HTML 버전 로직 |
| `docs/model.js` | HTML 버전용 작은 AI 모델 + 설정값 (`python web_export.py`로 생성) |
| `update_campus_now.py` | 안양대 500m 격자 최신값 → `docs/campus_now.json` (GitHub Actions가 매시간 실행) |
| `.github/workflows/pages.yml` | GitHub Pages 배포 + 캠퍼스 실측 매시간 갱신 |
| `web_export.py` | 웹용 모델 내보내기 — `config.py`·`outfit.py`·`train_data.csv`를 고쳤으면 다시 실행 |

## 사용하는 AI 모델

### 캠퍼스 보정 모델: Random Forest 회귀 (scikit-learn)

날씨앱이 보여주는 만안구 날씨로 **안양대 캠퍼스의 기온·습도·풍속**을 예측합니다.
고도·바람 배율 같은 추정 계수는 쓰지 않고, 3년치 실제 기상 데이터에서 두 지점의 관계를 배웁니다.

| 항목 | 내용 |
|---|---|
| 종류 | 지도학습 · 회귀 · 정답마다 모델 1개 (기온·습도·풍속) |
| 입력 (7개) | 만안 기온·습도·풍속, 시간대·월 (sin·cos로 순환 표현) |
| 정답 | 안양대 500m 격자(기상청 고해상도 분석) 기온·습도·풍속 — 실제로는 '안양대 − 만안' 보정량을 학습 |
| 학습 데이터 | 안양만안 AWS(377) 관측 ↔ 안양대 500m 격자, 2023~2025년 시간별 26,150줄 |
| 크기 | 정답마다 나무 100그루 (서버) / 20그루 축소판 (HTML 버전) |
| 성능 (2025년 시험) | 체감온도 ±1°C 적중 **72.0%** (보정 안 함 56.7%), ±2°C 적중 **90.8%** (보정 안 함 71.8%) |
| 한계 | 겨울은 풍속이 1.3 m/s 기준선을 넘나들면 체감온도가 크게 바뀌어 정확도가 낮음 → 기준선 근처는 화면에 범위로 표시 |

체감온도는 **기상청 공식 그대로** 계산합니다 (10~4월 바람: 기온 10°C 이하·풍속 1.3 m/s 이상일 때만, 5~9월 습도).

### 옷차림 모델: Random Forest 분류기 (scikit-learn)

결정트리 여러 개가 각자 옷차림 단계를 고르고, 다수결로 최종 결과를 정하는 머신러닝 모델입니다.
표 형태 데이터(기온·풍속 같은 숫자)에 강하고, 데이터가 적어도 안정적이며, 무료입니다.

| 항목 | 내용 |
|---|---|
| 종류 | 지도학습 · 다중 분류 (8단계) |
| 라이브러리 | scikit-learn `RandomForestClassifier` (선택: XGBoost `--model xgb`) |
| 크기 | 서버 버전 트리 200개 / HTML 버전 트리 30개 |
| 입력 (8개) | 캠퍼스 기온, 캠퍼스 풍속, 캠퍼스 습도, 개인 체감온도, 시각, 월, 추위 민감도, 강수확률 |
| 출력 | 옷차림 단계 0(한여름)~7(한겨울) → 상의·하의·신발·의류 잡화 추천 |
| 학습 데이터 | 안양 기후를 흉내 낸 합성 날씨 6,000시간 × 민감도 3종 = 14,400건 (+ 기상청 AWS CSV 가능) |
| 라벨 | '기온별 옷차림 표' 규칙 + 사람마다 다른 체감(±1.2°C 노이즈) |
| 재학습 | 학생 제보 "이렇게 입었는데 추워요/더워요" → 라벨 한 단계 수정, 5배 가중치 |
| 성능 | 단계 정확히 일치 약 81%, ±1단계 이내 99.9% (검증 데이터 20%) |

> ⚠️ 현재 정확도는 **규칙으로 만든 라벨** 기준이라 참고용입니다. 본선 검증(7/10)은
> 실제 사용자 5명의 제보로 다시 측정해야 의미가 있습니다.

### 전체 처리 흐름

1. **날씨 수집**: 기상청 단기예보(만안구 격자, 또는 Open-Meteo) 시간대별 기온·풍속·습도·강수확률
2. **캠퍼스 날씨** (AI): 남은 시각은 캠퍼스 보정 모델이 안양대 기온·습도·풍속을 예측, 이미 지난 시각은 안양대 500m 격자 실측값 사용
3. **체감온도 계산** (공식): 기상청 체감온도식 → 추위 민감도(±2°C). 겨울 풍속이 기준선 근처면 범위로 표시
4. **옷차림 분류** (AI): 위 값을 Random Forest에 넣어 시간대별 단계 예측 → 가장 두껍게 입어야 하는 시간 기준으로 추천
5. **AI 한마디**: `OPENAI_API_KEY`가 있으면 GPT(gpt-4o-mini)가 문장 생성, 없으면 템플릿 문장

### 앞으로: Air Fit 모델로 교체 예정

예선 자료(6쪽)의 **Air Fit**(MIT 라이선스, GitHub 공개 코드)을 이용해 **새 모델을 학습**하고,
지금의 Random Forest 대신 옷차림 분류에 쓸 계획입니다.
교체할 곳은 `model.py` 한 파일입니다.

- `make_model()` — 모델 생성 (지금은 Random Forest)
- `train()` — 학습
- `predict_level()` — 예측 → `(단계 0~7, 확률)` 형태로만 돌려주면 화면·서버는 수정할 필요 없음

Air Fit 모델의 입력이 다르면 `make_row()`에서 입력 값을 맞춰 주면 됩니다.
MIT 라이선스이므로 사용 시 README에 출처(저장소 주소·저작권 표시)를 남겨야 합니다.

## 시연 팁

- 주소 끝에 `#analyze`를 붙여 열면 (예: `http://localhost:5000/#analyze`) 바로 분석 화면부터 시작 (녹화용)
- '하교 시간 알림'은 브라우저 알림이라 탭이 열려 있어야 울립니다
- 로그인은 '사용자 정보 수집 안 함' 범위에 따라 넣지 않았습니다

## 캠퍼스 모델 데이터 새로 만들기

`data/train_data.csv`를 다시 만들거나 기간을 늘리려면 (Colab 노트북 3개):

1. 안양만안(377) AWS 시간자료 CSV를 기상자료개방포털에서 연도별로 받기 (기온·풍속·습도)
2. 안양대 500m 격자 자료 수집 (API허브 고해상도 격자 · 특정지점 다중요소, 위경도 37.3921, 126.9206)
3. 같은 시각끼리 합쳐 `tm, aws_ta, aws_hm, aws_ws, uni_ta, uni_hm, uni_ws, month, hour` 열로 저장
4. `python train_campus.py` → 채점 결과 확인 후 `python web_export.py`

옷차림 모델을 실제 날씨로 학습하려면: 기상청 AWS 자료 CSV로 `python train.py --aws 파일.csv`
