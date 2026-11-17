# 실험 로그 (experiment_log)

총 62건. 생성: 2026-11-16 00:39

## RQ2-B01-005930 [baseline] — ok
- **목적**: 사전 지정 주 분석: 규제가 없었을 때의 log RV 반사실 (CausalImpact 방식)
- **데이터**: proc.rv_daily rv5 005930, 2025-10-01~2026-10-30, 학습≤07-15 (190일)
- **모형**: BSTS local level + spike-and-slab (src/rq2/bsts.py)
- **설정**: `{"controls": ["don_mean", "don_pc2", "don_pc3", "kosdaq_lrv", "us_lvix", "us_abs_sox", "gap_day"], "niter": 4000, "burn": 1000, "expected_model_size": 3, "prior_level_sd": 0.01, "prior_level_df": 32, "seed": 20261115}`
- **진단**: inclusion={'don_mean': 1.0, 'don_pc2': 0.213, 'don_pc3': 0.738, 'kosdaq_lrv': 0.999, 'us_lvix': 0.234, 'us_abs_sox': 0.205, 'gap_day': 0.146}
- **결과**: 005930/transition: +0.209 [-0.096,+0.522] p=0.183; 005930/post_effective: +0.052 [-0.180,+0.281] p=0.627

## RQ2-B01-000660 [baseline] — ok
- **목적**: 사전 지정 주 분석: 규제가 없었을 때의 log RV 반사실 (CausalImpact 방식)
- **데이터**: proc.rv_daily rv5 000660, 2025-10-01~2026-10-30, 학습≤07-15 (190일)
- **모형**: BSTS local level + spike-and-slab (src/rq2/bsts.py)
- **설정**: `{"controls": ["don_mean", "don_pc2", "don_pc3", "kosdaq_lrv", "us_lvix", "us_abs_sox", "gap_day"], "niter": 4000, "burn": 1000, "expected_model_size": 3, "prior_level_sd": 0.01, "prior_level_df": 32, "seed": 20261115}`
- **진단**: inclusion={'don_mean': 0.981, 'don_pc2': 1.0, 'don_pc3': 0.116, 'kosdaq_lrv': 0.99, 'us_lvix': 0.124, 'us_abs_sox': 0.536, 'gap_day': 0.219}
- **결과**: 000660/transition: +0.331 [+0.019,+0.642] p=0.038; 000660/post_effective: -0.053 [-0.265,+0.153] p=0.635

## RQ2-B01-005935 [falsification] — ok
- **목적**: 음성통제: 상품이 없는 삼성전자우에 같은 분석 — 효과가 나오면 규제 귀속 불가
- **데이터**: proc.rv_daily rv5 005935, 2025-10-01~2026-10-30, 학습≤07-15 (190일)
- **모형**: BSTS local level + spike-and-slab (src/rq2/bsts.py)
- **설정**: `{"controls": ["don_mean", "don_pc2", "don_pc3", "kosdaq_lrv", "us_lvix", "us_abs_sox", "gap_day"], "niter": 4000, "burn": 1000, "expected_model_size": 3, "prior_level_sd": 0.01, "prior_level_df": 32, "seed": 20261115}`
- **진단**: inclusion={'don_mean': 1.0, 'don_pc2': 0.262, 'don_pc3': 0.19, 'kosdaq_lrv': 0.995, 'us_lvix': 0.081, 'us_abs_sox': 0.235, 'gap_day': 0.065}
- **결과**: 005935/transition: +0.184 [-0.132,+0.522] p=0.272; 005935/post_effective: +0.188 [-0.049,+0.422] p=0.117

## RQ2-B01-FAMILY [baseline] — ok
- **목적**: 주 가설 검정 family (2종목×2창) 다중검정 보정
- **데이터**: RQ2-B01
- **모형**: Holm-Bonferroni
- **설정**: `{"family": "005930/000660 × transition/post_effective"}`
- **결과**: 005930/transition: p=0.183→Holm 0.550; 005930/post_effective: p=0.627→Holm 1.000; 000660/transition: p=0.038→Holm 0.152; 000660/post_effective: p=0.635→Holm 1.000

## RQ2-E-WITHIN [extension] — ok
- **목적**: 같은 기업·같은 펀더멘털, 상품만 없는 삼성전자우를 통제로 써 기업·업종 충격 제거
- **데이터**: rv5 005930, 005935
- **모형**: 차이 시계열 ITS(HAC 5) / 005935 를 회귀변수로 둔 OLS·BSTS
- **설정**: `{"dummies": "listing, transition, post_effective (기준=상장 전)"}`
- **결과**: within_DID_HAC/listing: -0.076 [-0.200,+0.048] p=0.231; within_DID_HAC/transition: -0.055 [-0.202,+0.092] p=0.464; within_DID_HAC/post_effective: -0.052 [-0.227,+0.123] p=0.561; within_DID_HAC_vs_listing/transition: +0.021 [-0.119,+0.161] p=0.769; within_DID_HAC_vs_listing/post_effective: +0.024 [-0.155,+0.203] p=0.793; within_ols_ctrl005935/transition: +0.170 [-0.186,+0.458] p=0.316; within_ols_ctrl005935/post_effective: +0.019 [-0.145,+0.163] p=0.791; within_bsts_ctrl005935/transition: +0.091 [-0.162,+0.350] p=0.503; within_bsts_ctrl005935/post_effective: -0.147 [-0.347,+0.045] p=0.149
- **해석**: SK하이닉스에는 같은 기업 대조가 없다 (우선주 없음).

## RQ2-D01 [diagnostic] — ok
- **목적**: 사전구간 회귀 (y~C_main) 가정 점검 — BSTS·OLS·SARIMAX 회귀부 공통
- **데이터**: rv5, 학습≤07-15
- **모형**: OLS 잔차 검정 묶음
- **설정**: `LB(5,10), BG(5), BP, White, JB, VIF, RESET(3), Cook's D, Chow@05-27, sup-F(wild boot 199), CUSUM`
- **결과**: 005930: LB10 p=0.000, BP p=0.216, JB p=0.870, VIFmax=5.8, RESET p=0.259, Chow p=0.043, supF p=0.101@2026-02-11; 005930_ITSfull: LB10 p=0.000, BP p=0.205, JB p=0.778, VIFmax=6.4, RESET p=0.345, Chow p=0.275, supF p=0.131@2026-05-22; 000660: LB10 p=0.311, BP p=0.229, JB p=0.191, VIFmax=5.8, RESET p=0.041, Chow p=0.001, supF p=0.020@2026-05-22; 000660_ITSfull: LB10 p=0.009, BP p=0.120, JB p=0.164, VIFmax=6.4, RESET p=0.116, Chow p=0.046, supF p=0.000@2026-07-03; 005935: LB10 p=0.022, BP p=0.629, JB p=0.000, VIFmax=5.8, RESET p=0.935, Chow p=0.193, supF p=0.387@2026-04-22; 005935_ITSfull: LB10 p=0.000, BP p=0.465, JB p=0.000, VIFmax=6.4, RESET p=0.496, Chow p=0.842, supF p=0.818@2026-04-22

## RQ2-D02 [diagnostic] — ok
- **목적**: 정상성 (모형 차분 여부 결정)
- **데이터**: 사전구간
- **모형**: ADF(AIC) + KPSS(c)
- **설정**: ``
- **결과**: {"000660/ambiguous": 4, "000660/stationary": 2, "000660/unit_root": 2, "005930/ambiguous": 5, "005930/stationary": 2, "005930/unit_root": 1, "005935/ambiguous": 5, "005935/stationary": 2, "005935/unit_root": 1}

## RQ2-D03 [robustness] — ok
- **목적**: ITS 효과의 비관측 교란 민감도 (Oster δ, Cinelli-Hazlett RV)
- **데이터**: rv5 전 표본
- **모형**: ITS-OLS (HAC 5)
- **설정**: `{"Rmax": "1.3×R2_long"}`
- **결과**: 005930/transition: τ=+0.153 pHAC=0.155 δ=3.36 RV=0.061 RVα=0.000; 005930/post_effective: τ=+0.066 pHAC=0.600 δ=-0.14 RV=0.037 RVα=0.000; 000660/transition: τ=+0.246 pHAC=0.105 δ=11.07 RV=0.097 RVα=0.000; 000660/post_effective: τ=-0.024 pHAC=0.877 δ=0.05 RV=0.014 RVα=0.000; 005935/transition: τ=+0.128 pHAC=0.324 δ=2.46 RV=0.044 RVα=0.000; 005935/post_effective: τ=+0.118 pHAC=0.422 δ=-0.22 RV=0.057 RVα=0.000

## RQ2-E-OLS-005930 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005930, 학습≤07-15
- **모형**: ITS-OLS (사전 회귀 + HAC, 계수·잔차 block bootstrap)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005930/transition: +0.245 [-0.081,+0.607] p=0.150; 005930/post_effective: +0.132 [-0.081,+0.340] p=0.208
- **해석**: 1s

## RQ2-E-OLS-000660 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 000660, 학습≤07-15
- **모형**: ITS-OLS (사전 회귀 + HAC, 계수·잔차 block bootstrap)
- **설정**: `{"controls": "C_main"}`
- **결과**: 000660/transition: +0.364 [-0.023,+0.732] p=0.067; 000660/post_effective: +0.062 [-0.147,+0.257] p=0.534
- **해석**: 1s

## RQ2-E-OLS-005935 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005935, 학습≤07-15
- **모형**: ITS-OLS (사전 회귀 + HAC, 계수·잔차 block bootstrap)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005935/transition: +0.202 [-0.165,+0.547] p=0.290; 005935/post_effective: +0.240 [+0.028,+0.455] p=0.029
- **해석**: 1s

## RQ2-E-BSTS-005930 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005930, 학습≤07-15
- **모형**: BSTS (주 분석)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005930/transition: +0.209 [-0.096,+0.522] p=0.183; 005930/post_effective: +0.052 [-0.180,+0.281] p=0.627
- **해석**: 27s

## RQ2-E-BSTS-000660 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 000660, 학습≤07-15
- **모형**: BSTS (주 분석)
- **설정**: `{"controls": "C_main"}`
- **결과**: 000660/transition: +0.331 [+0.019,+0.642] p=0.038; 000660/post_effective: -0.053 [-0.265,+0.153] p=0.635
- **해석**: 27s

## RQ2-E-BSTS-005935 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005935, 학습≤07-15
- **모형**: BSTS (주 분석)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005935/transition: +0.184 [-0.132,+0.522] p=0.272; 005935/post_effective: +0.188 [-0.049,+0.422] p=0.117
- **해석**: 25s

## RQ2-E-UC-005930 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005930, 학습≤07-15
- **모형**: 상태공간 UC: local level + 회귀 (MLE)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005930/transition: +0.071 [-0.340,+0.500] p=0.750; 005930/post_effective: -0.156 [-0.837,+0.519] p=0.641
- **해석**: 1s

## RQ2-E-UC-000660 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 000660, 학습≤07-15
- **모형**: 상태공간 UC: local level + 회귀 (MLE)
- **설정**: `{"controls": "C_main"}`
- **결과**: 000660/transition: +0.049 [-0.318,+0.424] p=0.787; 000660/post_effective: -0.335 [-0.826,+0.153] p=0.176
- **해석**: 2s

## RQ2-E-UC-005935 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005935, 학습≤07-15
- **모형**: 상태공간 UC: local level + 회귀 (MLE)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005935/transition: +0.202 [-0.086,+0.497] p=0.171; 005935/post_effective: +0.240 [+0.105,+0.380] p=0.001
- **해석**: 1s

## RQ2-E-SARIMAX-005930 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005930, 학습≤07-15
- **모형**: SARIMAX: ARMA + exog (AIC 선택)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005930/transition: +0.274 [-0.124,+0.672] p=0.175; 005930/post_effective: +0.155 [-0.023,+0.318] p=0.082
- **해석**: 11s

## RQ2-E-SARIMAX-000660 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 000660, 학습≤07-15
- **모형**: SARIMAX: ARMA + exog (AIC 선택)
- **설정**: `{"controls": "C_main"}`
- **결과**: 000660/transition: +0.352 [+0.005,+0.702] p=0.045; 000660/post_effective: +0.090 [-0.066,+0.241] p=0.261
- **해석**: 13s

## RQ2-E-SARIMAX-005935 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005935, 학습≤07-15
- **모형**: SARIMAX: ARMA + exog (AIC 선택)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005935/transition: +0.232 [-0.133,+0.591] p=0.195; 005935/post_effective: +0.243 [+0.086,+0.386] p=0.001
- **해석**: 8s

## RQ2-E-HARX-005930 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005930, 학습≤07-15
- **모형**: HAR-RV-X (재귀 반사실 시차)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005930/transition: +0.128 [-0.175,+0.408] p=0.376; 005930/post_effective: +0.092 [-0.026,+0.215] p=0.116
- **해석**: 3s

## RQ2-E-HARX-000660 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 000660, 학습≤07-15
- **모형**: HAR-RV-X (재귀 반사실 시차)
- **설정**: `{"controls": "C_main"}`
- **결과**: 000660/transition: +0.042 [-0.223,+0.353] p=0.748; 000660/post_effective: -0.076 [-0.205,+0.065] p=0.300
- **해석**: 2s

## RQ2-E-HARX-005935 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005935, 학습≤07-15
- **모형**: HAR-RV-X (재귀 반사실 시차)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005935/transition: +0.212 [-0.110,+0.558] p=0.204; 005935/post_effective: +0.258 [+0.142,+0.376] p=0.000
- **해석**: 2s

## RQ2-E-RIDGE-005930 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005930, 학습≤07-15
- **모형**: Ridge (TimeSeriesSplit CV)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005930/transition: +0.278 [-0.113,+0.619] p=0.179; 005930/post_effective: +0.135 [-0.025,+0.273] p=0.092
- **해석**: 4s

## RQ2-E-RIDGE-000660 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 000660, 학습≤07-15
- **모형**: Ridge (TimeSeriesSplit CV)
- **설정**: `{"controls": "C_main"}`
- **결과**: 000660/transition: +0.375 [-0.173,+0.785] p=0.141; 000660/post_effective: +0.057 [-0.172,+0.233] p=0.551
- **해석**: 4s

## RQ2-E-RIDGE-005935 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005935, 학습≤07-15
- **모형**: Ridge (TimeSeriesSplit CV)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005935/transition: +0.237 [-0.312,+0.825] p=0.375; 005935/post_effective: +0.229 [+0.006,+0.444] p=0.042
- **해석**: 5s

## RQ2-E-ENET-005930 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005930, 학습≤07-15
- **모형**: ElasticNet (TimeSeriesSplit CV)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005930/transition: +0.305 [-0.053,+0.620] p=0.091; 005930/post_effective: +0.162 [+0.023,+0.293] p=0.025
- **해석**: 2s

## RQ2-E-ENET-000660 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 000660, 학습≤07-15
- **모형**: ElasticNet (TimeSeriesSplit CV)
- **설정**: `{"controls": "C_main"}`
- **결과**: 000660/transition: +0.366 [-0.205,+0.839] p=0.162; 000660/post_effective: +0.060 [-0.179,+0.253] p=0.555
- **해석**: 3s

## RQ2-E-ENET-005935 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005935, 학습≤07-15
- **모형**: ElasticNet (TimeSeriesSplit CV)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005935/transition: +0.244 [-0.143,+0.749] p=0.258; 005935/post_effective: +0.263 [+0.116,+0.434] p=0.000
- **해석**: 2s

## RQ2-E-LGBM-005930 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005930, 학습≤07-15
- **모형**: LightGBM (얕은 트리, OOF 잔차 bootstrap)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005930/transition: +0.414 [-0.056,+0.849] p=0.084; 005930/post_effective: +0.249 [+0.094,+0.405] p=0.001
- **해석**: 8s

## RQ2-E-LGBM-000660 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 000660, 학습≤07-15
- **모형**: LightGBM (얕은 트리, OOF 잔차 bootstrap)
- **설정**: `{"controls": "C_main"}`
- **결과**: 000660/transition: +0.395 [-0.157,+0.801] p=0.151; 000660/post_effective: +0.120 [-0.088,+0.311] p=0.263
- **해석**: 3s

## RQ2-E-LGBM-005935 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005935, 학습≤07-15
- **모형**: LightGBM (얕은 트리, OOF 잔차 bootstrap)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005935/transition: +0.283 [-0.224,+0.855] p=0.366; 005935/post_effective: +0.351 [+0.141,+0.591] p=0.000
- **해석**: 3s

## RQ2-E-XGB-005930 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005930, 학습≤07-15
- **모형**: XGBoost (얕은 트리, OOF 잔차 bootstrap)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005930/transition: +0.353 [-0.058,+0.686] p=0.085; 005930/post_effective: +0.188 [+0.059,+0.316] p=0.002
- **해석**: 4s

## RQ2-E-XGB-000660 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 000660, 학습≤07-15
- **모형**: XGBoost (얕은 트리, OOF 잔차 bootstrap)
- **설정**: `{"controls": "C_main"}`
- **결과**: 000660/transition: +0.339 [-0.240,+0.755] p=0.171; 000660/post_effective: +0.072 [-0.158,+0.258] p=0.487
- **해석**: 3s

## RQ2-E-XGB-005935 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005935, 학습≤07-15
- **모형**: XGBoost (얕은 트리, OOF 잔차 bootstrap)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005935/transition: +0.291 [-0.126,+0.838] p=0.225; 005935/post_effective: +0.234 [+0.072,+0.441] p=0.001
- **해석**: 4s

## RQ2-E-SCM-005930 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005930, 학습≤07-15
- **모형**: 합성통제 (절편 허용, 가중치≥0 합1, donor log RV)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005930/transition: +0.406 [+0.087,+0.703] p=0.010; 005930/post_effective: +0.079 [-0.046,+0.199] p=0.235
- **해석**: 2s

## RQ2-E-SCM-000660 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 000660, 학습≤07-15
- **모형**: 합성통제 (절편 허용, 가중치≥0 합1, donor log RV)
- **설정**: `{"controls": "C_main"}`
- **결과**: 000660/transition: +0.376 [+0.073,+0.632] p=0.014; 000660/post_effective: +0.025 [-0.112,+0.130] p=0.650
- **해석**: 2s

## RQ2-E-SCM-005935 [extension] — ok
- **목적**: 모형 의존성 확인 (같은 설계·같은 창)
- **데이터**: rv5 005935, 학습≤07-15
- **모형**: 합성통제 (절편 허용, 가중치≥0 합1, donor log RV)
- **설정**: `{"controls": "C_main"}`
- **결과**: 005935/transition: +0.443 [+0.048,+0.788] p=0.034; 005935/post_effective: +0.224 [+0.054,+0.368] p=0.008
- **해석**: 1s

## RQ2-P03 [placebo] — ok
- **목적**: 긴 사전구간(2016~2026.5) 가짜 개입일 분포로 실제 효과의 이례성 평가 (모형 오차의 경험적 귀무분포)
- **데이터**: F1 일별 OHLC → log Garman-Klass, donor 131종목 평균
- **모형**: OLS(학습 250일; don_mean, VIX, |SOX|, |KOSDAQ|) 예측오차 평균
- **설정**: `{"n_fake": 238, "step": 10, "windows": "10일 / 61일"}`
- **결과**: 005930/transition: +0.368, 경험 p=0.412, 순위 0.77; 005930/post_effective: +0.357, 경험 p=0.265, 순위 0.86; 000660/transition: +0.490, 경험 p=0.197, 순위 0.90; 000660/post_effective: +0.144, 경험 p=0.702, 순위 0.67; 005935/transition: +0.457, 경험 p=0.319, 순위 0.82; 005935/post_effective: +0.402, 경험 p=0.315, 순위 0.84; 005930-005935/transition: -0.089, 경험 p=0.702, 순위 0.35; 005930-005935/post_effective: -0.045, 경험 p=0.756, 순위 0.43

## RQ2-E-GARCH [extension] — ok
- **목적**: 일별 수익률 기반 대안 측정: 시장모형 고유수익률의 GJR-GARCH 다기간 예측 대비 실제 고유분산
- **데이터**: F1 ret_log 2016~, 시장요인=donor 평균수익률
- **모형**: GJR-GARCH(1,1)-t, 시뮬레이션 예측
- **설정**: `{"train_end": "2026-07-15", "nsim": 1000}`
- **결과**: 005930/transition: +0.210 [-1.108,+1.462] p=0.738; 005930/post_effective: -0.079 [-1.031,+0.744] p=0.838; 000660/transition: +0.336 [-0.634,+1.507] p=0.558; 000660/post_effective: -0.106 [-0.862,+0.544] p=0.732; 005935/transition: +0.356 [-0.860,+1.581] p=0.594; 005935/post_effective: +0.124 [-0.959,+0.995] p=0.796
- **해석**: GARCH 다기간 예측은 비조건 분산으로 회귀 → 사전 고변동기의 영향이 크다. 공통충격은 고유수익률로만 제거.

## RQ2-P02 [placebo] — ok
- **목적**: placebo-in-space: 규제 없는 donor 종목에 같은 개입일 → 처치 종목 효과가 이례적인가
- **데이터**: rv5, main donor 175종목
- **모형**: OLS(C_main, 자기 제외 donor 평균) / SCM RMSPE 비
- **설정**: `순열 p = (#{|donor| ≥ |treated|}+1)/(N+1)`
- **결과**: 005930/transition/ols_transition_std: +0.578, p=0.205; 005930/transition/scm_transition_ratio: +1.342, p=0.091; 005930/transition/scm_transition: +0.406, p=0.074; 005930/post_effective/ols_post_effective_std: +0.312, p=0.608; 005930/post_effective/scm_post_effective_ratio: +1.168, p=0.394; 005930/post_effective/scm_post_effective: +0.079, p=0.737; 000660/transition/ols_transition_std: +0.839, p=0.057; 000660/transition/scm_transition_ratio: +1.236, p=0.109; 000660/transition/scm_transition: +0.376, p=0.109; 000660/post_effective/ols_post_effective_std: +0.143, p=0.790; 000660/post_effective/scm_post_effective_ratio: +0.968, p=0.754; 000660/post_effective/scm_post_effective: +0.025, p=0.926; 005935/transition/ols_transition_std: +0.438, p=0.347; 005935/transition/scm_transition_ratio: +1.468, p=0.063; 005935/transition/scm_transition: +0.443, p=0.057; 005935/post_effective/ols_post_effective_std: +0.521, p=0.438; 005935/post_effective/scm_post_effective_ratio: +1.269, p=0.263; 005935/post_effective/scm_post_effective: +0.224, p=0.349

## RQ2-D04 [diagnostic] — ok
- **목적**: 사전 추세 (상대 변동성 격차의 추세) — 반사실 외삽의 위험 점검
- **데이터**: rv5 분봉 / F1 GK
- **모형**: OLS 추세 + HAC(10)
- **설정**: `격차 = 종목 log 변동성 − donor 평균`
- **결과**: 005930/rv5 2025-10~2026-07-15/전체: +0.099/월 (p=0.000); 005930/rv5 2025-10~2026-07-15/상장 전: +0.074/월 (p=0.001); 005930/GK 2024-01~2026-07-15/전체: +0.034/월 (p=0.000); 005930/GK 2024-01~2026-07-15/상장 전: +0.028/월 (p=0.000); 000660/rv5 2025-10~2026-07-15/전체: -0.006/월 (p=0.824); 000660/rv5 2025-10~2026-07-15/상장 전: -0.077/월 (p=0.000); 000660/GK 2024-01~2026-07-15/전체: +0.012/월 (p=0.023); 000660/GK 2024-01~2026-07-15/상장 전: +0.008/월 (p=0.142); 005935/rv5 2025-10~2026-07-15/전체: +0.094/월 (p=0.000); 005935/rv5 2025-10~2026-07-15/상장 전: +0.055/월 (p=0.072); 005935/GK 2024-01~2026-07-15/전체: +0.049/월 (p=0.000); 005935/GK 2024-01~2026-07-15/상장 전: +0.044/월 (p=0.000); 005930-005935/rv5 2025-10~2026-07-15/전체: +0.005/월 (p=0.689); 005930-005935/rv5 2025-10~2026-07-15/상장 전: +0.020/월 (p=0.337); 005930-005935/GK 2024-01~2026-07-15/전체: -0.015/월 (p=0.000); 005930-005935/GK 2024-01~2026-07-15/상장 전: -0.016/월 (p=0.000)

## RQ2-E-FM-chronos2_cov [extension] — ok
- **목적**: 시계열 foundation model 반사실 (예측력 ≠ 인과식별; 공변량 유무 대조)
- **데이터**: rv5 005930/000660/005935, 절단일 2026-07-15 + 상장 전 가짜 절단일 17개
- **모형**: Chronos-2 + 공변량
- **설정**: `{"covariates": ["don_mean", "kosdaq_lrv", "us_lvix", "us_abs_sox", "gap_day"], "venv": ".venv-fm", "inference": "placebo 오차 분포 보정"}`
- **진단**: placebo h30 평균오차 +0.044, sd 0.248, RMSE 0.538
- **결과**: 000660/transition: +0.197 [-0.282,+0.675] 경험p=0.538; 000660/post_effective: -0.307 [-0.654,+0.039] 경험p=0.154; 005930/transition: +0.258 [-0.435,+0.952] 경험p=0.462; 005930/post_effective: +0.110 [-0.339,+0.559] 경험p=0.923; 005935/transition: +0.380 [-0.296,+1.057] 경험p=0.385; 005935/post_effective: +0.201 [-0.310,+0.711] 경험p=0.538

## RQ2-E-FM-chronos2_uni [extension] — ok
- **목적**: 시계열 foundation model 반사실 (예측력 ≠ 인과식별; 공변량 유무 대조)
- **데이터**: rv5 005930/000660/005935, 절단일 2026-07-15 + 상장 전 가짜 절단일 17개
- **모형**: Chronos-2 (amazon/chronos-2) 단변량
- **설정**: `{"covariates": ["don_mean", "kosdaq_lrv", "us_lvix", "us_abs_sox", "gap_day"], "venv": ".venv-fm", "inference": "placebo 오차 분포 보정"}`
- **진단**: placebo h30 평균오차 +0.192, sd 0.494, RMSE 0.897
- **결과**: 000660/transition: -0.006 [-0.977,+0.965] 경험p=1.000; 000660/post_effective: -0.937 [-1.577,-0.296] 경험p=0.077; 005930/transition: +0.094 [-0.922,+1.110] 경험p=0.846; 005930/post_effective: -1.021 [-1.837,-0.205] 경험p=0.077; 005935/transition: +0.304 [-1.098,+1.705] 경험p=0.769; 005935/post_effective: -0.640 [-2.000,+0.720] 경험p=0.615

## RQ2-E-FM-timesfm_uni [extension] — ok
- **목적**: 시계열 foundation model 반사실 (예측력 ≠ 인과식별; 공변량 유무 대조)
- **데이터**: rv5 005930/000660/005935, 절단일 2026-07-15 + 상장 전 가짜 절단일 17개
- **모형**: TimesFM 2.5 200M 단변량
- **설정**: `{"covariates": ["don_mean", "kosdaq_lrv", "us_lvix", "us_abs_sox", "gap_day"], "venv": ".venv-fm", "inference": "placebo 오차 분포 보정"}`
- **진단**: placebo h30 평균오차 +0.269, sd 0.423, RMSE 0.865
- **결과**: 000660/transition: +0.079 [-0.729,+0.888] 경험p=0.923; 000660/post_effective: -0.862 [-1.345,-0.378] 경험p=0.077; 005930/transition: +0.129 [-0.839,+1.098] 경험p=0.923; 005930/post_effective: -0.878 [-1.500,-0.256] 경험p=0.077; 005935/transition: +0.352 [-1.003,+1.708] 경험p=0.615; 005935/post_effective: -0.493 [-1.661,+0.675] 경험p=0.692

## RQ2-E-FM-timesfm_xreg [extension] — ok
- **목적**: 시계열 foundation model 반사실 (예측력 ≠ 인과식별; 공변량 유무 대조)
- **데이터**: rv5 005930/000660/005935, 절단일 2026-07-15 + 상장 전 가짜 절단일 17개
- **모형**: TimesFM 2.5 xreg+timesfm (OLS 잔차 예측)
- **설정**: `{"covariates": ["don_mean", "kosdaq_lrv", "us_lvix", "us_abs_sox", "gap_day"], "venv": ".venv-fm", "inference": "placebo 오차 분포 보정"}`
- **진단**: placebo h30 평균오차 +0.020, sd 0.201, RMSE 0.491
- **결과**: 000660/transition: +0.113 [-0.324,+0.550] 경험p=0.692; 000660/post_effective: -0.303 [-0.524,-0.082] 경험p=0.154; 005930/transition: +0.169 [-0.376,+0.714] 경험p=0.769; 005930/post_effective: +0.111 [-0.237,+0.458] 경험p=0.692; 005935/transition: +0.186 [-0.334,+0.706] 경험p=0.385; 005935/post_effective: +0.230 [-0.056,+0.517] 경험p=0.231

## RQ2-P01 [placebo] — ok
- **목적**: 가짜 개입일(상장 전)에서 모형별 위양성률·예측 편향 → 구간 보정 확인
- **데이터**: rv5, 가짜 개입일 12개 (2026-01-15~2026-04-10, 5일 간격), 사후 10·30일
- **모형**: ols,bsts,uc,harx,ridge,lgbm,scm,sarimax
- **설정**: `{"train": "표본시작~가짜일 전날"}`
- **결과**: bsts/h10: FPR=0.14, |eff|=0.191, RMSE=0.449; bsts/h30: FPR=0.19, |eff|=0.140, RMSE=0.475; harx/h10: FPR=0.47, |eff|=0.219, RMSE=0.503; harx/h30: FPR=0.64, |eff|=0.208, RMSE=0.564; lgbm/h10: FPR=0.19, |eff|=0.221, RMSE=0.565; lgbm/h30: FPR=0.44, |eff|=0.182, RMSE=0.617; ols/h10: FPR=0.17, |eff|=0.203, RMSE=0.456; ols/h30: FPR=0.17, |eff|=0.150, RMSE=0.477; ridge/h10: FPR=0.17, |eff|=0.190, RMSE=0.447; ridge/h30: FPR=0.25, |eff|=0.149, RMSE=0.479; sarimax/h10: FPR=0.22, |eff|=0.185, RMSE=0.448; sarimax/h30: FPR=0.25, |eff|=0.146, RMSE=0.479; scm/h10: FPR=0.33, |eff|=0.217, RMSE=0.461; scm/h30: FPR=0.78, |eff|=0.241, RMSE=0.523; uc/h10: FPR=0.17, |eff|=0.246, RMSE=0.493; uc/h30: FPR=0.11, |eff|=0.211, RMSE=0.517
- **해석**: FPR 이 0.05 를 크게 넘는 모형의 구간은 과소추정 → 그 모형의 p 값은 해석하지 않는다.

## RQ2-P04 [placebo] — ok
- **목적**: 주 모형(BSTS)의 placebo-in-space 순열 추론 — 모형 구간의 과소추정(placebo_time FPR)을 보정
- **데이터**: rv5, main donor 175종목 + 처치·음성통제
- **모형**: BSTS C_main (niter 1200, burn 400)
- **설정**: `perm p = (#{|donor| ≥ |treated|}+1)/(N+1); z = 효과/사후 sd`
- **진단**: donor 중 |z|>1.96 비율 = 모형 구간의 실제 위양성률: transition 0.12; post_effective 0.43
- **결과**: 005930/transition: +0.223, perm p=0.347; 005930/transition_z: +1.441, perm p=0.239; 005930/post_effective: +0.054, perm p=0.875; 005930/post_effective_z: +0.455, perm p=0.881; 000660/transition: +0.316, perm p=0.176; 000660/transition_z: +2.107, perm p=0.102; 000660/post_effective: -0.055, perm p=0.864; 000660/post_effective_z: -0.514, perm p=0.858; 005935/transition: +0.182, perm p=0.466; 005935/transition_z: +1.063, perm p=0.420; 005935/post_effective: +0.195, perm p=0.506; 005935/post_effective_z: +1.627, perm p=0.500

## RQ2-R-WITHIN-CALIB [robustness] — ok (A=failed: 보정 불가 판정)
- **목적**: 같은 기업 대조 결과(특히 BSTS post_eff_21_61d −0.25, 모형 p=0.023)를 다중검정·placebo 로 보정
- **데이터**: A) rv5 2025-10~2026-07-15 가짜 개입일; B) F1 GK 2016~2026-05 가짜 개입일
- **모형**: A) 주 분석과 같은 within OLS/BSTS(1500/500); B) OLS GK_005930 ~ GK_005935 (학습 250일)
- **설정**: `{"window_days": "10 + 61", "rv5_placebos": 20, "rv5_step": 3, "gk_step": 5, "family": "2 모형 × 4 창 = 8, Holm"}`
- **결과**: ols/transition: +0.170 p=0.316 (Holm 1.000), p_rv5=0.667 (Holm 1.000), GK +0.139 p_gk=0.658; ols/post_effective: +0.019 p=0.791 (Holm 1.000), p_rv5=0.048 (Holm 0.381), GK +0.077 p_gk=0.788; ols/post_eff_20d: +0.199 p=0.147 (Holm 1.000), p_rv5=0.952 (Holm 1.000), GK +0.205 p_gk=0.396; ols/post_eff_21_61d: -0.069 p=0.515 (Holm 1.000), p_rv5=0.048 (Holm 0.381), GK +0.014 p_gk=0.851; bsts/transition: +0.091 p=0.503 (Holm 1.000), p_rv5=1.000 (Holm 1.000), GK +0.139 p_gk=0.658; bsts/post_effective: -0.147 p=0.149 (Holm 1.000), p_rv5=0.048 (Holm 0.381), GK +0.077 p_gk=0.788; bsts/post_eff_20d: +0.059 p=0.605 (Holm 1.000), p_rv5=0.429 (Holm 1.000), GK +0.205 p_gk=0.396; bsts/post_eff_21_61d: -0.248 p=0.023 (Holm 0.181), p_rv5=0.048 (Holm 0.381), GK +0.014 p_gk=0.851
- **해석**: 핵심 결과 BSTS 21~61일: 모형 p 0.023 → Holm 0.181; GK 긴 표본 같은 설계 +0.014, p 0.851 → 유지되지 않음. A(rv5 placebo)는 쓸 수 없다: 가짜 창 20개가 71일씩 겹쳐 사실상 독립 표본이 2~3개이고, placebo 평균이 +0.26~+0.30 으로 치우쳤다. 원인은 005930~005935 회귀계수 β≈0.77~0.84(<1)인데 두 종목 수준이 2025-10~2026-07 에 약 2 로그단위 올라 (1−β)×수준변화 만큼 반사실이 낮게 예측된 것. 효과가 0 인 OLS 도 같은 최소 p(1/21)를 받는다 → p_rv5 는 효과가 아니라 수준 추세를 잰다. 결론: β=1 을 강제한 차분 설계(상장기간 대비 +0.02, p 0.79)가 같은 기업 비교의 주 사양이다.

## RQ2-R-BSTS-000660 [robustness] — ok
- **목적**: 사양 민감도: 통제·donor·측정치·사건일·학습창·donut·이상치(·BSTS 사전분포)
- **데이터**: rv5 등 000660
- **모형**: bsts
- **설정**: `{"n_specs": 24}`
- **강건성**: main: -0.052 (p=0.63); ctrl=C0: -0.305 (p=0.05); ctrl=C1: +0.045 (p=0.66); ctrl=C2: -0.108 (p=0.34); ctrl=C_us_only: -0.130 (p=0.36); ctrl=C_contaminated: -0.205 (p=0.03); pool=all: -0.056 (p=0.61); pool=strict: -0.066 (p=0.56); measure=rv5_close: -0.052 (p=0.62); measure=rv_pk: +0.097 (p=0.51); event=07-24: -0.063 (p=0.54); event=07-31: -0.073 (p=0.50); event=08-05: -0.122 (p=0.25); event=08-19: -0.167 (p=0.11); train=pre-listing only: +0.156 (p=0.23); train=2026 only: -0.107 (p=0.42); listing dummy: -0.116 (p=0.41); donut 07-28~08-04: -0.073 (p=0.52); winsorize 1/99: -0.049 (p=0.65); linear trend: -0.066 (p=0.59); bsts: level sd 0.1: -0.479 (p=0.27); bsts: model size 1: -0.063 (p=0.56); bsts: model size 6: -0.027 (p=0.82); bsts: no level (static reg): +0.008 (p=0.91)
- **결과**: post_effective 효과 범위 [-0.479, +0.156], 부호 반전 4건, p<.05 1/24

## RQ2-R-OLS-000660 [robustness] — ok
- **목적**: 사양 민감도: 통제·donor·측정치·사건일·학습창·donut·이상치(·BSTS 사전분포)
- **데이터**: rv5 등 000660
- **모형**: ols
- **설정**: `{"n_specs": 20}`
- **강건성**: main: +0.062 (p=0.53); ctrl=C0: +0.006 (p=0.89); ctrl=C1: +0.058 (p=0.68); ctrl=C2: -0.088 (p=0.50); ctrl=C_us_only: +0.096 (p=0.58); ctrl=C_contaminated: -0.233 (p=0.01); pool=all: +0.050 (p=0.60); pool=strict: +0.056 (p=0.57); measure=rv5_close: +0.055 (p=0.57); measure=rv_pk: +0.221 (p=0.24); event=07-24: +0.053 (p=0.59); event=07-31: +0.057 (p=0.57); event=08-05: +0.013 (p=0.89); event=08-19: -0.048 (p=0.67); train=pre-listing only: +0.193 (p=0.03); train=2026 only: +0.024 (p=0.87); listing dummy: -0.128 (p=0.41); donut 07-28~08-04: +0.040 (p=0.67); winsorize 1/99: +0.067 (p=0.48); linear trend: -0.226 (p=0.21)
- **결과**: post_effective 효과 범위 [-0.233, +0.221], 부호 반전 5건, p<.05 2/20

## RQ2-R-BSTS-005930 [robustness] — ok
- **목적**: 사양 민감도: 통제·donor·측정치·사건일·학습창·donut·이상치(·BSTS 사전분포)
- **데이터**: rv5 등 005930
- **모형**: bsts
- **설정**: `{"n_specs": 24}`
- **강건성**: main: +0.057 (p=0.58); ctrl=C0: -0.678 (p=0.00); ctrl=C1: +0.198 (p=0.08); ctrl=C2: +0.053 (p=0.64); ctrl=C_us_only: -0.446 (p=0.03); ctrl=C_contaminated: -0.094 (p=0.37); pool=all: +0.076 (p=0.51); pool=strict: +0.063 (p=0.57); measure=rv5_close: +0.065 (p=0.54); measure=rv_pk: +0.229 (p=0.15); event=07-24: +0.058 (p=0.60); event=07-31: +0.037 (p=0.72); event=08-05: -0.003 (p=0.99); event=08-19: -0.041 (p=0.73); train=pre-listing only: +0.234 (p=0.03); train=2026 only: +0.116 (p=0.34); listing dummy: +0.019 (p=0.86); donut 07-28~08-04: +0.043 (p=0.67); winsorize 1/99: +0.066 (p=0.59); linear trend: -0.390 (p=0.06); bsts: level sd 0.1: -0.234 (p=0.66); bsts: model size 1: +0.056 (p=0.59); bsts: model size 6: +0.054 (p=0.64); bsts: no level (static reg): +0.168 (p=0.03)
- **결과**: post_effective 효과 범위 [-0.678, +0.234], 부호 반전 7건, p<.05 4/24

## RQ2-R-OLS-005930 [robustness] — ok
- **목적**: 사양 민감도: 통제·donor·측정치·사건일·학습창·donut·이상치(·BSTS 사전분포)
- **데이터**: rv5 등 005930
- **모형**: ols
- **설정**: `{"n_specs": 20}`
- **강건성**: main: +0.132 (p=0.21); ctrl=C0: +0.256 (p=0.43); ctrl=C1: +0.326 (p=0.00); ctrl=C2: +0.205 (p=0.04); ctrl=C_us_only: +0.364 (p=0.21); ctrl=C_contaminated: -0.043 (p=0.53); pool=all: +0.121 (p=0.21); pool=strict: +0.132 (p=0.20); measure=rv5_close: +0.136 (p=0.19); measure=rv_pk: +0.195 (p=0.08); event=07-24: +0.135 (p=0.20); event=07-31: +0.136 (p=0.19); event=08-05: +0.107 (p=0.32); event=08-19: +0.071 (p=0.49); train=pre-listing only: +0.206 (p=0.05); train=2026 only: +0.144 (p=0.28); listing dummy: -0.005 (p=0.96); donut 07-28~08-04: +0.120 (p=0.25); winsorize 1/99: +0.140 (p=0.17); linear trend: -0.329 (p=0.04)
- **결과**: post_effective 효과 범위 [-0.329, +0.364], 부호 반전 3건, p<.05 3/20

## RQ2-R-BSTS-005935 [robustness] — ok
- **목적**: 사양 민감도: 통제·donor·측정치·사건일·학습창·donut·이상치(·BSTS 사전분포)
- **데이터**: rv5 등 005935
- **모형**: bsts
- **설정**: `{"n_specs": 24}`
- **강건성**: main: +0.198 (p=0.11); ctrl=C0: -0.525 (p=0.02); ctrl=C1: +0.329 (p=0.01); ctrl=C2: +0.186 (p=0.13); ctrl=C_us_only: -0.220 (p=0.36); ctrl=C_contaminated: +0.103 (p=0.38); pool=all: +0.212 (p=0.07); pool=strict: +0.194 (p=0.12); measure=rv5_close: +0.168 (p=0.17); measure=rv_pk: +0.205 (p=0.20); event=07-24: +0.206 (p=0.09); event=07-31: +0.172 (p=0.17); event=08-05: +0.186 (p=0.10); event=08-19: +0.278 (p=0.02); train=pre-listing only: +0.324 (p=0.02); train=2026 only: +0.202 (p=0.15); listing dummy: +0.169 (p=0.19); donut 07-28~08-04: +0.205 (p=0.09); winsorize 1/99: +0.190 (p=0.13); linear trend: +0.162 (p=0.27); bsts: level sd 0.1: +0.104 (p=0.89); bsts: model size 1: +0.188 (p=0.11); bsts: model size 6: +0.203 (p=0.09); bsts: no level (static reg): +0.243 (p=0.00)
- **결과**: post_effective 효과 범위 [-0.525, +0.329], 부호 반전 2건, p<.05 5/24

## RQ2-R-OLS-005935 [robustness] — ok
- **목적**: 사양 민감도: 통제·donor·측정치·사건일·학습창·donut·이상치(·BSTS 사전분포)
- **데이터**: rv5 등 005935
- **모형**: ols
- **설정**: `{"n_specs": 20}`
- **강건성**: main: +0.240 (p=0.03); ctrl=C0: +0.295 (p=0.40); ctrl=C1: +0.370 (p=0.00); ctrl=C2: +0.241 (p=0.01); ctrl=C_us_only: +0.434 (p=0.15); ctrl=C_contaminated: +0.061 (p=0.55); pool=all: +0.225 (p=0.03); pool=strict: +0.241 (p=0.04); measure=rv5_close: +0.220 (p=0.04); measure=rv_pk: +0.191 (p=0.16); event=07-24: +0.249 (p=0.02); event=07-31: +0.242 (p=0.02); event=08-05: +0.252 (p=0.01); event=08-19: +0.316 (p=0.00); train=pre-listing only: +0.334 (p=0.01); train=2026 only: +0.226 (p=0.10); listing dummy: +0.131 (p=0.34); donut 07-28~08-04: +0.250 (p=0.03); winsorize 1/99: +0.242 (p=0.03); linear trend: +0.028 (p=0.91)
- **결과**: post_effective 효과 범위 [+0.028, +0.434], 부호 반전 0건, p<.05 13/20

## RQ2-F01 [failure] — failed→fixed
- **목적**: UC(local level+exog) 반사실 예측
- **데이터**: 
- **모형**: 
- **설정**: `{"fix": "horizon 을 int() 로 변환 (models.cf_uc)"}`
- **진단**: 원인: get_forecast 에 np.int64 horizon → 'Prediction must have end after start' 오류로 3종목 모두 실패
- **결과**: 수정 후 재실행: RQ2-E-UC-* ok
- **해석**: 실패 기록 (결과에는 수정 후 재실행분만 사용)

## RQ2-F02 [failure] — failed→fixed
- **목적**: ridge/enet/LGBM/XGB 반사실 구간
- **데이터**: 
- **모형**: 
- **설정**: `{"fix": "stationary bootstrap 에서 잔차를 중심화 → 구간 폭만 잔차로부터 만든다"}`
- **진단**: 원인: OOF 잔차 평균이 0 이 아니어서 bootstrap 구간이 점추정에서 비켜남 (구간 중심 편향)
- **결과**: 수정 후 구간이 점추정을 중심으로 대칭
- **해석**: 실패 기록 (결과에는 수정 후 재실행분만 사용)

## RQ2-F03 [failure] — failed→fixed
- **목적**: placebo-in-time BSTS (가짜 개입일 12개)
- **데이터**: 
- **모형**: 
- **설정**: `{"fix": "bsts.py: 학습구간 상수열을 변수선택 대상에서 제외하고 Ω 대각 보정; test_constant_column 추가"}`
- **진단**: 원인: 짧은 학습구간에서 gap_day 가 상수열 → 표준화 시 0 나눗셈, Ω 특이 → LinAlgError, 12개 중 7개 실패
- **결과**: 재실행 36/36 적합 성공 (RQ2-P01)
- **해석**: 실패 기록 (결과에는 수정 후 재실행분만 사용)

## RQ2-F04 [failure] — failed→fixed
- **목적**: run_all placebo_time,placebo_space_bsts,robustness,figures 일괄 실행
- **데이터**: 
- **모형**: 
- **설정**: `{"fix": "프로세스 중단 → 적합마다 진행 출력(flush) 추가 후 단계별 재실행"}`
- **진단**: 원인: placebo_time 단계에서 2시간 이상 출력 없이 멈춤 (CPU 7분, '응답 없음'). 원인 미확인 (이전 정상 실행 1015s)
- **결과**: 재실행 placebo_time 215s, 오류 0; 결과가 중단 전 실행과 다르지 않음을 확인할 수 없어 재실행 결과만 사용
- **해석**: 실패 기록 (결과에는 수정 후 재실행분만 사용)

## RQ2-F05 [failure] — failed→fixed
- **목적**: 강건성 ctrl=C0 (통제변수 없음) BSTS
- **데이터**: 
- **모형**: 
- **설정**: `{"fix": "bsts.py reshape 수정 + test_no_regressors 추가"}`
- **진단**: 원인: 회귀변수 0개일 때 X_post reshape(-1,0) 오류로 3종목 모두 실패
- **결과**: 재실행 성공: C0 BSTS 005930 post −0.68, 005935 −0.53 (공통충격 미통제의 착시)
- **해석**: 실패 기록 (결과에는 수정 후 재실행분만 사용)

## RQ2-D05 [diagnostic] — ok
- **목적**: 사후에도 쓰는 국내 통제변수가 규제 영향을 받지 않았다는 가정 점검
- **데이터**: build_controls 2025-10~2026-10
- **모형**: 통제변수 ~ US 변수 + gap_day + 상장/전환/시행 더미, HAC 5
- **설정**: ``
- **결과**: don_mean/transition: -0.672 (사전 sd 의 -1.32배) p=0.000; don_mean/post_effective: -1.241 (사전 sd 의 -2.43배) p=0.000; don_pc2/transition: -0.299 (사전 sd 의 -0.10배) p=0.752; don_pc2/post_effective: -2.174 (사전 sd 의 -0.75배) p=0.031; don_pc3/transition: +0.569 (사전 sd 의 +0.22배) p=0.609; don_pc3/post_effective: +1.036 (사전 sd 의 +0.40배) p=0.355; kosdaq_lrv/transition: -1.547 (사전 sd 의 -1.34배) p=0.000; kosdaq_lrv/post_effective: -2.175 (사전 sd 의 -1.88배) p=0.000
- **해석**: 국내 통제변수의 이동이 크면 C_us_only 사양(국내 통제 제외)과 비교해야 한다. 다른 국내 충격과 구분 불가.

## RQ2-R-USONLY [robustness] — ok
- **목적**: 국내 통제변수 오염 가능성(D05) 대응: 미국 변수만 통제한 반사실의 긴 표본 보정 + donor 평균 자체의 이례성
- **데이터**: F1 GK 2016~2026-10
- **모형**: OLS(학습 250일; VIX, |SOX|) 예측오차 평균
- **설정**: `{"n_fake": 238, "step": 10}`
- **결과**: 005930/transition: +1.363, 경험 p=0.008, 순위 1.00; 005930/post_effective: +0.651, 경험 p=0.134, 순위 0.94; 000660/transition: +1.155, 경험 p=0.013, 순위 1.00; 000660/post_effective: +0.340, 경험 p=0.387, 순위 0.81; 005935/transition: +1.502, 경험 p=0.008, 순위 1.00; 005935/post_effective: +0.711, 경험 p=0.147, 순위 0.95; don_mean/transition: +0.684, 경험 p=0.021, 순위 0.98; don_mean/post_effective: +0.202, 경험 p=0.454, 순위 0.80
