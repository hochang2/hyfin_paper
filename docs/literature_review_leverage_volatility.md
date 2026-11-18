# 선행연구 정리: 레버리지 ETF는 기초자산 변동성을 높이는가

작성일: 2026-11-18
용도: 후속연구의 문헌 기반. 후속연구 설계와 실행 가능성 판단은 [followup_leverage_scale_volatility.md](followup_leverage_scale_volatility.md)에 있다.

## 1. 요약

- **논쟁의 핵심은 효과의 유무보다 효과의 크기다.** 양쪽 모두 레버리지 ETF의 일일 리밸런싱이 수익률과 같은 방향의 매매를 만든다는 점에는 동의한다. 갈리는 지점은 그 매매가 실제로 얼마나 체결되고, 시장이 얼마나 흡수하며, 경제적으로 의미 있는 크기인가이다.
- **증가 입장**은 이론적 리밸런싱 수요가 장 마감 무렵 변동성·가격 압력과 함께 움직인다는 증거를 제시한다 (Cheng & Madhavan 2009, Tuzun 2013, Shum et al. 2016).
- **제한적 효과 입장**은 자금 유출입이 리밸런싱 수요를 상당 부분 상쇄하고, 이를 반영하면 효과가 경제적으로 작다고 본다 (Ivanov & Lenkey 2018). Lenkey (2024)는 기존 연구 다수에 방법론 문제가 있다고 평가한다.
- **두 결론은 서로 다른 것을 측정한 결과일 수 있다.** 총수요와 순수요, 통계적 유의성과 경제적 크기, 마감 구간과 하루 전체, 지수와 개별 종목이 연구마다 다르다.
- **한국 단일종목 상품은 기존 문헌의 범위 밖에 있다.** 대부분의 근거는 미국 지수형 상품에서 나왔다. 두 종목에 집중된 한국 사례에 관해서는 동료심사를 거친 다종목 인과 연구를 찾지 못했다.

## 2. 논쟁의 출발점: 리밸런싱 메커니즘

배수 L을 매일 유지하는 상품은 기초자산 수익률이 r일 때 다음 크기의 익스포저를 조정해야 한다.

```
리밸런싱 금액 ≈ L(L − 1) × AUM × r
```

- 2배 상품은 L(L−1) = 2, −2배 상품은 6이다.
- 두 경우 모두 값이 양수이므로, 오른 날에는 사고 내린 날에는 판다. 롱과 인버스의 헤지가 서로 상쇄되지 않고 같은 방향으로 쌓인다.
- 이 구조는 옵션 매도 포지션의 감마 헤지와 비슷하다. 그래서 문헌은 이를 숏감마 수요나 포트폴리오 보험과 비교한다.

여기까지는 양쪽이 대체로 같다. 논쟁은 이 식의 값이 실제 시장 충격으로 이어지는 단계에서 시작된다.

## 3. 두 입장

### 3.1 변동성을 높인다는 입장

| 연구 | 표본 | 주장 | 핵심 근거 | 주의점 |
|---|---|---|---|---|
| Cheng & Madhavan (2009, JOIM) | 미국 레버리지·인버스 ETF | 일일 재조정이 장 마감 부근 변동성을 키울 수 있음 | 리밸런싱의 구조적 계산과 마감 압력 분석 | 초기 연구, 인과 식별보다 메커니즘 제시 |
| Tuzun (2013, FEDS 2013-48) | 미국 주식형 108개, 2006-06~2011-12, 약 201억 달러 | 레버리지 ETF가 포트폴리오 보험과 비슷한 수요를 만듦 | 지수 1% 변동 시 약 10.4억 달러 수요(일 거래량의 약 0.8%). 평균 대형주에서 가격 6.9bp, 변동성 22.7bp 추가. 소형주는 2011년 일 거래량의 최대 6%, 마감 1시간의 최대 18% | 수요가 **가상 계산치**이며, 저자도 시장을 교란한다고 입증된 것은 아니라고 적음 |
| Shum, Hejazi, Haryanto & Rodier (2016, Review of Finance) | 미국, 2006~2011 | 마감 변동성이 잠재 리밸런싱과 양의 관계 | 잠재 리밸런싱/총거래량과 마감 변동성의 유의한 양의 관계 | 모든 경우 경제적으로 큰 것은 아니며, 변동성이 큰 날에 효과가 가장 큼 |
| Zhao (2026, arXiv 2608.03703, 동료심사 전) | 한국 단일종목 레버리지 | 리밸런싱이 자기강화적으로 변동성을 높이고 전략적 거래의 대상이 됨 | 미국 뉴스에 대한 첫날 반응의 약 75%가 다음 날 종가까지 되돌림. SK하이닉스 연환산 변동성이 9주간 100.0%에서 136.7%로 상승했다고 귀속 | 초록만 확인. 변동성 지표와 식별 방법은 전문 확인 필요 |

### 3.2 효과가 제한적이라는 입장

| 연구 | 표본 | 주장 | 핵심 근거 | 주의점 |
|---|---|---|---|---|
| Ivanov & Lenkey (2018, J. Financial Markets; 2014 FEDS판) | 미국 주식형, 2006~2014 | 우려가 과장되었음 | 자금 유출입이 리밸런싱 수요를 크게 줄이며 시장 스트레스 시기에도 그러함. 이를 반영하면 장 후반 수익률·변동성 영향이 경제적으로 미미함 | 분산된 지수형 상품 중심. 집중도가 높은 단일종목에 그대로 옮길 수 없음 |
| Lenkey (2024, QFE, 서베이) | 기존 문헌 전반 | 전체적으로 우려가 뒷받침되지 않음 | 많은 논문이 유의한 연관을 보고하지만 다수에 심각할 수 있는 방법론 문제가 있고, 경제적 연관은 작아 보임 | 이 입장의 저자가 쓴 서베이이므로 독립 검증으로 보지 않음 |
| Trainor (2010, Technology and Investment) | 미국 | 변동성 증가 증거를 찾지 못함 | 세부 근거는 전문 확인 필요 | 초기 연구. 검색 목록 수준으로만 확인 |

### 3.3 쟁점별 비교

| 쟁점 | 증가 입장 | 제한적 효과 입장 | 우리 연구에 주는 시사점 |
|---|---|---|---|
| 수요 측정 | L(L−1)·AUM·r의 이론값 | 자금흐름을 반영한 순수요 | 총수요와 순수요를 모두 계산해야 함 |
| 판단 기준 | 통계적으로 유의한 연관 | 경제적 크기 | 유의성과 함께 변동성 기여분을 보고해야 함 |
| 관측 구간 | 장 마감 구간 | 장 후반 수익률·변동성, 경제적 크기 | 하루 RV만 보면 마감 효과가 희석될 수 있음 |
| 기초자산 | 지수와 소형주·금융주 등 민감 집단 | 시장 전체 관점 | 집중도가 핵심 변수 |
| 시장 상태 | 고변동 구간에서 효과가 큼 | 스트레스 시기에도 자금흐름이 상쇄 | 변동성 국면별로 나눠야 함 |
| 인과 해석 | 동시 관계 위주 | 동시 관계의 교란을 지적 | 변동성이 크면 수익률과 수요가 함께 커지는 역인과를 통제해야 함 |

가장 중요한 방법론 문제는 마지막 줄이다. 리밸런싱 수요는 정의상 |r|에 비례한다. 변동성이 큰 날에는 |r|도 크므로, 수요와 변동성의 상관은 인과가 없어도 나타난다. 단순 회귀의 양의 계수만으로는 증가 입장을 입증하기 어렵다.

## 4. 판단에 필요한 인접 문헌

### 4.1 헤지 수요가 가격에 주는 영향

- **Baltussen, Da, Lammers & Martens (2021, JFE):** 60개 이상의 선물 시장, 1974~2020년에서 일중 모멘텀을 찾고, 옵션과 레버리지 ETF의 숏감마 헤지를 원인 후보로 제시한다.
- **Ni, Pearson, Poteshman & White (2021, RFS):** 옵션 마켓메이커의 헤지 리밸런싱이 주식 수익률 변동성과 큰 가격 변동 확률에 영향을 준다는 증거를 제시한다. 레버리지 ETF 연구는 아니지만 감마 헤지 경로를 뒷받침한다.
- **Ben-David, Franzoni & Moussawi (2018, JF):** ETF 보유 비중이 높은 종목일수록 변동성이 높다. S&P 500 종목에서 1표준편차 증가가 변동성 표준편차의 9~15%에 해당한다. 일반 ETF의 자금흐름·차익거래 경로이므로 레버리지 효과로 읽으면 안 된다.

### 4.2 매매 규모를 가격 충격으로 바꾸는 모형

| 모형 | 형태 | 의미 | 사용 시 주의 |
|---|---|---|---|
| 제곱근 법칙 (Tóth et al. 2011) | Δ = Yσ√(Q/V) | 충격은 일 변동성에 비례하고 거래 비중의 제곱근에 비례 | Y 값은 시장·표본별 추정 필요. 원래 개별 메타오더 모형이라 모두가 아는 예측 가능한 수요에 적용하면 과대추정일 수 있음 |
| 선형 충격 (Tuzun 2013의 계산 방식 등) | Δ ∝ Q/V | 거래 비중에 비례 | 큰 주문에서 과대, 작은 주문에서 과소 추정 가능 |
| 비탄력 시장 (Gabaix & Koijen) | Δ = M × Q/시가총액, M 약 5 | 지속적 자금 유입의 총량 효과 | 하루 안에 되돌려지는 헤지 흐름이 아니라 지속적 수요에 관한 모형 |

세 모형은 같은 매매에 대해 크게 다른 답을 낸다. 따라서 "어느 규모면 움직이는가"의 답은 모형 선택에 크게 좌우되며, 단일 숫자로 제시하면 안 된다.

### 4.3 단일종목 레버리지 상품

- **Bessembinder (SSRN 5369417 등):** 단일종목 레버리지 상품의 수익률 열위를 분석한다. 투자자 성과에 관한 연구이며 기초자산 변동성 연구가 아니다.
- **AMRO 블로그 (Yoon, 2026-07-29):** 한국 단일종목 상품의 거래대금이 정점일에 KOSPI 거래대금의 약 40%였지만, 자산은 두 기업 시가총액의 1% 미만이라고 정리한다. 증거는 혼재되어 있고 집중도가 중요하다고 본다. 연구 논문이 아닌 정책기관 블로그다.

## 5. 우리 프로젝트 결과의 위치

현재까지의 결과는 어느 한쪽을 그대로 지지하지 않는다.

| 항목 | 결과 | 어느 입장과 가까운가 |
|---|---|---|
| RQ1 상장 효과 | 삼성전자 +0.574, SK하이닉스 +0.325 (log RV), 치환 p 0.057, 0.233. MDE 0.78 | 판단 불가. 값은 양수지만 탐지 한계 아래이고 음성통제도 함께 상승 |
| RQ1 같은 기업 차분 | −0.07 (rv5), +0.06 (GK), 유의하지 않음 | 제한적 효과와 양립 (MDE 0.47) |
| RQ1 마감−장중 ln RV (사후 분석) | +0.27, +0.32, Holm 조정 p 0.20 | 방향은 증가 입장과 같지만 유의하지 않음 |
| RQ2 규제 효과 | BSTS +5.3%, −5.2%, 유의한 감소 없음. −20% 탐지력 11~31% | 판단 불가. 저검정력 |
| RQ3 예측력 | 압력 변수 추가 효과 없음 (OOS −0.7%, −7.3%) | 제한적 효과와 양립하나, 절대수익률과의 상관이 0.93~0.94로 분리가 어려움 |
| RQ4 거래 감소 | 규제 상품 거래대금 −94% | 거래대금은 리밸런싱 수요가 아니므로 직접 근거가 아님 |

후속연구용 사전 계산에서는 다음이 추가로 확인됐다 (정의와 한계는 후속연구 문서 3절).

- 상장 후 106거래일 평균 이론 리밸런싱 수요는 일 거래대금의 **SK하이닉스 5.5%, 삼성전자 3.8%** 다.
- 기초자산 1% 변동당으로는 일 거래대금의 약 **1.2%, 1.0%** 다. Tuzun의 미국 지수 기준 약 0.8%와 자릿수가 같다. 금액 비중이 더 큰 주된 이유는 두 종목의 일별 변동폭이 컸기 때문이다.
- 자금흐름을 반영해도 평균 수요 크기는 줄지 않았다. 다만 자금흐름의 체결 시점을 확인하지 못했으므로, Ivanov & Lenkey의 상쇄 경로가 없다고 단정할 수는 없다.

## 6. 문헌 공백과 우리 연구의 기여 가능성

1. **집중도:** 기존 연구는 주로 분산된 지수형 상품을 다룬다. 시가총액 상위 두 종목에 몰린 상품의 효과는 거의 다뤄지지 않았다.
2. **규제 충격:** 상품 거래가 급감했지만 규모 비율은 크게 줄지 않은 사례는 거래 활동과 헤지 수요를 분리해 볼 기회다.
3. **필요 규모:** 많은 연구가 효과의 유무를 묻는다. 어느 규모부터 탐지 가능하거나 경제적으로 의미 있는지를 명시적으로 계산한 연구는 이번 조사에서 찾지 못했다.
4. **변동성 정의:** 같은 충격이 5분 RV, 마감 구간 RV, 종가 대 종가 변동성에 다르게 나타날 수 있다. 입장 차이의 일부가 여기서 나올 수 있다.

## 7. 근거 수준과 한계

이번 조사는 웹에서 접근 가능한 자료로 수행했다. 논문에 인용하기 전에 아래 표에서 전문 미확인 항목을 원문으로 확인해야 한다.

| 수준 | 해당 자료 |
|---|---|
| 본문 일부를 직접 확인 | Tuzun (2013) |
| 초록·공식 요약 확인 | Ivanov & Lenkey (FEDS판), Lenkey (2024), Shum et al. (2016), Baltussen et al. (2021), Ni et al. (2021), Tóth et al. (2011), Zhao (2026), AMRO 블로그 |
| 검색 요약·2차 인용 수준 | Cheng & Madhavan (2009), Trainor (2010), Ben-David et al. (2018), Gabaix & Koijen, Bessembinder |

- Tóth et al.의 계수 Y 값은 확인하지 못했다. 후속연구 문서의 Y는 가정한 시나리오 값이다.
- Gabaix & Koijen의 M 약 5는 원문이 아닌 요약을 통해 확인했다.
- 언론에 인용된 투자은행의 한국 상품 변동성 기여 추정치는 원자료를 확인할 수 없어 이 문서에서 제외했다.
- 한국은행 경고, 금융당국 발언 같은 언론 보도는 정책 맥락일 뿐 효과의 증거로 쓰지 않았다.

## 참고문헌

- Baltussen, G., Da, Z., Lammers, S., & Martens, M. (2021). Hedging demand and market intraday momentum. *Journal of Financial Economics*, 142(1), 377–403. [링크](https://ideas.repec.org/a/eee/jfinec/v142y2021i1p377-403.html)
- Ben-David, I., Franzoni, F., & Moussawi, R. (2018). Do ETFs increase volatility? *Journal of Finance*, 73(6), 2471–2535.
- Bessembinder, H. Single-stock leveraged ETF 수익률 연구. [SSRN 5369417](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5369417), [SSRN 7383378](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7383378) (제목과 판본 확인 필요)
- Cheng, M., & Madhavan, A. (2009). The dynamics of leveraged and inverse exchange-traded funds. *Journal of Investment Management*. [PDF](https://joim.com/wp-content/uploads/emember/downloads/p0283.pdf)
- Gabaix, X., & Koijen, R. In search of the origins of financial fluctuations: The inelastic markets hypothesis. NBER Working Paper 28967.
- Ivanov, I. T., & Lenkey, S. L. (2018). Do leveraged ETFs really amplify late-day returns and volatility? *Journal of Financial Markets*, 41, 36–56. 초기판: [Are Concerns About Leveraged ETFs Overblown?](https://www.federalreserve.gov/econres/feds/are-concerns-about-leveraged-etfs-overblown.htm) (학술지판 제목은 인용 전 확인 필요)
- Lenkey, S. L. (2024). The market impact of leveraged ETFs: A survey of the literature. *Quantitative Finance and Economics*, 8(4), 815–840. [doi:10.3934/QFE.2024031](https://www.aimspress.com/article/doi/10.3934/QFE.2024031)
- Ni, S. X., Pearson, N. D., Poteshman, A. M., & White, J. (2021). Does option trading have a pervasive impact on underlying stock prices? *Review of Financial Studies*, 34(4), 1952–1986. [링크](https://ideas.repec.org/a/oup/rfinst/v34y2021i4p1952-1986..html)
- Shum, P., Hejazi, W., Haryanto, E., & Rodier, A. (2016). Intraday share price volatility and leveraged ETF rebalancing. *Review of Finance*, 20(6), 2379–2409. [링크](https://ideas.repec.org/a/oup/revfin/v20y2016i6p2379-2409..html)
- Tóth, B., et al. (2011). Anomalous price impact and the critical nature of liquidity in financial markets. *Physical Review X*, 1, 021006. [doi](https://doi.org/10.1103/PhysRevX.1.021006)
- Trainor, W. J. (2010). Do leveraged ETFs increase volatility? *Technology and Investment*. [링크](https://file.scirp.org/Html/8-9900042_2498.htm)
- Tuzun, T. (2013). Are leveraged and inverse ETFs the new portfolio insurers? FEDS 2013-48. [링크](https://www.federalreserve.gov/pubs/feds/2013/201348/index.html)
- Yoon, W. (2026-07-29). Do leveraged ETFs make markets more volatile? AMRO. [링크](https://www.amro-asia.org/do-leveraged-etfs-make-markets-more-volatile)
- Zhao (2026). Preying on Leveraged ETFs. [arXiv:2608.03703](https://arxiv.org/abs/2608.03703)
