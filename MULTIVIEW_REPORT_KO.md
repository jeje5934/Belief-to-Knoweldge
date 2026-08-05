# 다중 단말 상관 뷰 검증 보고서

## 한 줄 판정

**상관 side view는 유효했지만, 현재의 재학습 없는 결합으로는 구조적 판도를
뒤집지 못했다.** 강한 합성 상관 조건과 side view 무손실·무비용이라는 낙관적
가정에서, 초반에만 켠 약한 side pull은 budget-100 altproj의 BLER 0.1 knee를
약 **0.193 dB** 낮췄다. 기존 source 이득은 raw BP 대비 `0.609 -> 0.802 dB`로
커졌고 WebP-MAX는 앞섰지만, PixelCNN-MAX까지의 hard-CRC 격차는
`0.886 -> 0.693 dB`로 줄었을 뿐 역전하지 못했다.

핵심 그림:

- 게이팅: [`results/multiview_gate.png`](results/multiview_gate.png)
- 통합 waterfall/MSE: [`results/multiview_waterfall.png`](results/multiview_waterfall.png)
- 파생 수치: [`results/multiview_summary.json`](results/multiview_summary.json)

## 1. 공통 조건과 데이터 구성

| 항목 | 값 |
|---|---|
| 브랜치 | `codex/multiview` |
| 소스 | Fashion-MNIST test, 28x28 grayscale, 8 bit/pixel |
| 타깃 X | 원본 이미지, payload 6,272 bit |
| 채널 | BPSK/AWGN/perfect CSI |
| 채널 블록 | N=12,600, CRC24A 포함, payload rate=0.4978 |
| 디코더 | 5G LDPC + altproj, BP schedule `[5]x20`, 명목 budget 100 |
| side Y 기본 가정 | 수신단에 무손실로 이미 존재, 추가 채널 사용 0 |
| 정합 | 1차 oracle; 후속은 첫 5-iteration BP hard image 기반 receiver grid estimate |
| production 코드 | `decoder.py` 무수정; 실험 proxy만 사용 |

Y는 X에 독립 난수 평행이동·회전·밝기 변환을 적용했다. 첫 세 수준은 요청한
값이며, 마지막 두 수준은 side-alone 90% 초과 시 자동으로 문제를 어렵게 만들기
위한 fallback이다.

| 강도 | max shift | max rotation | brightness scale |
|---|---:|---:|---:|
| strong | 1 px | 3 deg | 0.95--1.05 |
| medium | 2 px | 7 deg | 0.90--1.10 |
| weak | 4 px | 15 deg | 0.80--1.20 |
| weaker fallback | 6 px | 25 deg | 0.65--1.35 |
| weakest fallback | 8 px | 35 deg | 0.50--1.50 |

## 2. 게이팅 계측

### 2.1 상관 강도 판정

256 block, oracle 정합 결과다. 어떤 강도도 side-alone 90%를 넘지 않아
“답을 그대로 주는” 자명한 문제가 아니었다. 반면 요청한 strong gate인
`결합 전체 >=85%`와 `b2--b5 +15%p`를 동시에 만족한 noise level은 없었다.
따라서 규약대로 **중간 신호, 축소 통합 진행**으로 판정했다.

| 강도 | side-alone | max genuine fusion | 판정 |
|---|---:|---:|---|
| strong | 80.72% | 86.39% | intermediate |
| medium | 80.50% | 86.20% | intermediate |
| weak | 80.29% | 85.86% | intermediate |
| weaker fallback | 79.66% | 85.07% | intermediate |
| weakest fallback | 79.40% | 84.52% | intermediate |

`max fusion`만 보면 85%를 넘는 수준이 있지만, 이는 cavity가 이미 91% 이상인
`sigma_actual=0.005`에서 cavity보다 나쁜 값이다. 즉 strong gate 실패를
“85% 달성”으로 바꾸어 읽으면 안 된다.

### 2.2 strong 상관의 전체/bit-plane 표

단위는 bit agreement %, bit-plane은 MSB부터 LSB다. `fusion`은 denoiser와
side 양쪽 가중치가 양수인 후보 중 전체 정합률이 가장 높은 조합이다.

| sigma | estimator | all | MSB | b6 | b5 | b4 | b3 | b2 | b1 | LSB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | cavity | 70.8 | 92.5 | 76.6 | 67.4 | 65.0 | 64.0 | 63.6 | 65.2 | 72.2 |
| 0.2 | denoiser | 74.9 | 96.9 | 90.2 | 78.1 | 67.6 | 63.7 | 60.2 | 69.1 | 73.4 |
| 0.2 | side | 80.7 | 97.1 | 91.0 | 83.2 | 78.9 | 75.3 | 73.1 | 73.3 | 73.9 |
| 0.2 | fusion | 79.3 | 97.4 | 91.7 | 82.8 | 77.4 | 72.1 | 66.3 | 72.5 | 73.9 |
| 0.1 | cavity | 74.3 | 96.6 | 88.2 | 73.3 | 66.6 | 64.7 | 64.4 | 67.5 | 73.1 |
| 0.1 | denoiser | 80.6 | 98.1 | 94.0 | 86.0 | 78.2 | 72.1 | 67.8 | 74.3 | 74.7 |
| 0.1 | side | 80.7 | 97.1 | 91.0 | 83.2 | 78.9 | 75.3 | 73.1 | 73.3 | 73.9 |
| 0.1 | fusion | 80.9 | 97.6 | 92.6 | 84.3 | 78.8 | 74.7 | 72.0 | 73.0 | 74.0 |
| 0.05 | cavity | 78.5 | 98.4 | 94.5 | 85.9 | 72.0 | 66.2 | 65.7 | 71.4 | 74.2 |
| 0.05 | denoiser | 83.7 | 98.9 | 96.3 | 91.0 | 83.0 | 76.8 | 73.7 | 75.3 | 75.0 |
| 0.05 | side | 80.7 | 97.1 | 91.0 | 83.2 | 78.9 | 75.3 | 73.1 | 73.3 | 73.9 |
| 0.05 | fusion | 82.9 | 98.7 | 95.9 | 90.3 | 81.9 | 75.0 | 72.3 | 74.7 | 74.6 |
| 0.02 | cavity | 84.4 | 99.4 | 97.9 | 94.4 | 88.4 | 75.5 | 69.1 | 75.3 | 75.1 |
| 0.02 | denoiser | 86.6 | 99.5 | 98.2 | 95.5 | 90.6 | 82.2 | 76.3 | 75.5 | 75.1 |
| 0.02 | side | 80.7 | 97.1 | 91.0 | 83.2 | 78.9 | 75.3 | 73.1 | 73.3 | 73.9 |
| 0.02 | fusion | 84.6 | 99.1 | 97.1 | 93.1 | 86.1 | 77.5 | 73.6 | 75.1 | 74.8 |
| 0.005 | cavity | 91.3 | 99.9 | 99.5 | 98.6 | 97.2 | 94.2 | 88.1 | 78.1 | 75.0 |
| 0.005 | denoiser | 91.5 | 99.9 | 99.5 | 98.7 | 97.2 | 94.4 | 88.4 | 78.7 | 75.2 |
| 0.005 | side | 80.7 | 97.1 | 91.0 | 83.2 | 78.9 | 75.3 | 73.1 | 73.3 | 73.9 |
| 0.005 | fusion | 86.4 | 99.3 | 97.7 | 94.7 | 89.3 | 81.7 | 77.3 | 76.2 | 74.8 |

핵심은 시간 의존성이다. side는 `sigma=0.2`에서 denoiser보다 전체 +5.8%p,
b2--b5 평균 +10.2%p 높지만, `sigma<=0.05`에서는 denoiser/cavity가 더 좋다.
따라서 side를 후반까지 상시 주입하면 정확한 cavity를 오염시킨다는 직접 예측이
나왔고, 통합에서는 scheduler sigma가 0.1 이상일 때만 큰 side step을 켰다.

## 3. altproj 통합

실험 proxy가 기존 source pull에 side pull을 더했다.

```text
C_next = rho*C + delta*D + delta_side*(L_side - P_current)
intrinsic = L_channel + C
```

CRC 위치·parity 위치에는 side term을 넣지 않고 payload 6,272 bit에만 넣었다.
`decoder.py`는 수정하지 않았으며 proxy가 반환하는 pull을 재스케일해 기존
`delta*D` 슬롯에서 위 식을 정확히 구현했다.

32/64-block 축소 탐색 후 hard-CRC canonical `rho=0.95`에서 확정한 값은:

- no-side anchor: `delta=.02`, `rho=.95`, geom sigma `0.3 -> 0.05`
- side best: `delta=.02`, `delta_side=.01`, `rho=.95`, side gate
  `scheduler sigma>=0.1`, geom sigma `0.3 -> 0.02`
- input-fusion 대안: `delta=.02`, side weight `.5`, end `.05`

동일 −3.05 dB/64 block에서 no-side 24 failures, explicit side term 9,
input-fusion 17이었다. 입력 자체를 섞는 방식보다 별도 code-free correction이
좋았다. `delta=.05` 계열은 성공 블록을 깨뜨렸고 축소했다.

### 3.1 canonical 512-block waterfall

| Es/N0 | no side failures/512, BLER [Wilson 95%] | with side failures/512, BLER [Wilson 95%] | no-side -> side paired | MSE no-side -> side | PSNR no-side -> side |
|---:|---|---|---|---:|---:|
| -3.20 | 259, .5059 [.4627,.5490] | 81, .1582 [.1292,.1923] | 178 rescued / 0 broken | .004583 -> .001971 | 23.39 -> 27.05 dB |
| -3.05 | 130, .2539 [.2181,.2933] | 41, .0801 [.0596,.1068] | 89 / 0 | .002824 -> .001327 | 25.49 -> 28.77 dB |
| -2.85 | 26, .0508 [.0349,.0734] | 4, .00781 [.00304,.01991] | 22 / 0 | .000657 -> .000155 | 31.82 -> 38.10 dB |
| -2.65 | 5, .00977 [.00418,.02265] | 3, .00586 [.00199,.01708] | 2 / 0 | .000241 -> .000155 | 36.19 -> 38.09 dB |

앞의 세 점은 CI가 분리되며, −2.65 dB 저-BLER 점은 CI가 겹친다. 따라서
저-BLER 꼬리의 작은 차이는 우위로 과장하지 않는다.

### 3.2 sigma endpoint confound 통제

side best와 동일한 `end=.02`를 no-side에도 적용해 같은 seed로 다시 계산했다.

| Es/N0 | no-side end=.02 failures | side end=.02 failures | paired rescued / broken |
|---:|---:|---:|---:|
| -3.20 | 341 | 81 | 261 / 1 |
| -3.05 | 159 | 41 | 119 / 1 |
| -2.85 | 20 | 4 | 16 / 0 |
| -2.65 | 4 | 3 | 1 / 0 |

따라서 side 곡선의 이득은 `end=.02` 재튜닝 artifact가 아니다. 실제로
`end=.02`는 낮은 SNR의 no-side를 악화했는데 side가 이를 크게 뒤집었다.

### 3.3 knee와 압축 baseline 격차

512-block 동일-seed 선형 보간이다.

| BLER | no side canonical | no side, matched end=.02 | with side | pure side gain |
|---:|---:|---:|---:|---:|
| 0.1 | -2.898 | -2.895 | -3.088 | **0.193 dB** |
| 0.01 | -2.651 | -2.664 | -2.856 | **0.192 dB** |

기존 대표본 hard-CRC no-side knee `-2.940 dB`에 동일-seed pure-side shift를
적용한 anchored estimate는 `-3.133 dB`다. 이것은 새 곡선의 직접 knee가 아니라
seed 분산을 줄이기 위한 **기존 anchor + paired shift**임을 명시한다.

| 시스템 | knee @ BLER .1 | side-best 대비 |
|---|---:|---:|
| raw + BP-100 | -2.331 | side가 0.802 dB 우세 |
| WebP-MAX + BP-100 | -2.801 | side가 0.332 dB 우세 |
| ours no-side | -2.940 | side가 0.193 dB 우세 |
| ours + side, anchored | **-3.133** | 0 |
| PixelCNN-MAX + BP-100 | -3.826 | side가 **0.693 dB 열세** |

결론적으로 기존 `0.886 dB` PixelCNN 격차의 약 0.193 dB만 메웠다. 카메라 간
비통신 조건에서 WebP/PixelCNN은 뷰 간 상관을 사용하지 못하지만, 현재의 단순
side pull은 그 구조적 기회를 전부 활용하지 못했다.

## 4. oracle 정합 완화

첫 5-iteration BP hard image와 Y 사이에서 translation/rotation grid와
least-squares brightness를 수신단이 추정했다. 동일 seed, −3.05 dB, 64 block:

| 정합 | failures/64 | BLER [Wilson 95%] |
|---|---:|---:|
| no side | 19 | .2969 [.1990,.4177] |
| oracle side | 6 | .0938 [.0437,.1898] |
| estimated side | 8 | .1250 [.0647,.2277] |

평균 절대 추정 오차는 `tx=0.148 px`, `ty=0.181 px`, `rotation=0.803 deg`,
`brightness=0.0818`이었다. 정합 이미지 MSE는 oracle `.00783`, estimated
`.01016`, 정합 전 side `.02359`였다. 64 block CI는 겹치므로 oracle 대비
2-block 손실의 크기를 확정적으로 주장하지 않지만, receiver 추정에서도 side
이득이 사라지지 않았다는 방향은 확인했다.

## 5. 현실적 2단 복호 가정

직접 2-view end-to-end 측정은 이번 우선순위에서 수행하지 않았다. 첫 전송 대상
Y는 회전·이동된 off-distribution 이미지이므로 “X의 no-side BLER와 같다”고 놓고
실측으로 포장하는 것보다 별도 실험으로 남기는 편이 정직하다.

대신 −3.05 dB 512-block 수치와 `첫 뷰/둘째 뷰 오류 독립`, `첫 뷰 no-side BLER가
X와 동일`이라는 낙관적 대칭 가정을 둔 회계만 기록했다.

- 두 뷰 모두 독립 no-side인 pair-BLER(any failure): `0.443`
- 첫 뷰 성공 시 둘째에 side 사용: pair-BLER `0.314`
- 첫 뷰 실패 시 둘째도 no-side fallback한 평균 per-view BLER: `0.189`

이는 **측정값이 아니라 민감도 계산**이다. 실제 수치는 (a) 변환된 첫 뷰의
off-distribution 복호성, (b) 두 링크 오류 상관, (c) 첫 뷰 실패 시 side confidence
검출에 의해 달라진다.

## 6. 자율 결정과 해석

1. side-alone 90% guard는 모든 강도에서 통과했지만 strong gate는 실패해,
   규약대로 full-scale가 아니라 축소 통합부터 시작했다.
2. 게이트가 보인 “초반 유효, 후반 유해”를 따라 큰 side step은
   `scheduler sigma>=.1`에서만 켰다. 상시 주입은 더 약한 `.002`만 견뎠다.
3. rho=.90 pilot 뒤 기존 hard-CRC canonical이 `.95`임을 확인해 최종 곡선을
   전부 `.95`로 다시 측정했다. pilot 수치는 최종 판정에 쓰지 않았다.
4. side best의 sigma endpoint가 달라 생기는 confound를 별도 512-block
   matched-endpoint 통제로 제거했다.
5. strong 상관 하나에서만 decoder 통합을 확장했다. medium/weak까지 full
   waterfall을 반복하는 것은 intermediate gate와 계산 우선순위에 맞지 않았다.
6. 직접 2단 복호는 첫 뷰 도메인 차이를 제대로 모델링하지 않은 채 단순 대칭
   가정으로 실행하면 오도하므로, 회계만 남기고 다음 단계로 분리했다.

## 7. 최종 권고

**현재 구현만으로도 “수신단만 바꿔 상관 뷰로 약 0.19 dB 추가 이득”이라는
positive 결과는 재현됐다.** 그러나 무손실·무비용 side라는 낙관 가정에서도
PixelCNN을 못 이기므로, 이 숫자만으로 다중 단말을 최종 주 시나리오로 확정하기엔
약하다.

다음 단계의 우선순위는 다음과 같다.

1. 재학습 없는 convex fusion을 더 튜닝하지 말고, `p(x|y_side)`를 직접 근사하는
   conditional denoiser 또는 calibrated virtual-channel LLR을 학습한다. 게이트에서
   side-alone이 denoiser보다 좋지만 fusion이 side를 넘지 못한 것이 근거다.
2. Fashion-MNIST 합성이 아닌 실제 multiview 데이터에서 registration uncertainty와
   side confidence를 함께 학습·보정한다.
3. 두 링크를 실제로 전송하는 2단 복호를 수행해 pair-BLER, 평균 채널사용,
   first-view 실패 fallback을 측정한다.
4. 이 세 단계 후에도 추가 이득이 약 0.2 dB라면, 다중 뷰는 보조 characterization으로
   남기고 주 논문 시나리오로는 확대하지 않는다.

## 재현 파일

- `multiview_gate.py`: 합성 뷰·oracle 정합·게이팅
- `multiview_altproj_study.py`: side term/input fusion/canonical waterfall/정합 추정
- `plot_multiview_results.py`: 그림·knee·paired 회계
- `results/multiview_gate_256.json`
- `results/multiview_altproj_rho095_confirm64.json`
- `results/multiview_altproj_canonical_waterfall512.json`
- `results/multiview_no_side_sigma_control512.json`
- `results/multiview_altproj_estimated_registration64.json`
- `results/multiview_altproj_oracle_registration64.json`
- `results/multiview_summary.json`

중간 32/64/256-block 스크리닝 JSON은 선택 경위 재현용이며 최종 성능 주장의
근거로 사용하지 않는다.
