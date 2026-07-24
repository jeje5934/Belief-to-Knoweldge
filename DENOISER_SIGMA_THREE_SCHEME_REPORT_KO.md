# σ 교정 후 legacy / EP / altproj 재평가

## 1. 최종 판정

Legacy `β_legacy=0`, EP `β_ep=1`, 공통 `sigma_post=3.0`,
geometric `σ: 0.3→0.02` 조건에서
**altproj `δ=0.02, ρ=0.9`가 이번 3-way의 명확한 승자**다.

- β 규약을 명시한 독립 재측정(seed 20260803)의 −2.7 dB에서
  legacy는 `51/1024`, altproj는 `10/1024` 실패였다. Wilson 95% CI가
  완전히 분리됐고, paired 회계는 **파괴 0 / 구제 41**이다.
- −2.5 dB에서는 legacy `3/1024`, altproj `1/1024`로 raw 우위지만
  CI는 겹친다.
- −2.4 dB에서는 세 스킴 모두 `0/1024`라 구분 불가능하다
  (각 Wilson 상한 0.003738).
- EP는 legacy를 따라잡지 못했다. −2.7 dB best EP의 CI가 legacy와
  분리되어 나빴고, −2.5 dB에서만 α=0.01이 통계적 동률 범위다.
- budget 100→50→30 축소는 세 스킴 모두 유지되지 않았다.
- damping 요구는 **부분적으로만 완화**됐다. `ρ=1.0`은 `δ=0.02`에서
  더 이상 붕괴하지 않고 −2.7 dB에서 legacy를 이겼지만, −2.5 dB에서는
  다시 나빠졌다. `δ=0.05, ρ=1.0`은 여전히 완전 붕괴했다.

따라서 σ 과대 신고가 altproj의 순수 누적 취약성 일부를 키웠다는
증거는 있으나, damping의 필요성 자체가 사라진 것은 아니다.

## 2. 공통 조건과 구현 감사

| 항목 | 값 |
|---|---|
| 브랜치 | `practical_sigma` |
| 채널 | BPSK / AWGN / perfect CSI |
| payload / 전송 길이 | 6272 / 12600 bit |
| CRC | CRC24A |
| 공통 schedule | budget 100: `[5]×20`, source call 19회 |
| legacy | α=0.1, `β_legacy=0` |
| EP | damped EP, α_ep sweep, `β_ep=ep_code_power=1.0` |
| altproj | δ/ρ sweep, warm-start, early-stop off |
| denoiser σ | geometric 0.3→0.02 |
| sigma_post | 3.0 고정 |
| 표본 | −2.7/−2.5 dB 각 1024 block, 유망 구성 −2.4 dB 1024 block |
| 비교 | 각 SNR에서 동일 payload와 동일 stateless AWGN을 모든 arm이 공유 |

### β 파라미터 규약

두 β는 이름만 비슷하고 의미가 반대다.

- **Legacy `β_legacy`**:
  `channel + β_legacy·bp_ext + α·src_ext`에서 이미 수행한 BP의
  extrinsic을 추가로 되먹이는 부스팅 계수다. `β_legacy=0`이 표준
  LDPC BP만큼만 수행하는 공정 기준선이며 모든 legacy arm에서 0이다.
- **EP `β_ep=ep_code_power`**:
  posterior를 구성하는 code-factor site의 damping이다. `β_ep=0`은
  코드 팩터를 제거해 EP 정식화를 무너뜨린다. 모든 EP arm은 정의상
  필요한 `β_ep=1`로 실행했다.

직전 3-way runner도 실제로 `ep_code_power=1.0`이었으므로 기존 EP
결과가 β_ep=0 때문에 무효였던 것은 아니다. 혼동을 제거하기 위해
runner에 두 값을 별도 상수·JSON 필드로 기록하고 EP arm 진입마다
`β_ep==1` assert를 추가했다. 이후 독립 seed 20260803으로 EP α 3종,
legacy, altproj를 다시 측정했으며 아래 EP 표와 최종 3-way 판정은
이 재측정 결과를 authoritative 값으로 사용한다.

Legacy와 EP는 production decoder의 `sigma_scheduler`를 직접 사용했다.
Altproj production 경로에는 per-chunk σ 훅이 없고 scalar
`altproj_sigma_den`만 있다. `decoder.py`를 수정하지 않기 위해 harness가
이미 로드된 `SoftDenoiser` 호출의 σ 인자만 19점 경로로 치환했다.
각 altproj decode 후 **정확히 19회 적용됐는지 assert**했으며 모든
실행에서 통과했다. C 누적, BP, δ/ρ, clipping 코드는 그대로다.

## 3. budget-100 전체 격자

각 셀은 `BLER [Wilson 95% CI]; legacy 대비 파괴/구제`다.

### EP α_ep

| 구성 | −2.7 dB | −2.5 dB |
|---|---|---|
| legacy α=0.1, β_legacy=0 | 0.04980 [0.03808, 0.06489]; 기준 | 0.002930 [0.000997, 0.008578]; 기준 |
| EP α_ep=0.01, β_ep=1 | 0.14551 [0.12523, 0.16843]; 106/8 | **0.002930 [0.000997, 0.008578]; 2/2** |
| EP α_ep=0.02, β_ep=1 | **0.13379 [0.11430, 0.15601]; 96/10** | 0.01172 [0.00672, 0.02037]; 10/1 |
| EP α_ep=0.05, β_ep=1 | 1.00000 [0.99626, 1.00000]; 973/0 | 1.00000 [0.99626, 1.00000]; 1021/0 |

−2.7 dB에서는 α=0.02가 raw best지만 α=0.01과 CI가 겹친다.
−2.5 dB에서는 α=0.01만 legacy와 동일한 3/1024를 기록했고 α=0.02는
성공 블록 10개를 깨고 1개만 구제했다. 두 SNR의 강건성을 우선해
EP 대표값은 α=0.01로 유지한다. α=0.05는 β_ep=1을 명시한
재측정에서도 두 SNR 모두 완전 붕괴했다. −2.5 dB에서는 8/8 batch
모두 site-change growth 경고가 발생했다.

### Altproj δ/ρ

Altproj 전체 격자는 seed 20260802 결과다. β 규약은 altproj에 적용되는
파라미터가 아니므로 이 표는 그대로 유효하며, 최종 3-way 표에서는
독립 재측정 seed 20260803의 altproj anchor를 사용한다.

| δ | ρ | −2.7 dB | −2.5 dB |
|---:|---:|---|---|
| 0.02 | **0.90** | **0.009766 [0.005313, 0.017883]; 0/52** | **0 [0, 0.003738]; 0/2** |
| 0.02 | 0.95 | 0.009766 [0.005313, 0.017883]; 0/52 | 0.000977 [0.000172, 0.005511]; 1/2 |
| 0.02 | 1.00 | 0.01660 [0.01039, 0.02643]; 2/47 | 0.01074 [0.00601, 0.01913]; 11/2 |
| 0.05 | 0.90 | 0.15918 [0.13805, 0.18286]; 128/27 | 0.15820 [0.13713, 0.18183]; 162/2 |
| 0.05 | 0.95 | 0.93359 [0.91666, 0.94728]; 899/5 | 0.99219 [0.98466, 0.99604]; 1014/0 |
| 0.05 | 1.00 | 0.99805 [0.99291, 0.99946]; 960/0 | 1.00000 [0.99626, 1.00000]; 1022/0 |

`ρ=1.0`의 판정은 조건부 생존이다.

- `δ=0.02`: 더 이상 붕괴하지 않았고 −2.7 dB에서 legacy보다 CI가
  분리되어 좋았다. 따라서 과거 순수 누적 붕괴의 일부는 σ 편향과
  결합된 결과였다는 증거다.
- 같은 구성은 −2.5 dB에서 성공 블록 11개를 깨고 2개만 구제했다.
- `δ=0.05, ρ=1.0`: 두 SNR 모두 여전히 붕괴했다.

즉 σ 교정은 pure accumulation의 안정 영역을 넓혔지만, 큰 δ까지
damping-free로 만들지는 못했다. robust optimum은 여전히 `ρ=0.9`다.

## 4. 최선 구성 3-way

| Es/N0 | legacy α=.1 | EP α_ep=.01 | altproj δ=.02, ρ=.9 |
|---:|---:|---:|---:|
| −2.7 | 51/1024, 0.04980 [0.03808, 0.06489] | 149/1024, 0.14551 [0.12523, 0.16843] | **10/1024, 0.009766 [0.005313, 0.017883]** |
| −2.5 | 3/1024, 0.002930 [0.000997, 0.008578] | 3/1024, 0.002930 [0.000997, 0.008578] | **1/1024, 0.000977 [0.000172, 0.005511]** |
| −2.4 | 0/1024, 0 [0, 0.003738] | 0/1024, 0 [0, 0.003738] | 0/1024, 0 [0, 0.003738] |

### Legacy canonical 대비 paired 회계

| Es/N0 | EP 파괴/구제 | Altproj 파괴/구제 |
|---:|---:|---:|
| −2.7 | 106 / 8 | **0 / 41** |
| −2.5 | 2 / 2 | **1 / 3** |
| −2.4 | 0 / 0 | 0 / 0 |

EP는 −2.7 dB에서 평균 지표의 문제가 아니라 성공 블록을 대량
파괴한다. Altproj는 같은 지점에서 legacy 실패 블록 41개를 구제하면서
성공 블록을 하나도 깨뜨리지 않았다.

## 5. BP budget 100→50→30

축소 예산에서는 source call 수에 맞춰 같은 endpoint를 재보간했다.

- budget 100: `[5]×20`, source call 19회
- budget 50: `[5]×10`, source call 9회
- budget 30: `[5]×6`, source call 5회

### −2.7 dB

| 스킴 | budget 100 | budget 50 | budget 30 |
|---|---:|---:|---:|
| legacy | **0.06055 [0.04752, 0.07686]** | 0.11133 [0.09350, 0.13207] | 0.84961 [0.82641, 0.87019] |
| EP | **0.17676 [0.15461, 0.20132]** | 0.84961 [0.82641, 0.87019] | 1.00000 [0.99626, 1.00000] |
| altproj | **0.009766 [0.005313, 0.017883]** | 0.36426 [0.33534, 0.39419] | 1.00000 [0.99626, 1.00000] |

동일 스킴 budget-100 대비 budget-50/30 파괴·구제:

- legacy: `53/1`, `808/0`
- EP: `689/0`, `843/0`
- altproj: `363/0`, `1014/0`

### −2.5 dB

| 스킴 | budget 100 | budget 50 | budget 30 |
|---|---:|---:|---:|
| legacy | **0.001953 [0.000536, 0.007093]** | 0.007812 [0.003964, 0.015340] | 0.28906 [0.26213, 0.31758] |
| EP | **0.002930 [0.000997, 0.008578]** | 0.18457 [0.16200, 0.20950] | 0.98047 [0.97002, 0.98732] |
| altproj | **0 [0, 0.003738]** | 0.01074 [0.00601, 0.01913] | 0.86523 [0.84295, 0.88479] |

동일 스킴 budget-100 대비 budget-50/30 파괴·구제:

- legacy: `6/0`, `294/0`
- EP: `186/0`, `1001/0`
- altproj: `11/0`, `886/0`

### 원래 budget-100 legacy canonical 대비 축소 arm 회계

| SNR | budget | legacy | EP | altproj |
|---:|---:|---:|---:|---:|
| −2.7 | 50 | 53/1 | 808/0 | 312/1 |
| −2.7 | 30 | 808/0 | 962/0 | 962/0 |
| −2.5 | 50 | 6/0 | 187/0 | 9/0 |
| −2.5 | 30 | 294/0 | 1002/0 | 884/0 |

어느 스킴도 두 SNR에서 budget-50 성능을 유지하지 못했다. Legacy
budget-50의 −2.5 dB CI는 budget-100과 겹치지만, paired 회계는
구제 없이 6개를 새로 깨뜨렸고 −2.7 dB에서는 CI도 분리된다.
Budget-30은 세 스킴 모두 명확히 불충분하다.

이번 결과는 “편향 제거로 더 빨리 수렴” 예측을 기각한다. 오히려
현재 구조는 source/code 교환 횟수 자체에 강하게 의존한다. 단,
축소 예산에서 σ가 0.02까지 더 빠르게 하강하므로 “BP iteration 부족”과
“빠른 σ 하강”의 효과는 완전히 분리되지 않는다. 요청한 공통 endpoint
조건을 유지하기 위해 별도 재튜닝은 하지 않았다.

## 6. 판정 축 요약

1. **EP가 legacy를 따라잡는가: 아니오.**
   β_ep=1 독립 재측정의 −2.7 dB에서 best EP와 legacy CI가 분리됐고
   EP가 약 2.7배 나쁘다. −2.5 dB에서 α_ep=0.01은 legacy와 정확히
   같은 3/1024이며, −2.4 dB에서도 소표본 동률 범위다.
2. **필요 iteration이 줄어드는가: 아니오.**
   Budget-50부터 세 스킴 모두 악화하며 EP/altproj는 특히 크게 무너진다.
3. **Damping 요구가 완화되는가: 부분적.**
   EP α=0.02는 일부 구간에서 견디지만 α=0.05는 붕괴한다.
   Altproj `δ=.02, ρ=1`은 생존하지만 robust optimum은 `ρ=.9`이며,
   `δ=.05, ρ=1`은 계속 붕괴한다.
4. **현재 3-way 승자: altproj `δ=.02, ρ=.9`, budget 100.**
   β 규약 독립 재측정의 −2.7 dB에서 CI 분리와 0/41 회계로 legacy를
   확실히 넘는다.

## 7. 자율 결정과 근거

1. EP 대표값은 두 SNR 합산 failure 최소가 아니라 **운영점 강건성**을
   기준으로 α=0.01을 선택했다. α=0.02의 −2.7 dB raw 우위는 CI가
   겹쳤고 −2.5 dB에서 더 많은 블록을 파괴했다.
2. Altproj 대표값은 두 SNR 모두 최소/공동 최소 failure이고 legacy
   성공 블록을 깨뜨리지 않은 `δ=.02, ρ=.9`로 정했다.
3. −2.4 dB는 유망 세 arm만 1024 block으로 확장했다. 전부 0건이므로
   순위를 주장하지 않고 Wilson 상한만 기록했다.
4. 축소 budget의 σ는 source call 수에 맞춰 0.3→0.02를 9점/5점으로
   재보간했다. 스킴별 σ 재튜닝 금지 제약을 지키기 위한 선택이다.
5. Altproj의 scheduler 부재는 production 코드 수정 대신 harness
   denoiser-call proxy와 호출 수 assert로 우회했다.
6. 예상과 달리 EP는 회복하지 않았고 altproj만 크게 개선됐다.
   측정 결과를 그대로 판정에 사용했다.

## 8. 산출물과 커밋 제안

- runner: `denoiser_sigma_three_scheme.py`
- −2.7 grid: `results/denoiser_sigma_three_scheme_grid_m2p7.json`
- −2.5 grid: `results/denoiser_sigma_three_scheme_grid_m2p5.json`
- −2.4 selected: `results/denoiser_sigma_three_scheme_selected_b100_m2p4.json`
- budget 50: `results/denoiser_sigma_three_scheme_selected_b50.json`
- budget 30: `results/denoiser_sigma_three_scheme_selected_b30.json`
- β_ep=1 recheck −2.7:
  `results/denoiser_sigma_ep_beta1_recheck_m2p7.json`
- β_ep=1 recheck −2.5:
  `results/denoiser_sigma_ep_beta1_recheck_m2p5.json`

이번 지시에 따라 커밋과 push는 수행하지 않는다. 별도 논리 커밋으로
다음을 제안한다.

`experiments: reevaluate legacy EP and altproj after sigma calibration`

포함 파일은 이 보고서와 `denoiser_sigma_three_scheme.py`다.
`results/`는 저장소 정책대로 ignore 상태로 보존한다.
