# β=0 denoiser σ 하강 스케줄 확정 보고서

## 1. 최종 판정

`practical_sigma`의 legacy `[5]×20` 경로에서 β를 0으로 고정했을 때,
고정 `σ=0.3`을 **기하 하강 `σ: 0.3→0.02`**로 교체한다.

- 확정 구성: `α=0.1`, `β=0`, `sigma_post=3.0`,
  19회 source call에 걸친 geometric `0.3→0.02`
- 고정 `σ=0.3` 대비 측정한 모든 SNR에서 Wilson 95% CI가 분리됐다.
- paired 회계상 −2.8 dB에서만 성공 블록 3개를 깨뜨렸지만 154개를
  구제했고, 나머지 네 SNR에서는 파괴 0개였다.
- 측정 격자 내에서는 끝점 0.02가 모든 SNR에서 최소 또는 공동 최소
  failure를 기록했다. 다만 0.02/0.05 사이 CI는 겹치므로 0.02를
  전역 최적값으로 주장하지 않고 **측정 후보 중 운영 최적값**으로
  승격한다.
- 상태 기반 σ는 시간 스케줄을 이기지 못했다. 가장 나은
  syndrome `c=1.5`도 CI가 겹치는 동률 범위에서 11개 파괴/3개 구제로
  paired 순실패가 8개 늘었다. 단순 시간 스케줄을 유지한다.
- α는 0.1이 명확히 최선이었다. 0.05와 0.2의 CI는 0.1과 완전히
  분리되어 악화됐다.

이 승격은 **β=0으로 고정한 조건 안에서의 판정**이다. 이번 작업은
β 자체를 재비교하지 않았다.

![β=0 σ schedule waterfall](results/denoiser_sigma_beta0_waterfall.png)

## 2. 실험 조건

| 항목 | 값 |
|---|---|
| 브랜치 | `practical_sigma` |
| 채널 | BPSK / AWGN / perfect CSI |
| payload / 전송 길이 | 6272 / 12600 bit |
| CRC | CRC24A |
| decoder | legacy, BP schedule `[5]×20` |
| source input | `bp_post` |
| α / β | 0.1 / **0** |
| sigma_post | **3.0 고정** |
| 비교 σ | fixed 0.3, geom 0.3→{0.02, 0.05, 0.08} |
| 표본 | −2.8~−2.5 dB: 각 1024 block, −2.4 dB: 3200 block |
| 비교 방식 | 각 SNR·shard 안에서 동일 payload와 동일 AWGN을 공유한 paired 비교 |

거친 fixed-σ 스캔에서 β=0 knee가 −2.8~−2.4 dB에 있음을 확인한 뒤
이 범위만 0.1 dB 간격으로 측정했다. −2.4 dB의 3200 block은 장시간
원격 세션 종료를 피하기 위해 독립 seed의 1024/1024/1152 block
세 shard로 실행했다. 각 shard 내부 pairing은 유지했으며 합산 후
Wilson CI와 전이 회계를 다시 계산했다.

## 3. β=0 waterfall

표의 값은 `CRC-BLER [Wilson 95% CI]`이다.

| Es/N0 (dB) | fixed 0.3 | geom 0.3→0.02 | geom 0.3→0.05 | geom 0.3→0.08 |
|---:|---:|---:|---:|---:|
| −2.8 | 0.30469 [0.27727, 0.33357] | **0.15723 [0.13622, 0.18080]** | 0.16211 [0.14081, 0.18594] | 0.19824 [0.17497, 0.22377] |
| −2.7 | 0.17578 [0.15369, 0.20029] | **0.05371 [0.04150, 0.06926]** | 0.05859 [0.04579, 0.07470] | 0.08203 [0.06674, 0.10044] |
| −2.6 | 0.08105 [0.06586, 0.09938] | **0.01367 [0.00816, 0.02282]** | 0.01953 [0.01268, 0.02998] | 0.02832 [0.01979, 0.04038] |
| −2.5 | 0.03223 [0.02304, 0.04491] | **0.00586 [0.00269, 0.01272]** | 0.00684 [0.00332, 0.01404] | 0.00684 [0.00332, 0.01404] |
| −2.4 | 0.008750 [0.006061, 0.012617] | **0.000625 [0.000171, 0.002276]** | 0.000938 [0.000319, 0.002753] | 0.001250 [0.000486, 0.003210] |

로그-BLER 선형 보간으로 본 waterfall 이동량은 다음과 같다.

| 목표 BLER | fixed 0.3 | geom 0.3→0.02 | 이득 |
|---:|---:|---:|---:|
| 10⁻¹ | −2.627 dB | −2.758 dB | **0.131 dB** |
| 10⁻² | −2.410 dB | −2.563 dB | **0.153 dB** |

### fixed 0.3 대비 블록 전이 회계

각 셀은 `파괴 / 구제`이다. 파괴는 fixed 성공→후보 실패, 구제는
fixed 실패→후보 성공이다.

| Es/N0 (dB) | geom 0.3→0.02 | geom 0.3→0.05 | geom 0.3→0.08 |
|---:|---:|---:|---:|
| −2.8 | **3 / 154** | 0 / 146 | 0 / 109 |
| −2.7 | **0 / 125** | 0 / 120 | 0 / 96 |
| −2.6 | **0 / 69** | 0 / 63 | 0 / 54 |
| −2.5 | **0 / 27** | 0 / 26 | 0 / 26 |
| −2.4 | **0 / 26** | 0 / 25 | 0 / 24 |
| 합계 | **3 / 401** | 0 / 380 | 0 / 309 |

고정 0.3 대비 0.02의 CI는 다섯 SNR 모두 분리됐고, 각 SNR에서
파괴 수가 구제 수보다 작다. 따라서 사전에 정한 canonical 승격
조건을 충족한다.

## 4. 상태 기반 σ 재시도

직전 조건부 계측의 256 block, 19 source call 자료로 다음 회귀를
적합했다.

`log(σ_actual) = f(chunk, chunk², log(1+proxy), log²(1+proxy), chunk×log(1+proxy))`

런타임에서는 원본 없이 syndrome weight 또는 현재 payload posterior의
mean|LLR|만 사용하고, `σ=clip(c·σ_hat_actual, 0.01, 0.3)`을 적용했다.
학습 profile은 β=0.1, 이번 평가는 β=0이라는 분포 이동이 있다.

| proxy | block 5-fold CV RMSE | CV Pearson |
|---|---:|---:|
| syndrome weight + chunk | 0.01512 | 0.9701 |
| mean\|LLR\| + chunk | 0.03173 | 0.8630 |

평가는 −2.7 dB, 1024 block, α=0.1, β=0에서 수행했다.

| σ 전략 | 실패/1024 | BLER [Wilson 95% CI] | 시간 스케줄 대비 파괴/구제 |
|---|---:|---:|---:|
| time geom 0.3→0.02 | **53** | **0.05176 [0.03979, 0.06708]** | 기준 |
| syndrome `c=1.5` | 61 | 0.05957 [0.04665, 0.07578] | 11 / 3 |
| syndrome `c=2` | 92 | 0.08984 [0.07383, 0.10893] | 40 / 1 |
| syndrome `c=3` | 133 | 0.12988 [0.11067, 0.15186] | 80 / 0 |
| mean\|LLR\| `c=2` | 100 | 0.09766 [0.08095, 0.11737] | 48 / 1 |

상태 회귀는 `σ_actual` 자체는 잘 예측했지만 성능 최적 σ를 예측하지
못했다. 이는 앞 단계에서 확인했듯 `σ_actual`이 likelihood 정합의
하한이지 prior 교정력과의 최적 균형점은 아니기 때문이다. 특히
`c`가 커질수록 시간 스케줄이 살린 블록을 다시 깨뜨렸다. 가장 나은
`c=1.5`도 CI로는 시간 스케줄과 동률이나 paired 순실패가 +8이고
구현 복잡도가 더 크므로 기각한다.

## 5. α 재확인

선정한 geom 0.3→0.02와 −2.7 dB에서 1024 block을 paired 비교했다.

| α | 실패/1024 | BLER [Wilson 95% CI] | α=0.1 대비 파괴/구제 |
|---:|---:|---:|---:|
| 0.05 | 213 | 0.20801 [0.18426, 0.23394] | 157 / 0 |
| **0.10** | **56** | **0.05469 [0.04235, 0.07035]** | 기준 |
| 0.20 | 495 | 0.48340 [0.45291, 0.51401] | 439 / 0 |

α=0.05는 source 주입이 부족하고, α=0.2는 과주입으로 붕괴한다.
두 후보 모두 α=0.1과 CI가 완전히 분리되므로 α=0.1을 유지한다.

## 6. 자율 결정과 근거

1. **최종 SNR 범위**: β=0 fixed anchor의 256-block 거친 스캔으로
   knee가 −2.8~−2.4 dB임을 확인하고, 요청 예시의 바깥 구간
   (−3.2~−2.9 dB)에는 큰 표본을 쓰지 않았다.
2. **저-BLER 표본 분할**: 3200-block 단일 원격 프로세스가 무출력
   종료되어 1024/1024/1152의 독립 seed shard로 우회했다. 후보 간
   pairing과 총 표본 수 3200은 유지했다.
3. **상태 비교점**: best 시간 스케줄 BLER가 약 0.05이고 fixed
   실패도 충분한 −2.7 dB를 골라 구제와 파괴를 동시에 관측했다.
4. **회귀 형태**: syndrome/LLR의 비선형성과 청크별 궤적을 함께
   반영하되 작은 256-block profile에 과적합하지 않도록 6항 로그
   회귀와 block-wise 5-fold 검증을 사용했다.
5. **mean|LLR| 배율**: 추가 튜닝을 피하기 위해 요청된 후보의 중앙인
   `c=2`를 사전 고정했다.
6. **0.02 선택**: 0.02와 0.05의 CI는 겹치지만, 0.02가 모든 SNR에서
   failure가 더 적거나 같고 총 순구제가 398 대 380으로 컸다.
   동일 복잡도 후보이므로 관측 최선인 0.02를 택했다.
7. **추가 탐색 중단**: 0.02가 시험 격자의 경계이지만 이번 작업의
   확정 후보 밖으로 확장하지 않았다. 따라서 0.02 미만의 전역 최적성은
   주장하지 않는다.

## 7. 코드·산출물·재현

production `decoder.py`와 denoiser 구현은 수정하지 않았다. 새 runner는
기존 `sigma_scheduler` setter, `alpha`, `beta` 속성만 사용한다.

- runner: `denoiser_sigma_beta0_waterfall.py`
- coarse scan: `results/denoiser_sigma_beta0_coarse.json`
- waterfall: `results/denoiser_sigma_beta0_waterfall.json`
- state comparison: `results/denoiser_sigma_beta0_state.json`
- alpha comparison: `results/denoiser_sigma_beta0_alpha.json`
- plot: `results/denoiser_sigma_beta0_waterfall.png`

대표 실행:

```bash
python denoiser_sigma_beta0_waterfall.py waterfall \
  --snrs=-2.8,-2.7,-2.6,-2.5,-2.4 \
  --blocks-per-snr=1024,1024,1024,1024,3200 \
  --batch 64 --seed 20260727 \
  --output results/denoiser_sigma_beta0_waterfall.json

python denoiser_sigma_beta0_waterfall.py state \
  --esn0-db=-2.7 --blocks 1024 --batch 64 --seed 20260731 \
  --time-endpoint 0.02 \
  --profile results/denoiser_sigma_conditional_256.json

python denoiser_sigma_beta0_waterfall.py alpha \
  --esn0-db=-2.7 --blocks 1024 --batch 64 --seed 20260801 \
  --strategy time --time-endpoint 0.02
```

## 8. 커밋 제안

이번 지시에 따라 커밋과 push는 수행하지 않는다. 기존 조건부 계측과
이번 성능 확정을 다음 두 논리 커밋으로 분리하는 것이 적절하다.

1. `diagnostics: measure conditional LDPC denoiser sigma alignment`
2. `experiments: finalize beta0 denoiser sigma schedule`

두 번째 커밋에는 이 보고서와
`denoiser_sigma_beta0_waterfall.py`를 포함한다. `results/`가 저장소
정책상 ignore 대상이면 원자료와 PNG는 원격 작업공간에 보존하고,
재현에 필요한 핵심 수치는 이 보고서에 유지한다.
