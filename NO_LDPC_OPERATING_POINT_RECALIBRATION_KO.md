# no-LDPC 실제 동작점 재조정 보고서

> 이 갈래는 종료됐다. 최종 판정과 문서 지도는
> [`NO_LDPC_SUMMARY.md`](NO_LDPC_SUMMARY.md)를 참조한다. 아래 내용은
> knee와 파라미터 재조정의 상세 근거다.

## 최종 판정

- 튜닝 동작점: **Es/N0 3.0 dB**, AWGN + perfect CSI, exact log-MAP.
- 관측상 최상 후보:
  `sigma_post=3`, `llr_clip=45`, `outer_iterations=4`, constant `alpha=0.1`.
- 기존 canonical:
  `sigma_post=3`, `llr_clip=30`, `outer_iterations=2`, constant `alpha=0.1`.
- 독립 seed 두 개를 합친 256 block에서 후보는 11 errors (4.30%,
  Wilson 95% CI 2.42--7.53%), canonical은 13 errors (5.08%,
  CI 2.99--8.49%)였다.
- CI가 겹치므로 개선은 입증되지 않았다. 후보는 runtime도 약 2배다.
  사전 판정 원칙에 따라 **후보를 기본값으로 승격하지 않고 기존 canonical을
  유지**한다. 이 결론은 후보가 나쁘다는 뜻이 아니라 128--256 block
  범위에서 차이를 확정할 근거가 부족하다는 뜻이다.

`sigma=0.3`은 요청된 재조정 축에 포함되지 않았으므로 고정했다. classic
turbo message 식, rate matching, exact log-MAP, 기존 LDPC 경로는 변경하지
않았다.

## 1. knee 이동

canonical 설정, seed `20260725`, 64 paired Fashion-MNIST block으로 스캔했다.

| Es/N0 (dB) | block errors | BLER | Wilson 95% CI | bit errors | BER |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2.0 | 22/64 | 34.38% | 23.92--46.60% | 60 | 1.495e-4 |
| 2.5 | 12/64 | 18.75% | 11.06--29.97% | 41 | 1.021e-4 |
| 3.0 | 5/64 | 7.81% | 3.38--17.02% | 19 | 4.733e-5 |
| 3.5 | 3/64 | 4.69% | 1.61--12.90% | 9 | 2.242e-5 |
| 4.0 | 2/64 | 3.12% | 0.86--10.70% | 2 | 4.982e-6 |

3.0, 3.5, 4.0 dB의 관측 BLER가 모두 목표 0.01--0.1 안에 있었다.
3.0 dB는 오류가 더 많이 관측되어 같은 screening 표본에서 후보 간 차이를
볼 가능성이 크므로 튜닝 지점으로 선택했다. 이후 모든 sweep과 validation은
3.0 dB에서만 수행했다.

## 2. sigma_post 및 llr_clip

### sigma_post coarse sweep

seed `20260726`, 64 paired block, 기존 `(outer=2, alpha=0.1, llr_clip=30)`을
사용했다.

| sigma_post | block errors | BLER | Wilson 95% CI | bit errors | BER |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 3/64 | 4.69% | 1.61--12.90% | 8 | 1.993e-5 |
| 6 | 5/64 | 7.81% | 3.38--17.02% | 12 | 2.989e-5 |
| 12 | 5/64 | 7.81% | 3.38--17.02% | 13 | 3.239e-5 |

CI는 넓게 겹치지만 coarse 축에서 BLER와 BER이 모두 가장 낮은
`sigma_post=3`을 유지했다. 세밀 최적화는 하지 않았다.

### llr_clip과 실제 LLR 분포

같은 seed와 block에서 `llr_clip={30,45,60}`을 paired 비교했다.

| llr_clip | block errors | bit errors | source extrinsic clip 비율 | final APP p99 | APP threshold 초과율 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 3/64 | 8 | 41.31% | 42.96 | 26.55% |
| 45 | 2/64 | 5 | 17.66% | 43.24 | 0.56% |
| 60 | 5/64 | 11 | 16.04% | 44.45 | 0.0007% |

45에서 final BCJR APP `|LLR|`은 p50 25.40, p90 34.83, p95 37.68,
p99 43.24, p99.9 49.36, max 61.76이었다. clip은 source extrinsic에
적용되며, score posterior의 매우 큰 값 때문에 45에서도 전체 pass 값의
17.66%에 실제로 발동했다. 동시에 BCJR APP 분포에서는 약 p99 부근이라
30처럼 광범위하지도, 60처럼 사실상 무의미하지도 않았다. 작은 표본 성능도
가장 좋아 후보 `llr_clip=45`를 선택했다.

## 3. outer_iterations x alpha 격자

첫 32-block screen(seed `20260727`)은 16개 cell 중 15개가 zero error라
판별력이 없었다. 동작점을 바꾸지 않고 64 block, seed `20260726`으로
재실행했다. 표 값은 `BLER (bit errors)`이며 모두 `sigma_post=3`,
`llr_clip=45`다.

| outer \ alpha | 0.02 | 0.05 | 0.10 | 0.15 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 10.94% (19) | 6.25% (11) | **3.12% (5)** | 15.62% (28) |
| 4 | 10.94% (19) | 6.25% (11) | **3.12% (5)** | 6.25% (10) |
| 6 | 10.94% (19) | 6.25% (11) | **3.12% (5)** | 4.69% (9) |
| 8 | 10.94% (19) | 6.25% (11) | **3.12% (5)** | 4.69% (9) |

alpha 0.02와 0.05는 pass를 늘려도 hard decision이 변하지 않았다. alpha
0.15는 2-pass에서 과주입됐고 추가 pass로 일부 회복했지만 alpha 0.1을
넘지 못했다. alpha 0.1은 64-block screen에서 outer 2--8이 동률이었다.

## 4. 상위 후보 CI와 독립 seed 검증

### 상위 후보 128-block 검증

seed `20260728`, `sigma_post=3`, `llr_clip=45` 결과다.

| outer | alpha | block errors | BLER | Wilson 95% CI | bit errors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.10 | **6/128** | **4.69%** | 2.17--9.85% | 18 |
| 6 | 0.10 | 6/128 | 4.69% | 2.17--9.85% | 18 |
| 8 | 0.10 | 6/128 | 4.69% | 2.17--9.85% | 18 |
| 2 | 0.05 | 8/128 | 6.25% | 3.20--11.85% | 23 |
| 2 | 0.10 | 9/128 | 7.03% | 3.74--12.82% | 22 |
| 6 | 0.15 | 10/128 | 7.81% | 4.30--13.78% | 29 |

outer 4/6/8은 error block과 bit error가 완전히 같았다. 따라서 관측상 최상
후보는 최소 pass인 `(outer=4, alpha=0.1)`로 정했다.

### 독립 seed와 canonical 비교

각 seed 안에서는 payload와 AWGN noise가 설정 간 paired다.

| seed | 설정 | block errors | BLER | Wilson 95% CI | bit errors |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20260728 | 후보 `(3,45,4,0.1)` | 6/128 | 4.69% | 2.17--9.85% | 18 |
| 20260728 | canonical `(3,30,2,0.1)` | 7/128 | 5.47% | 2.67--10.86% | 19 |
| 20260729 | 후보 `(3,45,4,0.1)` | 5/128 | 3.91% | 1.68--8.82% | 12 |
| 20260729 | canonical `(3,30,2,0.1)` | 6/128 | 4.69% | 2.17--9.85% | 13 |
| combined | 후보 | **11/256** | **4.30%** | **2.42--7.53%** | 30 |
| combined | canonical | 13/256 | 5.08% | 2.99--8.49% | 32 |

후보의 point estimate는 두 seed 모두 낮았지만 CI는 각 seed와 합산 모두
겹친다. 후보 runtime은 seed별 약 20.6--20.9초, canonical은 약
10.6초였다. 따라서 0.78 percentage-point 관측 차이를 약 2배 계산량과
교환할 통계적 근거가 없다. canonical 대비 개선 판정은 **미확정**이다.

## 자율 판단 및 실행 우회

1. knee 후보 세 점 중 3.0 dB를 고른 이유는 목표 범위 안에서 error event가
   가장 많아 screening 판별력을 높이기 위해서다.
2. 최초 32-block 격자가 거의 모두 zero error라 중단하지 않고 같은 3.0 dB,
   64 block으로 확장했다. 동작점을 낮춰 서사를 맞추지 않았다.
3. 64-block 격자에서 outer 2/4/6/8이 alpha 0.1로 동률이었으나 128-block
   검증에서 outer 4가 outer 2보다 낮았다. 이어 outer 6/8을 추가 확인했고
   outer 4와 완전히 같아 최소 pass인 4를 후보로 정했다.
4. 장시간 GPU 실행 중 CRC `bool` callable 오류가 이전 실행에 이어 두 번째로
   재현됐다. 명시적 0 비교로 동일 연산을 수행하도록 바꿔 재실행했으며 CRC
   known vector와 전체 테스트로 동작 불변을 확인했다.
5. CI가 겹치는데도 point estimate만으로 우위를 주장하지 않았다. runtime
   비용까지 고려해 후보를 코드 기본값으로 승격하지 않았다.

## 재현 명령 개요

모든 명령은 `--source full_score --bcjr-mode logmap --device cuda`를 사용한다.

```bash
python3 experiments/no_ldpc_schedule_search.py --blocks 64 \
  --esn0-db 2 2.5 3 3.5 4 --schedules 0.1,0.1 \
  --sigma-post 3 --llr-clip 30 --seed 20260725 \
  --source full_score --bcjr-mode logmap --device cuda

python3 experiments/no_ldpc_schedule_search.py --blocks 64 \
  --esn0-db 3 --schedules 0.1,0.1 --sigma-post 3 --llr-clip 45 \
  --seed 20260726 --source full_score --bcjr-mode logmap --device cuda

python3 experiments/no_ldpc_schedule_search.py --blocks 128 \
  --esn0-db 3 --schedules 0.1,0.1 0.05,0.05 0.1,0.1,0.1,0.1 \
  0.15,0.15,0.15,0.15,0.15,0.15 --sigma-post 3 --llr-clip 45 \
  --seed 20260728 --source full_score --bcjr-mode logmap --device cuda
```

전체 JSON 결과는 ignore된 `results/no_ldpc_recal_*.json`에 남겨 두었다.
