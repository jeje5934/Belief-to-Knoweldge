# no-LDPC alpha schedule 최적화 보고서

> 후속 실제 knee 재조정은 `NO_LDPC_OPERATING_POINT_RECALIBRATION_KO.md`에
> 기록했다. 그 단계에서 `(sigma_post=3, llr_clip=45, outer=4, alpha=0.1)`이
> 관측상 최상 후보였지만 기존 canonical과 95% Wilson CI가 겹치고 계산량이
> 약 2배여서 기본값으로 승격하지 않았다. 따라서 이 문서의 2-pass 설정은
> 현재 canonical으로 유지된다.

## 결론

기본 설정은 다음과 같이 고정한다.

```text
outer_iterations = 2
alpha_schedule = (0.1, 0.1)
bcjr_mode = logmap
```

`(0.125, 0.1)`은 독립 검증에서 block error 1개 차이로 사실상 동률이었고
bit error는 더 많았다. 3-pass 후보는 fine search의 이득이 새 seed에서
재현되지 않았다. 따라서 seed 의존성이 낮고 계산량이 작은 2-pass constant
schedule을 canonical 설정으로 선택했다. max-log는 선택 가능한 throughput
모드로 유지하되 exact log-MAP을 정확도 기준 기본값으로 둔다.

## 탐색 방법

- 모든 schedule은 같은 payload와 같은 표준 정규 잡음을 사용한 paired
  comparison이다.
- Fashion-MNIST full-score source/SPC SISO를 사용했다.
- knee probe로 탐색 구간을 대략 Es/N0 0.5--2.0 dB로 좁혔다.
- seed `20260722`에서 coarse/fine search를 수행했다.
- seed `20260723`, `20260724`는 후보 선택 이후의 독립 validation으로만
  사용했다.
- 우선순위는 block error, bit error, BCJR pass 수, runtime 순서다.
- 표본 수가 작으므로 통계적 유의성이나 최종 성능 우위를 주장하지 않는다.

## 결과

coarse search에서는 `(0.1, 0.1)`이 48개 평가 block 중 35 block error,
237 bit error로 1위였다. 4-pass `(0.1, 0.1, 0.1, 0.1)`은 block error가
같고 bit error와 계산량이 더 컸다. alpha 0.2 이상은 일관되게 불리했다.

fine search에서 `(0.075, 0.1, 0.1)`이 28/48 block error로 일시적으로
앞섰지만, 독립 seed `20260723`에서는 42/48로 `(0.1, 0.1)`과 같고 bit
error가 254 대 231로 더 많았다. 3-pass 이득은 재현되지 않았다.

독립 validation 합계는 다음과 같다.

| mode | schedule | block errors | bit errors | 비고 |
| --- | --- | ---: | ---: | --- |
| log-MAP | `(0.1, 0.1)` | 102/144 | 458 | canonical |
| log-MAP | `(0.125, 0.1)` | 101/144 | 469 | 사실상 동률 |
| log-MAP | `(0.0,)` | 117/144 | 1,163 | source feedback 없음 |
| max-log | `(0.1, 0.1)` | 100/144 | 451 | optional throughput |
| max-log | `(0.0,)` | 116/144 | 1,162 | source feedback 없음 |

max-log의 작은 수치 우위는 표본 변동으로 간주한다. 32-frame 실행에서
2-pass max-log는 약 2.1--2.4초, log-MAP은 약 2.6--3.0초로 max-log가
대략 20% 빨랐다.

## BCJR 실행 최적화

schedule 탐색 전에 BCJR alpha/beta recursion을 batch-vectorized 형태로
바꿨다. scalar reference는 유지했으며 log-MAP/max-log와 puncturing 조합을
포함한 무작위 batch가 `1e-12` 이내에서 일치한다.

16-frame, 7,075-step log-MAP 1-pass 측정은 약 51.5초에서 약 0.35초로
단축됐다. 이는 수식이나 hard-decision 정책 변경이 아니라 frame 축을
동시에 처리한 실행 최적화다.

## 재현 명령

```bash
python3 experiments/no_ldpc_schedule_search.py --blocks 16 \
  --esn0-db 0.5 1.0 1.5 --source full_score --bcjr-mode logmap \
  --seed 20260722 \
  --output results/no_ldpc_schedule_coarse_seed20260722.json

python3 experiments/no_ldpc_schedule_search.py --blocks 16 \
  --esn0-db 0.75 1.25 1.75 \
  --schedules 0 0.1,0.1 0.125,0.1 0.075,0.1,0.1 0.1,0.1,0.1 \
  --source full_score --bcjr-mode logmap --seed 20260723 \
  --output results/no_ldpc_schedule_validation_seed20260723.json

python3 experiments/no_ldpc_schedule_search.py --blocks 32 \
  --esn0-db 1.0 1.5 2.0 --schedules 0 0.1,0.1 0.125,0.1 \
  --source full_score --bcjr-mode logmap --seed 20260724 \
  --output results/no_ldpc_schedule_final_logmap_seed20260724.json

python3 experiments/no_ldpc_schedule_search.py --blocks 32 \
  --esn0-db 1.0 1.5 2.0 --schedules 0 0.1,0.1 0.125,0.1 \
  --source full_score --bcjr-mode maxlog --seed 20260724 \
  --output results/no_ldpc_schedule_final_maxlog_seed20260724.json
```

다음 단계는 이 설정을 고정한 채 Stage C에서 독립 frame 수를 늘리고,
AWGN 이후 perfect/imperfect-CSI fading으로 동일한 paired protocol을 확장하는
것이다.
