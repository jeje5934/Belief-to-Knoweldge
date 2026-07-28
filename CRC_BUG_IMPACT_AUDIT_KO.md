# CRC hard-decision 버그 영향 범위 전수 감사

## 최종 판정

Sionna `CRCDecoder`에 `hard_out=False` decoder의 soft logit을 직접 넘긴 경로는
CRC 검사가 아니다. 저장소 규약인 `logit > 0 -> bit 1`로 먼저 hard decision해야
한다. 모든 branch와 과거 commit에서 CRC 관련 Python 경로 49개를 추적한 결과,
과거 결과를 만든 **직접 오염 경로 43개**, 오염 helper를 호출한 **간접 경로 1개**,
정상/비영향 경로 5개로 분류했다.

영향은 다음과 같다.

- practical_sigma에서 CRC-BLER, CRC 조건부 그룹, CRC early-stop을 사용한 최근
  sigma/3-way/fading/altproj/compression/latency 결과는 hard-CRC 재측정 전까지
  무효 보류한다.
- BER, true payload block error, syndrome, LLR, RMSE처럼 CRC와 독립적으로 직접
  계산한 계측은 유지할 수 있다. 단, CRC 성공/실패로 나눠 집계한 값은 무효다.
- small probe에서는 soft 방식이 실제 성공을 실패로 만드는 방향이 압도적이었다.
  BP-20에서 hard BLER `29/256=0.1133`을 soft 방식은
  `164/256=0.6406`으로 과대 계상했다. hard-pass 227개 중 135개(59.47%)가
  가짜 실패였고 가짜 통과는 0개였다.
- 과거 altproj의 `syn=0인데 CRC 27/32` 관측은 soft-CRC artifact다. 같은 조건의
  512-block hard-CRC 재현에서 `syn=0 & CRC fail=0`,
  `syn=0 & payload wrong=0`이었다. 따라서 “source가 valid-but-wrong codeword
  사이를 판별한다”는 명제와 그에 기반한 CRC-ES 설명은 철회한다.
- CRC-ES 자체가 syndrome-ES와 같은 것은 아니다. hard CRC가 full-graph
  syndrome zero보다 먼저 통과한 block이 164/512였고, 반대는 0개였다.
  CRC-ES의 남는 근거는 **더 이른 payload+CRC 확인에 의한 계산 절감**이다.

긴급 교정은 `crc_utils.py`의 단일 규약으로 practical_sigma의 직접 호출 37개를
통일했다. production `decoder.py`는 CRC를 자체 호출하지 않고 callback을 받으므로
수치 코드는 중립이었다. 오염은 callback 생성자인 실험 script에 있었으며, 잘못된
production 주석만 이번 감사에서 정정했다.

## Sionna에서 soft 입력이 CRC가 아닌 이유

설치된 Sionna의 `CRCDecoder.call()`은 전달된 전체 tensor를 내부
`CRCEncoder`로 다시 처리하고 생성된 remainder의 합이 0인지 본다.
`CRCEncoder.call()`은 float matrix product를 `int64`로 절삭한 뒤 modulo-2를
취한다. 따라서 soft logit의 부호를 bit로 해석하지 않고 **logit 크기의 정수부**가
parity 계산에 들어간다. 예를 들어 같은 hard bit를 뜻하는 `+0.2`, `+1.2`,
`+30`도 서로 다른 정수 입력이 된다.

이 오용의 편향이 수학적으로 항상 한쪽이라고 보장되지는 않는다. 우연한
remainder zero로 가짜 통과도 가능하다. 다만 이번 동일-block probe에서 관측된
방향은 명확히 가짜 실패/지연 쪽이었다.

## [1] 경로 전수 감사

### A. practical_sigma: 직접 오염, 긴급 수정 완료

아래 37개는 긴급 수정 직전 모두 `hard_out=False` 또는 soft info history를
`CRCDecoder`에 직접 전달했다. 현재는 모두 `hard_crc_decode()`를 호출한다.

```text
altproj_compression_study.py
altproj_screen.py
bench_recipe.py
calibrate_sigma_lookup.py
channel_sign_check.py
codec_budget_waterfall.py
colored_duel.py
denoiser_sigma_beta0_waterfall.py
denoiser_sigma_conditional_diag.py
denoiser_sigma_scheme_fairness.py
denoiser_sigma_three_scheme.py
ep_snr_sweep.py
experiment.py
fading_3way.py
fading_experiment.py
fading_legacy_tune.py
latency_baseline_sweep.py
low_budget_source_study.py
ms_smoke.py
plot_comparison.py
practical_ep100_search.py
practical_final_table.py
practical_lowsnr_baseline.py
practical_promptB_fullep_diag.py
practical_promptC_sweep.py
practical_promptOpt1_alphasched.py
practical_promptOpt2_incalpha.py
practical_promptOpt3_hybrid.py
practical_purity_discriminate.py
practical_source_purity.py
practical_true_turbo_search.py
practical_true_turbo_verify.py
practical_turbo_vs_ep.py
sigma_experiment.py
sigma_sweep.py
syndrome_diagnostics.py
visualize_progression.py
```

`fading_4way.py`는 `CRCDecoder` 객체를 `fading_experiment.run_point()`에 넘기는
간접 경로다. 과거 결과는 오염됐고, helper 수정 후 현재 실행은 정상이다.
`latency_source_sweep.py`도 직접 CRC를 호출하지 않지만 과거
`low_budget_source_study.py`의 오염 함수를 호출했으므로 산출물은 오염이다.

### B. 다른 branch의 오염 경로: 아직 branch별 수정 필요

| branch | 오염 script | 범위 |
|---|---|---|
| `pure-EP_tweedie_practical` 계열 | `practical_promptD_damped.py`, `practical_promptD_tweedie_fullep.py` | BLER/NACK만 오염. hard payload BER와 per-round BER은 유지 가능 |
| `compression-baseline` | `compression_baseline/channel_our_waterfall.py`, `compression_baseline/channel_fail_ours.py` | ours legacy/EP waterfall과 CRC-failure image 선정 오염 |
| `no-LDPC` | `experiments/three_arm_ldpc_runner.py`, `experiments/source_residual_diagnostic.py`의 LDPC control | A/B CRC-BLER와 LDPC CRC-failure 조건부 그룹 오염. RSC custom CRC 경로는 정상 |

긴급 commit `de06dc6`은 practical_sigma만 수정했다. 위 세 branch는 재측정 전에
같은 helper 또는 명시적 `(logits > 0)` 변환을 별도로 적용해야 한다.

### C. 정상 또는 비영향 경로

| 경로 | 판정 | 근거 |
|---|---|---|
| `bp20_failure_diagnostic.py` | 정상 | 명시적 hard bits를 CRCDecoder에 전달 |
| `compression_baseline/channel_experiment.py` | 정상 | conventional decoder가 `hard_out=True` |
| `compression_baseline/channel_fail_base.py` | 정상 | conventional decoder가 `hard_out=True` |
| `practical_direction_diag.py` | 비영향 | CRC 객체를 만들지만 결과 계산에 사용하지 않음 |
| no-LDPC `coding/crc.py` 및 RSC/BCJR 경로 | 정상 | `(crc_llr > 0).astype(uint8)` 뒤 custom CRC-16 검사 |
| production `decoder.py` | 수치 경로 정상 | CRC는 caller callback; decoder 내부 직접 CRCDecoder 호출 없음 |

### D. 결과 파일, 보고서, commit 매핑

| 실험군 | 오염 결과 파일/산출물 | 직접 영향 보고서 | 관련 commit |
|---|---|---|---|
| 초기 altproj/colored | `altproj_screen.py`의 stage JSON/CSV, `colored_duel.py` 결과 | `docs/EP_RESEARCH_SUMMARY.md`, `FADING_CHANNEL_REPORT.md` 일부 | `710ef79` 및 후속 altproj commit |
| 과거 EP/legacy/purity | `results/practical_*`, `results/ep_snr_sweep*`, `results/sigma_*` 중 CRC-BLER/NACK | `docs/EP_RESEARCH_SUMMARY.md`, `docs/EP_SCHEDULING_EXPERIMENT.md`, `SIGMA_REPORT.md` | `6247abd`, `196f728`, `0d1958d`, `9a65254` 등 |
| sigma 조건부 | `denoiser_sigma_conditional_256.json`, `denoiser_sigma_schedule_1024.json` | `DENOISER_SIGMA_CONDITIONAL_REPORT_KO.md` | `2d314ef` |
| beta0 sigma waterfall | `denoiser_sigma_beta0_{coarse,waterfall,state,alpha}.json`, shard JSON, derived PNG | `DENOISER_SIGMA_BETA0_WATERFALL_REPORT_KO.md` | `83ddf47` |
| sigma 3-way | `denoiser_sigma_three_scheme_*.json` | `DENOISER_SIGMA_THREE_SCHEME_REPORT_KO.md` | `2b91f1c` |
| scheme fairness/fading | `denoiser_sigma_scheme_{profile,endpoint}_*.json`, `denoiser_sigma_fair_*.json`, `denoiser_sigma_fair_fading_grid_512.{json,png}` | `DENOISER_SIGMA_SCHEME_FAIRNESS_FADING_REPORT_KO.md` | `c6a7b48` |
| fading 일반 | `fading_experiment.py`, `fading_3way.py`, `fading_4way.py`, `fading_legacy_tune.py`의 BLER CSV/JSON | `FADING_CHANNEL_REPORT.md`, `CHANNEL_DIAG_REPORT.md` | `1cbb936`, `f0d38bf`; sigma 공정화는 `c6a7b48` |
| compression gap/SPC/budget | `altproj_compression_waterfall.json`, `altproj_spc_*.json`, `altproj_budget*.json`, `altproj_rate07_probe.json`, derived PNG | `ALTPROJ_COMPRESSION_GAP_REPORT_KO.md` | `ae22a81`~`1aaa317` |
| 저예산 retune/codec duel | `low_budget_{sigma_post,coarse,refined,ours,codec}_*.json`, `low_budget_codec_duel.{json,png}` | `LOW_BUDGET_CODEC_DUEL_REPORT_KO.md` | `4731053`, `20e4c1e`, `e70a48a` |
| latency matching | `latency_source_sweep_512.json`, `latency_baseline_sweep_1024.json`, `latency_matching_*.{json,png}` | `LATENCY_MATCHING_REPORT_KO.md` | `8572d8a`~`94ebfe8` |
| fixed latency SNR | `fixed_latency_{source,baseline}_*.json`, `fixed_latency_snr_*.{json,png}` | `FIXED_LATENCY_SNR_REPORT_KO.md` | `987993c`~`b761780` |
| compression-baseline ours | `compression_baseline/results/our_waterfall.json`, ours failure NPZ | `compression_baseline/REPORT_PHASE2.md`의 ours arm | `6187337` |
| no-LDPC LDPC controls | `three_arm_A/B_*.json`, `three_arm_waterfall_final.{json,csv,png}`, `source_residual_ldpc64.json`의 CRC-conditioned fields | `NO_LDPC_SUMMARY.md`, `THREE_ARM_WATERFALL_REPORT_KO.md`, `RSC_SOURCE_WIRING_AUDIT_KO.md` | `d772ca1`, 문서 통합 `1368cbb` |

다음 산출물은 위 파일과 함께 있어도 CRC와 독립적이므로 정상이다.

- `denoiser_sigma_alignment_32.json`: 원본 대비 soft-pixel RMSE/denoiser 출력 계측
- `latency_architecture_profile.json`: 추상 latency 상수 계측
- true payload BER/BLER, syndrome, LLR, source RMSE/bit-plane 계측 필드

다만 `denoiser_sigma_conditional_256.json`처럼 최종 CRC로 성공/실패를 나눈 파일은
그룹 라벨 자체가 오염됐으므로 RMSE 원시값이 있어도 기존 조건부 평균은 다시
계산해야 한다.

#### no-LDPC 결론의 특례

`three_arm_ldpc_runner.py`는 잘못된 CRC-BLER와 별도로 true payload BLER를 저장했다.
저장 원자료로 다시 계산하면 knee @ BLER `1e-2`는 B가 CRC 기준 `-2.6608 dB`,
true payload 기준 `-2.6713 dB`로 차이가 약 `0.0105 dB`뿐이다. A는 해당 knee에서
같다. C의 true/CRC knee는 `+3.6624 dB`로 같다. 따라서 no-LDPC의 5.9~6.3 dB
열세와 LDPC 복귀 판정은 **true payload BLER로 독립 보존된다**. 문서의 “CRC pass가
성공 기준” 표기와 LDPC CRC-conditioned 보조 계측은 정정 대상이지만 갈래 종료
결론을 뒤집지는 않는다.

## [2] 버그 방향 소규모 재현

조건: raw Fashion-MNIST, CRC24A + LDPC, BPSK/AWGN, Es/N0 `-1.92 dB`,
256 block, 동일 logits에 hard/soft CRC를 동시에 적용했다.

| BP iter | hard pass | soft pass | hard-pass/soft-fail | hard-fail/soft-pass |
|---:|---:|---:|---:|---:|
| 15 | 32 | 0 | 32 | 0 |
| 20 | 227 | 92 | 135 | 0 |
| 25 | 256 | 245 | 11 | 0 |
| 30 | 256 | 256 | 0 | 0 |

BP-20에서:

- 실제 hard BLER: `29/256 = 0.1133`
- soft 방식 apparent BLER: `164/256 = 0.6406`
- 가짜 실패율: `135/227 = 59.47%`
- 가짜 통과율: `0/29 = 0%` (95% 상한을 주장하는 성능 실험이 아니라 관측치)

CRC-ES 첫 통과 iteration은 hard 기준 `15:32, 20:195, 25:29`, soft 기준
`20:92, 25:153, 30:11`이었다. 256개 중 177개에서 soft CRC가 늦었고,
79개는 같았으며 soft가 이른 block은 0개였다. 따라서 이 조건에서는 BLER을
과대평가하고 early-stop을 5~10 BP iteration 늦춰 latency/실제-iteration을
과대평가했다.

스킴마다 logit scale이 다르므로 가짜 실패율도 다를 수 있다. 그러므로 기존
3-way 순위가 공통 offset이라고 가정할 수 없고, 특히 EP/altproj처럼 output
scale이 다른 arm의 상대 순위는 재측정해야 한다.

## [3] `syn=0 != 정답` 재검증

과거 `altproj_screen.py`의 조건을 그대로 사용했다.

- Fashion-MNIST payload 6272, CRC24A, `N=12600`
- Eb/N0 `+0.6 dB` = Es/N0 `-2.4131 dB`
- altproj `delta=0.02`, `rho=0.9`, `sigma_den=0.3`, `sigma_post=3.0`
- warm-start, `[5]x20`, probe 중 early-stop off
- 512 block, batch 128, seed 4242

| round | syndrome zero | hard CRC pass | 잘못된 soft CRC pass | syn0 & hard CRC fail | syn0 & payload wrong |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 10 | 0 | 0 | 0 |
| 5 | 117 | 201 | 14 | 0 | 0 |
| 6 | 383 | 442 | 240 | 0 | 0 |
| 7 | 492 | 502 | 452 | 0 | 0 |
| 8 | 508 | 510 | 503 | 0 | 0 |
| 9 | 512 | 512 | 510 | 0 | 0 |

첫 event 비교는 hard CRC가 먼저 164, 같은 round 348, syndrome이 먼저 0이다.
soft CRC를 쓰면 round 6에서 syndrome zero 383개 중 143개가 CRC fail로 보인다.
이것이 과거 32-block `32 vs 27` 패턴의 재현이며 실제 valid-but-wrong이 아니다.

따라서 다음 문구를 철회한다.

1. “zero syndrome은 정답을 뜻하지 않는다”는 당시 실측 주장
2. “source가 valid codeword들 사이에서 정답을 판별한다”는 altproj gain 설명
3. “post-syn0 source round가 그 판별을 수행하므로 CRC-ES가 필요하다”는 근거

정정 대상은 `decoder.py`의 해당 주석/속성 docstring, `altproj_screen.py`의
“syndrome NOT sufficient” 주석, `docs/EP_RESEARCH_SUMMARY.md`의 altproj 구조개선
서술, 이 내용을 인용하는 발표 자료/메모리다. 코드 주석 두 곳과 HARQ 보고서는
이번 감사에서 바로 정정했다. 연구 요약과 과거 결과표는 재측정 결과와 함께
일괄 갱신해야 하므로 목록으로 남겼다.

## [4] 재측정 우선순위와 예상 비용

비용은 기존 실행의 block 수와 GPU throughput에 기반한 계획값이다. 실제 wall
time은 source call 수, codec I/O, early-stop 분포에 따라 달라지므로 범위로 적었다.

| 우선 | 재측정 | 이유 | 권장 규모 | 예상 비용 |
|---:|---|---|---:|---:|
| P0 | 저예산 ours/codec 공정 대결 | 현재 논문 포지션과 budget-LUT의 직접 근거. BLER와 실제 iteration 모두 오염 | 4 budgets x 관심 SNR, 1024; `1e-2` 부근 3200 | 약 15~30만 block, GPU 3~6 h |
| P1 | sigma 교정 canonical + 공정 3-way | 이후 모든 ours 설정의 anchor. 스킴별 logit scale 때문에 순위 보존 불가 | sigma endpoints 최소격자 후 legacy/EP/altproj 2~4 SNR | 약 5~10만 block, GPU 1.5~3 h |
| P2 | fixed-latency SNR/latency graph | 교수/논문 핵심 그림. P0/P1 결과를 최대 재사용 가능 | 부족 SNR만 1024/3200 | 재사용 시 2~6만 block, GPU 0.5~1.5 h + plotting |
| P3 | fading 공정 3-way | “AWGN 최강이 mismatch에서 무너짐” 서사의 근거 | `sigma_e2={0,.1,.2}` x 과거 Eb/N0 x 3 arms, 512~1024 | 약 3~7만 block, GPU 1~2 h |
| P4 | 초기 altproj/CRC-ES 스크리닝 | valid-codeword 판별 서사는 이미 기각. 성능 sweetspot/early-stop 절감만 재확인 | 상위 설정만 512~1024 | 약 1~2만 block, GPU 0.3~0.8 h |
| P5 | 과거 EP/purity/sigma 배제 사슬 | 역사적 설명 정리용. 논문 최종 표에 쓰지 않으면 생략 가능 | 대표점/상위 후보만 | 약 2~5만 block, GPU 0.5~1.5 h |
| P6 | no-LDPC A/B CRC 곡선 | true payload BLER가 결론을 이미 보존. 문서 정합성 목적뿐 | 기존 SNR 재집계 또는 선택 재실행 | 재집계 0; 재실행 GPU 0.3~0.7 h |

권장 실행 순서는 `P1 -> P0 -> P2 -> P3`다. P1이 새 canonical 설정과 스킴
순위를 정하고, P0가 그 설정으로 논문의 저예산 대결을 복구하며, P2는 앞 두
원자료를 재사용한다. 이번 작업에서는 요청대로 [3]의 512-block 재검증 외 대규모
성능 재측정을 수행하지 않았다.

## 산출물

- 방향 probe: `crc_soft_hard_probe.py`
- 방향 원자료: `results/crc_soft_hard_probe_256.json`
- altproj 재검증: `altproj_syn_crc_recheck.py`
- altproj 원자료: `results/altproj_syn_crc_recheck_512.json`
- 긴급 공통 helper: `crc_utils.py`
- 본 감사: `CRC_BUG_IMPACT_AUDIT_KO.md`
