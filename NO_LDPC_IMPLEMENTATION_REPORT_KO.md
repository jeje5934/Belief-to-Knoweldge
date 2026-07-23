# no-LDPC 구현 진행 보고서

> 이 갈래는 종료됐다. 최종 판정과 문서 지도는
> [`NO_LDPC_SUMMARY.md`](NO_LDPC_SUMMARY.md)를 참조한다. 아래 내용은
> 단계별 구현·검증 기록으로 보존한다.

## 1단계: 저장소 감사

- 작업 브랜치: `no-LDPC`
- 기존 `practical_sigma` 및 LDPC 구현은 수정하지 않았다.
- Fashion-MNIST 표현은 `28 x 28`, `uint8`, 픽셀당 8비트이며 비트 순서는 MSB-first다.
- 원본 payload는 6,272비트이고 기존 고정 채널 길이는 12,600비트다.
- LLR 부호는 `log P(1)/P(0)`이며 양수가 bit 1이다.
- 기존 CRC는 CRC24A지만 새 경로는 명세대로 CRC-16-CCITT-FALSE를 사용한다.
- 기존 채널은 BPSK/AWGN과 QPSK fast Rayleigh fading, perfect/imperfect CSI를 제공한다.
- pretrained score 모델은 그대로 동결하며 첫 구현은 one-step EDM 호출만 사용한다.
- 최초 Git 감사에서는 lossless neural-compression baseline을 찾지 못했지만,
  이후 ignore된 `compression_baseline/` 아래에서 checkpoint, legacy bytecode,
  과거 결과 자원을 발견했다.

추가/수정 파일:

- `archive/CODEX_HANDOFF_NO_LDPC.md` (당시 root의 원 작업 명세)
- `NO_LDPC_DESIGN.md`
- `NO_LDPC_IMPLEMENTATION_REPORT_KO.md`

미해결 위험:

- RSC-only/no-SPC rate-1/2 출력은 12,582비트라 목표보다 18비트 짧다. 이 충돌은 명시적 `allow_parity_repetition` 정책으로 해결했다. 모든 parity를 한 번 전송하고 균일한 18개 위치를 반복하며, 수신기에서 반복 LLR을 합산한다.
- pretrained score 모델은 현재 `M=8`, `28 x 28` 구조에 묶여 있다.
- neural-compression checkpoint는 57 MB의 ignore된 외부 자원이다. 소스는
  복원해 Git에 포함하지만 checkpoint 자체는 포함하지 않는다.
- 고정 4,873비트 압축 컨테이너는 복구된 2,500-image 측정의 관측 최대값이다.
  전체 데이터셋에 대한 수학적 상한은 아니며, 초과 시 절대 truncate하지
  않고 해당 index와 길이를 보고하며 실행을 중단한다.

## 2단계: CPU 코딩 컴포넌트

구현한 수식과 규칙:

- CRC-16-CCITT-FALSE: `poly=0x1021`, `init=0xFFFF`, MSB-first.
- SPC: 각 M비트 심볼에 XOR parity 1비트를 체계적으로 추가.
- RSC: feedback `13_o`, parity `15_o`, 8-state, zero termination.
- rate matching: systematic 7,075개를 모두 보존하고 parity 5,525개를 균일하게 선택해 정확히 12,600비트 전송.
- BCJR: exact log-MAP 기본, max-log 선택 가능.
- 채널 메시지: `L_C_to_S = L_APP - L_A`; systematic channel LLR은 다시 빼지 않는다.

추가 파일:

- `coding/crc.py`
- `coding/interleaver.py`
- `coding/spc.py`
- `coding/rsc.py`
- `coding/puncturing.py`
- `coding/bcjr.py`

검증:

- CRC known vector `123456789 -> 0x29B1` 통과.
- SPC 후보 전수 parity 및 minimum distance 2 검증 통과.
- RSC zero termination 통과.
- 짧은 trellis에서 log-MAP/max-log 모두 brute-force MAP과 일치.
- punctured parity LLR=0 조건에서도 brute-force와 일치.

## 3단계: source/SPC SISO와 반복 디코더

구현한 업데이트:

```text
log P(a | Q) = normalize(log P_score(a | Q_sys)
                         + log P_Q(parity(a)))
L_S_to_C = L_source_posterior - Q
L_A_next = alpha * clip(L_S_to_C, -L_max, L_max)
```

score categorical은 systematic cavity를 denoiser 입력으로 이미 소비하므로 SPC 결합에서 systematic likelihood를 다시 곱하지 않는다. CRC 위치의 source extrinsic은 항상 0이다.

추가 파일:

- `decoders/source_spc_siso.py`
- `decoders/rsc_source_iterative_decoder.py`
- `experiments/no_ldpc_smoke.py`
- `tests/test_*.py`

검증:

- 총 50개 `unittest`가 원격 환경에서 통과했다.
- noiseless payload exact recovery 및 CRC pass 통과.
- `alpha=0`이 source module 없는 BCJR-only 결과와 일치.
- CRC source-extrinsic 0 검증 통과.
- 기본 자원 길이 12,600 검증 통과.
- 최종 hard decision도 raw source posterior가 아니라
  `Q + alpha * clip(L_S_to_C)`를 사용하도록 검증했다.
- no-SPC baseline의 18 parity 반복, LLR 합산, exact `N=12,600`, noiseless CRC recovery를 검증했다.
- 기존 5G LDPC-only, LDPC warm-score, RSC/no-SPC, RSC/SPC,
  RSC/score, RSC/score+SPC, neural-compression+LDPC의 7-arm paired matrix
  진입점을 1블록으로 실행 검증했다.
- source-SISO 호출 수와 실제 score-network 호출 수를 별도 계측한다.
- 모든 새 실험은 PSNR, 7×7 uniform-window SSIM, 평균 normalized MSE,
  CRC 실패 조건부 PSNR을 함께 기록한다.

소규모 실행 결과:

| payload/조건 | 방식 | true BLER | CRC BLER | BER | 비고 |
| --- | --- | ---: | ---: | ---: | --- |
| Fashion-MNIST 8블록, Es/N0 8 dB | BCJR-only, same SPC frame | 0/8 | 0/8 | 0 | alpha=0 불변성 arm |
| Fashion-MNIST 8블록, Es/N0 8 dB | SPC-only | 0/8 | 0/8 | 0 | exact APP |
| Fashion-MNIST 8블록, Es/N0 4 dB | BCJR-only, same SPC frame | 1/8 | 1/8 | `5.979e-5` | 작은 표본 |
| Fashion-MNIST 8블록, Es/N0 4 dB | SPC-only | 0/8 | 0/8 | 0 | 작은 표본, 성능 결론 금지 |
| Fashion-MNIST 1블록, Es/N0 8 dB | frozen score+SPC | 0/1 | 0/1 | 0 | shape/finite smoke only |
| Fashion-MNIST 1블록, Es/N0 12 dB | neural compression+LDPC | 0/1 | 0/1 | 0 | 2,500-bit stream, bit-exact decompression |
| Fashion-MNIST 1블록, Es/N0 -8 dB | neural compression+LDPC | 1/1 | 1/1 | 0.2007 | CRC outage, entropy decode 호출 0회 |
| Fashion-MNIST 8블록, Es/N0 -4 dB | neural compression+LDPC | 7/8 | 7/8 | 0.1981 | CRC-pass 1개만 decode, 해당 block bit-exact |

위 표본 수로 SNR knee 또는 성능 우위를 주장하지 않는다.

재현 명령:

```bash
python3 -m unittest discover -s tests -v
python3 experiments/no_ldpc_smoke.py --blocks 8 --esn0-db 4 8 \
  --schemes bcjr_only spc_only
python3 experiments/no_ldpc_smoke.py --blocks 1 --esn0-db 8 \
  --schemes full_score --device cuda
python3 experiments/no_ldpc_compare.py --blocks 8 --esn0-db 4 8 \
  --outer-list 1 2 4 6 8 --alpha-list 0 .05 .1 .15 .2 .3 1 \
  --source spc_only
python3 experiments/no_ldpc_fading.py --blocks 8 --esn0-db 8 12 \
  --perfect-csi --schemes bcjr_only spc_only
python3 experiments/no_ldpc_fading.py --blocks 8 --esn0-db 8 12 \
  --sigma-e2 0.05 --schemes bcjr_only spc_only
python3 experiments/no_ldpc_baseline_matrix.py --blocks 8 --esn0-db 4 8 \
  --outer-iterations 2 --alpha 0.1 --device cuda
python3 experiments/no_ldpc_baseline_matrix.py --blocks 1 --esn0-db 12 \
  --schemes neural_compression_ldpc --device cuda
python3 experiments/no_ldpc_schedule_search.py --blocks 32 \
  --esn0-db 1.0 1.5 2.0 --schedules 0 0.1,0.1 0.125,0.1 \
  --source full_score --bcjr-mode logmap --seed 20260724 \
  --output results/no_ldpc_schedule_final_logmap_seed20260724.json
python3 experiments/no_ldpc_plot.py results/no_ldpc_smoke_stageA.json \
  --output results/no_ldpc_smoke_stageA_bler.png
```

다음 단계:

1. 선택된 `(0.1, 0.1)` exact log-MAP 설정을 Stage C의 더 큰 독립 표본에서
   통계적으로 검증한다.
2. 4,873-bit compression container의 전체 Fashion-MNIST test-set overflow
   비율을 별도 측정하고, 용량 정책을 고정한 뒤 Stage C 표본 수를 늘린다.
3. AWGN에서 후보 설정을 고른 후 perfect/imperfect-CSI fading 비교를 확장한다.

## 4단계: lossless neural compression 비교군

- 복구 자원: `compression_baseline/results/pixelcnn_fmnist.pt`, gated PixelCNN
  12 layer, 72 channel, kernel 7, 총 7,312,504 parameters.
- 복원 소스: `compression_baseline/pixelcnn.py`,
  `compression_baseline/arithmetic_coder.py`, `compression_baseline/codec.py`.
- legacy bytecode API/disassembly와 checkpoint 구조를 바탕으로 소스를
  복원했다. 과거 batched archive stream은 GPU batch shape에 따라 PMF가
  16-bit 양자화 경계를 넘는 현상이 있어 canonical bitstream으로 재사용하지
  않는다.
- encoder와 decoder 모두 probability model을 이미지별 batch size 1로
  호출하도록 고정했다. 서로 다른 프로세스와 서로 다른 batch partition에서도
  arithmetic stream round-trip이 bit-exact임을 검증했다.
- variable-length stream을 고정 4,873-bit container에 zero pad하고 컨테이너
  전체에 공통 CRC-16을 붙여 `K_ldpc=4,889`, `N=12,600`으로 전송한다.
- stream 길이는 별도 전송하지 않는다. decoder는 정확히 784개 symbol만
  복원하므로 trailing zero padding을 무시한다.
- CRC 실패는 outage로 선언하고 entropy decode를 호출하지 않는다. CRC 통과
  블록만 decode하며 container error, undetected error, corrupted output,
  exact decompression을 분리 계측한다.
- 12 dB 성공 smoke, -8 dB CRC-outage smoke, -4 dB 8-block mixed
  pass/outage smoke, 12 dB 전체 7-arm 동시 실행을 모두 통과했다. mixed
  smoke에서는 CRC-pass 1개만 entropy decode되어 bit-exact로 복원됐고,
  CRC-fail 7개에는 decoder를 호출하지 않았다. 이 작은 표본으로 성능
  우위를 주장하지 않는다.

## 5단계: alpha schedule 및 BCJR 실행 최적화

- BCJR의 frame별 순차 계산을 batch-vectorized alpha/beta recursion으로
  교체하고, 기존 scalar reference와 log-MAP/max-log, puncturing 조건에서
  `1e-12` 오차 이내 일치를 검증했다.
- 16-frame log-MAP 1-pass는 약 51.5초에서 약 0.35초로 단축됐다.
- paired coarse/fine search 뒤 독립 seed 2개로 후보를 재검증했다.
- canonical 기본값은 `outer_iterations=2`,
  `alpha_schedule=(0.1, 0.1)`, exact log-MAP이다.
- 두 독립 log-MAP 검증을 합치면 `(0.1, 0.1)`은 102/144 block error와
  458 bit error, `(0.125, 0.1)`은 101/144와 469, alpha=0은 117/144와
  1,163이었다. 두 nonzero 후보의 BLER 차이는 선택을 정당화할 수준이
  아니므로 더 단순하고 bit error가 적은 `(0.1, 0.1)`을 채택했다.
- max-log는 이 표본에서 약 20% 빠르고 성능 저하는 관찰되지 않았지만,
  작은 표본 결과이므로 선택 가능한 throughput 모드로만 유지한다.
- 세부 결과와 재현 명령은 `NO_LDPC_ALPHA_SCHEDULE_REPORT_KO.md`에 기록했다.

## 6단계: 실제 knee 기준 동작점 재조정

- canonical 설정을 64 block으로 스캔해 Es/N0 3.0 dB에서 BLER 5/64
  (7.81%)를 확인하고 모든 후속 튜닝 지점을 3.0 dB로 고정했다.
- `sigma_post={3,6,12}` 결과는 각각 3/64, 5/64, 5/64 block error여서
  `sigma_post=3`을 유지했다.
- final-pass BCJR APP의 `|LLR|` p99가 약 43.2, p99.9가 약 49.4였다.
  `llr_clip=30`은 source extrinsic의 41.3%를 잘랐고, 45는 17.7%를
  자르면서 BCJR APP 기준 초과율이 0.56%였다. 60은 APP 기준 사실상
  발동하지 않았고 성능도 나빠 45를 후보값으로 선택했다.
- 64-block `outer={2,4,6,8} x alpha={0.02,0.05,0.1,0.15}` 격자에서
  alpha 0.1이 모든 outer에서 2/64로 최상이었다. 128-block 상위 후보
  검증에서는 outer 4/6/8이 모두 6/128로 같아 최소 계산량인 outer 4를
  관측상 최상 후보로 정했다.
- 독립 seed 2개 합계에서 후보 `(3,45,4,0.1)`은 11/256,
  기존 canonical `(3,30,2,0.1)`은 13/256이었다. 각각의 95% Wilson CI는
  `[0.0242,0.0753]`, `[0.0299,0.0849]`로 겹쳤다.
- 따라서 개선은 통계적으로 확인되지 않았다. 후보는 runtime도 약 2배라
  CI-first 원칙에 따라 canonical 기본값을 변경하지 않았다.
- 장시간 GPU 실험에서 두 번째로 재현된 CRC의 희귀 `bool` callable
  실패는 명시적 0 비교와 scalar `.item()` 반환으로 우회했다. CRC 수식과
  known vector는 변하지 않는다.
- 실험 runner에 Wilson CI와 BCJR APP/source-extrinsic clip 진단을 추가했다.
  전체 표와 명령은 `NO_LDPC_OPERATING_POINT_RECALIBRATION_KO.md`에 있다.

## 7단계: practical_sigma 5G-LDPC와 공통 자원 waterfall

- `practical_sigma`를 별도 worktree에 체크아웃하고 A/B 코드는 수정하지 않은
  채 독립 프로세스에서 호출했다. worktree는 실험 후에도 clean이다.
- payload 6,272, N=12,600, payload rate 0.49778, 명시적 BPSK/AWGN Es/N0,
  `log P(1)/P(0)` 양수=bit 1을 세 arm에 공통 적용했다.
- CRC pass를 주 BLER 성공 조건으로 사용했다. A/B는 CRC24A, C는 CRC-16이며
  true payload BLER와 CRC BLER을 별도 저장했다. 측정된 undetected error는
  전 arm/전 최종 점에서 0이었다.
- CRC-BLER 0.1/0.01 knee는 A `-2.351/-2.240 dB`, B
  `-2.906/-2.661 dB`, C `+2.630/+3.662 dB`였다.
- C는 A보다 4.98/5.90 dB, B보다 5.54/6.32 dB 불리했다. 따라서 공통 자원
  최종 성능 기준으로는 RSC/BCJR 경로가 두 LDPC arm에 모두 패했고 LDPC
  경로 복귀가 합리적이라는 판정을 내렸다.
- C knee 2.6 dB의 alpha sweep에서 CRC-BLER는 alpha 0.1의 9.57%에서
  0.2의 77.34%, 0.3 이상의 100%로 붕괴했다. gentle source injection
  제약이 네 번째 구조에서도 재현됐다.
- 256-block 보조 분해에서 2.6 dB의 RSC-only와 RSC+SPC는 모두 2/256,
  score+SPC는 24/256이었다. 이 점에서는 score 주입이 유의하게 해로웠다.
- runner/표/플롯과 상세 판단은 `THREE_ARM_WATERFALL_REPORT_KO.md`에 있다.
