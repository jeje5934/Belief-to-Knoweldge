# HARQ source 보정 사전 진단: BP-20 실패 블록 구성

## 판정

제안한 `BP-20 실패 -> source 보정 1회 -> BP-20 재전파` 본 실험은 **중단을
권고**한다. 올바른 hard-decision CRC 기준에서 네 동작점 모두
`valid-but-wrong(syndrome=0, CRC fail)`이 0개였고, BP-20 실패는 같은 BP
message state를 총 40 iteration까지 연장하는 것만으로 100% 구제됐다.

사용자가 정의한 source-only-recoverable proxy,

`BP40 미구제 valid-but-wrong + BP40 미구제 far-from-convergence`,

는 네 SNR 모두 `0%`다. 진행 기준 30%에 못 미치는 정도가 아니라 중단 기준
10%보다도 명확히 낮다.

## 선행 발견: CRC 입력 규약 오류

최종 수치는 Sionna `CRCDecoder`에 0/1 hard bits를 전달해 측정했다. 진단 초안
도중 soft logits를 직접 전달했을 때 `syndrome=0 & CRC fail`이 대량으로 보였지만,
그 블록들의 payload와 CRC bits가 모두 정답인 모순이 발생했다. 설치된 Sionna
구현을 확인한 결과 `CRCDecoder`는 입력을 binary codeword로 간주하며, 내부
`CRCEncoder`는 입력을 `int64`로 절삭한 뒤 modulo-2를 취한다. 따라서 soft logits
직접 입력은 CRC 검사가 아니다.

현재 진단은 저장소 규약인 `logit > 0 -> bit 1`로 hard decision한 뒤 CRC를
검사한다. 고SNR 64-block smoke test에서 64/64 통과, payload undetected error 0을
확인했다.

이 오류는 읽기 전용 감사에서 최소한 다음 최근 경로에 확인됐다.

- `latency_baseline_sweep.py`: `hard_out=False` logits를 CRCDecoder에 직접 전달
- `codec_budget_waterfall.py`: 동일
- `low_budget_source_study.py` / `altproj_compression_study.py`: 기록한 soft
  info history와 final logits를 직접 전달하는 경로

따라서 직전 fixed-latency 그래프와 그것이 재사용한 저예산 codec/ours CRC-BLER은
hard-CRC로 재측정하기 전까지 **무효 보류**가 필요하다. 요청 순서상 fixed-latency
세 커밋은 이 발견 전에 이미 push됐으며, 이번 작업에서 되돌리거나 결과를 조용히
덮어쓰지 않았다.

## 실험 조건

- raw Fashion-MNIST payload 6272 bits, CRC24A, Sionna 5G LDPC, N=12600
- BPSK/AWGN + perfect CSI
- BP: boxplus-phi, flooding, warm BP state, `llr_max=30`
- BP-20: `[5]x4`, 매 5 iteration마다 hard CRC early-stop
- BP-40 대조: BP-20에서 실패한 동일 블록의 동일 `msg_v2c` state를
  `[5]x4` 더 연장, 매 5 iteration마다 hard CRC early-stop
- syndrome: rate recovery 후 full pruned LDPC graph posterior에서 계산
- 오류 수: 6272-bit payload hard decision 기준
- PSNR: MSB-first/raster로 복원한 28x28 uint8 이미지와 원본 비교
- SNR 간 동일 stateless unit noise, seed `20261001`
- production `decoder.py` 무수정, source/denoiser 호출 없음

## 1. BP-20 동작점

512-block 거친/정밀 스캔으로 네 목표점을 선정했다.

| 목표 BLER | Es/N0 | BP-20 실패/512 | BLER [Wilson 95%] |
|---:|---:|---:|---:|
| 0.5 | -2.05 dB | 271 | .529 [.486,.572] |
| 0.3 | -1.99 dB | 155 | .303 [.265,.344] |
| 0.1 | -1.92 dB | 52 | .1016 [.0783,.1308] |
| 0.05 | -1.88 dB | 27 | .0527 [.0365,.0756] |

최종 1024-block에서도 목표가 유지됐다.

| Es/N0 | BP-20 실패/1024 | BLER [Wilson 95%] | CRC undetected payload error |
|---:|---:|---:|---:|
| -2.05 | 539 | .526 [.496,.557] | 0 |
| -1.99 | 316 | .309 [.281,.338] | 0 |
| -1.92 | 111 | .108 [.0908,.1289] | 0 |
| -1.88 | 59 | .0576 [.0449,.0736] | 0 |

## 2. BP-20 실패 구성

| Es/N0 | 실패 수 | valid-but-wrong | near (syn 1~50) | far (syn >50) |
|---:|---:|---:|---:|---:|
| -2.05 | 539 | **0 (0%)** | 183 (34.0%) | 356 (66.0%) |
| -1.99 | 316 | **0 (0%)** | 154 (48.7%) | 162 (51.3%) |
| -1.92 | 111 | **0 (0%)** | 68 (61.3%) | 43 (38.7%) |
| -1.88 | 59 | **0 (0%)** | 38 (64.4%) | 21 (35.6%) |

valid-but-wrong 비율 0의 Wilson 95% 상한은 SNR 순서대로
`0.71%, 1.20%, 3.35%, 6.11%`다. 표본이 가장 작은 BLER 0.05 지점에서도
10% 중단 경계 아래다.

### 오류 비트 수와 이미지 PSNR

표의 오류 수는 `평균 / 중앙값 / 최대`, PSNR은 `평균 / 중앙값 / 범위`다.

| Es/N0 | 범주 | payload 오류 bits | 이미지 PSNR dB |
|---:|---|---:|---:|
| -2.05 | near | 3.22 / 2 / 15 | 46.92 / 40.95 / 29.97~77.07 |
|  | far | 46.40 / 27 / 311 | 28.95 / 28.26 / 17.39~64.06 |
| -1.99 | near | 3.28 / 2 / 13 | 46.60 / 40.95 / 30.11~77.07 |
|  | far | 32.64 / 18.5 / 202 | 30.74 / 30.24 / 19.31~52.65 |
| -1.92 | near | 2.72 / 2 / 15 | 48.10 / 45.84 / 30.15~77.07 |
|  | far | 23.21 / 20 / 85 | 31.26 / 29.17 / 23.36~63.09 |
| -1.88 | near | 2.71 / 2 / 8 | 44.75 / 40.94 / 29.81~77.07 |
|  | far | 14.10 / 13 / 48 | 32.88 / 31.14 / 24.69~58.75 |

모든 BP-20 CRC 실패 블록은 payload 오류가 최소 1개 있었으므로 image-domain에서
source가 볼 수 있는 손상은 존재한다. near 군은 중앙값 2 bits로 매우 미세하고,
far 군만 상대적으로 명확한 이미지 손상을 갖는다. 그러나 다음 절처럼 양쪽 모두
추가 BP만으로 전부 풀려 source 고유 기회가 되지 않는다.

## 3. 동일 state BP-40 연장

| Es/N0 | valid 구제 | near 구제 | far 구제 | 전체 구제 | BP-40 BLER [95%] |
|---:|---:|---:|---:|---:|---:|
| -2.05 | 0/0 | 183/183 | 356/356 | **539/539** | 0/1024 [0,.00374] |
| -1.99 | 0/0 | 154/154 | 162/162 | **316/316** | 0/1024 [0,.00374] |
| -1.92 | 0/0 | 68/68 | 43/43 | **111/111** | 0/1024 [0,.00374] |
| -1.88 | 0/0 | 38/38 | 21/21 | **59/59** | 0/1024 [0,.00374] |

최초 CRC 통과 iteration 분포:

| Es/N0 | iter 25 | iter 30 | iter 35 | iter 40 |
|---:|---:|---:|---:|---:|
| -2.05 | 451 | 73 | 13 | 2 |
| -1.99 | 287 | 28 | 1 | 0 |
| -1.92 | 109 | 2 | 0 | 0 |
| -1.88 | 58 | 1 | 0 | 0 |

즉 대부분은 추가 5 iteration만으로 해결되고, 가장 나쁜 동작점에서도 추가 20회
안에 전부 해결된다. BP-20에서 syndrome>50이었던 블록조차 BP-40으로 100%
구제됐으므로 `far = source가 아니면 못 고침`이라는 해석도 성립하지 않는다.

## 4. source-only-recoverable 판정

| Es/N0 | BP40 미구제 valid | BP40 미구제 far | source-only proxy |
|---:|---:|---:|---:|
| -2.05 | 0 | 0 | **0/539 = 0%** |
| -1.99 | 0 | 0 | **0/316 = 0%** |
| -1.92 | 0 | 0 | **0/111 = 0%** |
| -1.88 | 0 | 0 | **0/59 = 0%** |

### 권고

본 source 보정 성능 실험은 중단하는 것이 합리적이다.

1. conventional BP-20 실패 중 source만 판별할 수 있는 valid codeword 경쟁이
   관측되지 않았다.
2. 실패는 전부 미수렴이며 BP 연장 baseline이 100% 구제했다.
3. source 보정은 denoiser 1회라는 추가 latency/FLOPs를 지불하면서 BP-40의
   100% 구제를 넘어야 하므로 구조적 여지가 없다.
4. 과거 altproj의 `syn=0 != 정답` 근거는 soft-logit CRC misuse 영향을 먼저
   재감사해야 하며, 현재 HARQ 프레임의 근거로 사용할 수 없다.

## 산출물과 제안 커밋

- 진단 스크립트: `bp20_failure_diagnostic.py`
- 최종 원자료: `results/bp20_failure_composition_hardcrc_1024.json`
- 동작점 스캔: `results/bp20_failure_scan_hardcrc_512.json`,
  `results/bp20_failure_scan_hardcrc_refine_512.json`

이번 작업은 아직 커밋하지 않았다. 다음 두 단계가 필요하다.

1. 현재 계측/보고서: `diagnostics: classify hard-CRC BP20 failures before HARQ`
2. 별도 긴급 교정: `fix: hard-decision CRC checks in latency and codec diagnostics`
   이후 fixed-latency/codec 결과를 hard-CRC로 재측정하고 기존 보고서를 정정

두 번째는 과거 결과 재계산을 동반하므로 이번 계측 커밋과 분리해야 한다.
