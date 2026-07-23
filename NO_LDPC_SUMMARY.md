# no-LDPC 갈래 최종 요약

> **종료 상태:** 이 문서는 no-LDPC 갈래의 단일 진입점이자 최종 판정의
> 기준 문서다. 상세 보고서는 재현 명령, 원자료, 단계별 근거를 보존하는
> 동결 기록이며, 결론이 충돌할 경우 이 문서를 우선한다.

## 문서 지도

- [`NO_LDPC_DESIGN.md`](NO_LDPC_DESIGN.md): 최종 설계, 자원 정합,
  메시지 규약과 의사결정 기록.
- [`NO_LDPC_IMPLEMENTATION_REPORT_KO.md`](NO_LDPC_IMPLEMENTATION_REPORT_KO.md):
  구현 단계, 파일 구성, 테스트와 smoke 검증.
- [`NO_LDPC_ALPHA_SCHEDULE_REPORT_KO.md`](NO_LDPC_ALPHA_SCHEDULE_REPORT_KO.md):
  초기 outer-pass/alpha 탐색과 BCJR 벡터화.
- [`NO_LDPC_OPERATING_POINT_RECALIBRATION_KO.md`](NO_LDPC_OPERATING_POINT_RECALIBRATION_KO.md):
  실제 knee 재탐색, `sigma_post`/clip/outer/alpha 격자와 Wilson CI 판정.
- [`THREE_ARM_WATERFALL_REPORT_KO.md`](THREE_ARM_WATERFALL_REPORT_KO.md):
  최종 3-arm 원자료, CI, knee 보간, alpha sweep와 보조 arm.
- [`RSC_SOURCE_WIRING_AUDIT_KO.md`](RSC_SOURCE_WIRING_AUDIT_KO.md):
  배선 감사, source 유효 조건, bit-plane/echo 진단과 최종 block 회계.
- [`archive/CODEX_HANDOFF_NO_LDPC.md`](archive/CODEX_HANDOFF_NO_LDPC.md):
  구현 전 원래 작업 명세. 완료된 handoff이므로 참고용으로 보관한다.

## 최종 판정

동일한 Fashion-MNIST payload `6,272 bit`, 전송 길이 `N=12,600`,
payload rate `0.4977778`, 명시적 BPSK Es/N0, AWGN + perfect CSI
조건에서 C는 conventional 기준선 A와 기존 최강 B를 모두 이기지
못했다.

| Arm | CRC-BLER `1e-1` knee | CRC-BLER `1e-2` knee |
| --- | ---: | ---: |
| A. 5G LDPC only, BP-100 | `-2.351 dB` | `-2.240 dB` |
| B. 5G LDPC + score, legacy `[5]x20` | **`-2.906 dB`** | **`-2.661 dB`** |
| C. RSC/BCJR + score + SPC | `+2.630 dB` | `+3.662 dB` |

`1e-2`에서 C는 A보다 **5.90 dB**, B보다 **6.32 dB** 더 필요하다.
CI 폭보다 훨씬 큰 곡선 간 분리다. 따라서 **no-LDPC 갈래를 종료하고
5G LDPC 경로로 복귀한다. 최종 성능 최선은 B**다.

## 시스템과 자원 정합

### 공통 규약

- Source: Fashion-MNIST `28 x 28`, `uint8`, raster 순서, 픽셀당
  MSB-first 8 bit.
- Payload: `K=28*28*8=6,272 bit`.
- Channel use: `N=12,600`; 공통 payload rate
  `6,272/12,600=0.4977778`.
- Modulation/channel: real BPSK, `0 -> -1`, `1 -> +1`, AWGN,
  perfect CSI.
- LLR: `log P(1)/P(0)`이므로 양수가 bit 1.
- Es/N0: `N0=10^(-EsN0/10)`을 세 arm에 동일하게 적용.
- 가능한 공통 점에서 `(seed, Es/N0)`별 payload index와 길이 12,600의
  표준 정규 noise를 arm 간 paired로 사용.
- 주 block 지표는 CRC failure이며 true payload BLER, BER,
  undetected error를 별도 계측.

A/B는 CRC24A와 Sionna 5G LDPC를 사용해 channel information
길이가 6,296이고, C는 CRC-16-CCITT-FALSE를 사용한다. CRC 길이 차이로
C의 이론적 undetected error 확률이 더 높지만(`2^-16` 대 `2^-24`),
최종 waterfall에서 C의 undetected error는 0이었다. payload와 `N`이
같으므로 별도 rate 보정 없이 직접 비교했다.

### C의 정확한 차원

| 단계 | 길이 |
| --- | ---: |
| 원본 payload | 6,272 |
| 픽셀별 SPC parity (`M=8`, 784개) 삽입 후 | 7,056 |
| CRC-16 추가 후 RSC information | 7,072 |
| memory-3 RSC tail | 3 |
| systematic trellis symbols | 7,075 |
| 유지한 parity symbols | 5,525 |
| puncture한 parity symbols | 1,550 |
| 최종 전송 길이 | **12,600** |

No-SPC arm은 `6,272+16+3=6,291` trellis input의 unpunctured
rate-1/2 출력이 12,582로 18 symbol 부족하다. 이 보조 arm에서만
균일한 parity 위치 18개를 반복하고 수신 LLR을 합산했다. Full C는
parity puncturing만 사용한다.

### 확정된 C 구성

```text
outer_iterations = 2
alpha_schedule = (0.1, 0.1)
sigma = 0.3
sigma_post = 3
llr_clip = 30
bcjr_mode = exact log-MAP
interleaver_seed = 20260722
```

Classic turbo 메시지
`L_C_to_S = L_APP - L_A`,
`L_S_to_C = L_source_posterior - Q`를 유지했다. Max-log는 throughput
선택지로만 보존했다. 실제 `3.0 dB` knee에서
`sigma_post=3, llr_clip=45, outer=4, alpha=0.1`이 관측상 최상
후보였지만, 독립 seed 합산 256 block에서 후보 `11/256`
(Wilson 95% CI `0.0242--0.0753`)와 canonical `13/256`
(`0.0299--0.0849`)의 CI가 겹쳤고 runtime이 약 2배였다. 따라서
canonical을 바꾸지 않았다.

## 최종 3-arm waterfall

괄호는 CRC-BLER의 Wilson 95% CI다. `true`는 CRC와 무관한 payload
block error다.

### A. 5G LDPC only, BP-100

| Es/N0 | Blocks | CRC-BLER (95% CI) | True errors | BER |
| ---: | ---: | ---: | ---: | ---: |
| -2.40 | 512 | 87/512 = 0.1699 (0.1399--0.2049) | 84 | 1.294e-2 |
| -2.35 | 1,024 | 101/1,024 = 0.0986 (0.0818--0.1184) | 99 | 6.802e-3 |
| -2.30 | 1,024 | 41/1,024 = 0.0400 (0.0297--0.0539) | 40 | 2.763e-3 |
| -2.25 | 1,024 | 12/1,024 = 0.0117 (0.0067--0.0204) | 12 | 8.688e-4 |
| -2.20 | 3,200 | 17/3,200 = 0.00531 (0.00332--0.00849) | 17 | 4.308e-4 |
| -2.15 | 3,200 | 2/3,200 = 0.000625 (0.000171--0.002276) | 2 | 7.623e-5 |

### B. 5G LDPC + score, legacy `[5]x20`, `alpha=beta=0.1`

| Es/N0 | Blocks | CRC-BLER (95% CI) | True errors | BER |
| ---: | ---: | ---: | ---: | ---: |
| -3.00 | 512 | 107/512 = 0.2090 (0.1760--0.2463) | 105 | 1.238e-2 |
| -2.90 | 1,024 | 98/1,024 = 0.0957 (0.0792--0.1153) | 95 | 5.066e-3 |
| -2.80 | 1,024 | 32/1,024 = 0.0313 (0.0222--0.0438) | 32 | 1.664e-3 |
| -2.75 | 1,024 | 29/1,024 = 0.0283 (0.0198--0.0404) | 29 | 1.560e-3 |
| -2.70 | 3,200 | 50/3,200 = 0.0156 (0.0119--0.0205) | 46 | 7.438e-4 |
| -2.60 | 3,200 | 16/3,200 = 0.00500 (0.00308--0.00811) | 13 | 1.830e-4 |
| -2.50 | 3,200 | 1/3,200 = 0.000313 (0.000055--0.001768) | 1 | 6.427e-6 |

### C. RSC/BCJR + score + SPC, canonical

| Es/N0 | Blocks | CRC-BLER (95% CI) | True errors | BER |
| ---: | ---: | ---: | ---: | ---: |
| +2.40 | 512 | 98/512 = 0.1914 (0.1597--0.2277) | 98 | 7.536e-5 |
| +2.60 | 1,024 | 110/1,024 = 0.1074 (0.0899--0.1279) | 110 | 4.360e-5 |
| +3.00 | 1,024 | 42/1,024 = 0.0410 (0.0305--0.0550) | 42 | 1.479e-5 |
| +3.40 | 1,024 | 23/1,024 = 0.0225 (0.0150--0.0335) | 23 | 9.186e-6 |
| +3.60 | 3,200 | 34/3,200 = 0.0106 (0.00761--0.0148) | 34 | 3.836e-6 |
| +3.80 | 3,200 | 28/3,200 = 0.00875 (0.00606--0.0126) | 28 | 3.637e-6 |
| +4.00 | 3,200 | 10/3,200 = 0.00313 (0.00170--0.00574) | 10 | 1.196e-6 |
| +4.20 | 3,200 | 8/3,200 = 0.00250 (0.00127--0.00493) | 8 | 9.965e-7 |
| +4.50 | 3,200 | 1/3,200 = 0.000313 (0.000055--0.001768) | 1 | 9.965e-8 |

직접 anchor에서도 같은 결론이다. `-2.5 dB`에서 C는 `128/128`,
A는 `59/128`, B는 `1/3,200` 실패했다. `+2.6 dB`에서 A/B는 각각
`0/128`이고 C는 `110/1,024`라 Wilson CI도 분리된다.

## 원인 확정

### 1. Code strength와 source 기회의 구조

성능 차이의 지배 요인은 code strength다. 약한 RSC는 유효 동작점을
약 `+2.6 dB`로 밀어 올린다. 그 지점에서 first-BCJR cavity는 bit 기준
약 `99.99%` 정확하고, 실패 block도 평균 오류 2.93 bit,
평균 PSNR 41.22 dB라 image prior가 볼 수 있는 훼손이 거의 없다.

Source는 cavity가 틀린 위치에서는 약 76% 맞지만, 그런 위치 자체가
너무 적다. Hard-decision disagreement 기준 교정:오염 기회는 RSC
`1:3,675`, LDPC `1:7.82`다. Source의 가치는 절대 SNR이나 marginal
정확도 하나가 아니라 channel decoder의 잔여 BER, 오류의 image-space
가시성, 교정 기회와 오염 기회의 비로 정해진다.

### 2. 최종 block 단위 직접 원인

`+2.6 dB`, seed `20260730`, 동일한 256 payload/noise에서 full
score-off + SPC와 waterfall 당시의 unbounded canonical score-on +
SPC를 paired 비교했다.

| 최종 block 전이 | 수 |
| --- | ---: |
| Score-off 성공 -> score-on 실패 | **22** |
| Score-off 실패 -> score-on 성공 | **0** |
| 둘 다 실패 | 2 |
| 둘 다 성공 | 232 |

따라서 `22 broken - 0 rescued = +22`이며, score-off `2/256`에서
score-on `24/256`으로 증가한 값과 정확히 일치한다. Total payload bit
error도 `6 -> 67`로 늘었다. **Source는 평균/중간 단계에서 일부 bit를
교정해도 최종적으로 성공 block 22개를 파괴하고 하나도 구제하지
못했으며, 이 소수 block의 파괴가 BLER을 지배했다.**

### `14/64` 해석의 명시적 무효화

과거 `14/64`는 outer iteration 1의 **source/SPC 주입 전 first-BCJR
extrinsic payload hard decision**이다. `2/256`은 두 outer pass와
`IndependentBitCategoricalProvider + SourceSPCSISO`를 모두 거친
**score-off + SPC의 최종 출력**이다. 둘은 seed와 block failure
판정은 같고 64 block은 256 paired plan의 앞부분이지만, decoder
단계와 source factor가 다르다.

따라서 **`14/64`를 score-off BLER 기준으로 사용한 이전 해석은
무효**다. 이 tensor는 wiring, LLR, bit-plane, local update 진단에는
유효하지만 최종 score-off 성능이나 최종 BLER 원인의 근거로 사용할
수 없다.

## 기각된 가설

- Payload/SPC/CRC 위치, interleaver 방향, LLR 부호, MSB-first/raster
  indexing 등 배선 오류.
- 다른 denoiser/checkpoint 사용 또는 systematic likelihood의 명시적
  이중 계산.
- Source categorical marginal의 bound 불일치가 주원인이라는 가설.
  불일치는 실제 bug였지만 수정 후 `24/256 -> 28/256`으로 회복되지
  않았다.
- `llr_clip=30`이 상시 발동해 생긴 문제.
- 첫 damped source 주입의 순오염. 첫 local update의 실제
  correction:contamination sign-flip 비는 RSC에서 오히려 `14:1`이었다.
- Denoiser cavity echo 또는 그로 인한 extrinsic double counting.
- 첫 주입의 새 오류가 trellis event를 일으켜 두 번째 BCJR에서 net
  bit-error가 폭발한다는 가설. 해당 BCJR의 bit error는 net 감소했다.

## LDPC 복귀 후 재사용할 계측

### Source posterior의 bit-plane 정합률

| 경로 | b7 (MSB) | b6 | b5 | b4 | b3 | b2 | b1 | b0 (LSB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RSC `+2.6 dB` | 95.57% | 86.92% | 79.39% | 75.84% | 74.01% | 66.18% | 54.04% | 41.01% |
| LDPC `-2.5 dB` | 94.93% | 81.37% | 68.29% | 67.73% | 66.64% | 63.50% | 68.00% | 73.14% |
| LDPC, cavity-wrong 조건부 | 95.83% | 81.95% | 61.12% | 62.34% | 59.24% | 55.96% | 61.22% | 69.78% |

RSC는 MSB에서 LSB로 대체로 감소해 exact 하위 pixel bit를 source가
알지 못함을 보여 준다. LDPC의 U자형은 입력 masking과 output/input
gain 계측으로 검사했으며, LSB posterior 정합이 단순 cavity echo라는
가설은 기각됐다.

### 재사용할 판정 프레임워크

Source prior 평가는 다음을 함께 봐야 한다.

1. Cavity가 틀린 위치에서의 조건부 source 정확도.
2. `cavity wrong/source correct` 교정 기회와
   `cavity correct/source wrong` 오염 기회.
3. 두 위치 집단의 cavity `|LLR|`와 실제 damped 주입 후 sign flip.
4. 실패 block의 오류 bit 수, PSNR/SSIM과 bit-plane 분포.
5. 최종 paired block의 성공->실패와 실패->성공 전이.

핵심 유효 조건은 **channel decoder의 잔여 오류가 image 공간에서
source가 볼 수 있을 만큼 조밀하고, 실제 교정이 오염을 상쇄하는가**다.
LDPC 실패 block은 RSC 실패 block보다 평균 오류 bit가 약 182배 많고
평균 PSNR이 25.9 dB 낮아 이 조건에 훨씬 가깝다.

## 미해결 항목과 종료 범위

LDPC purity 실험의 `minus < legacy (D<C)` 메커니즘은 열려 있다.
Cavity echo는 직접 기각돼 이 현상을 설명하지 못한다. 이 문제는 LDPC
경로에서 별도로 다룰 대상이며, 종료된 no-LDPC 갈래에서는 추가
탐색하지 않는다.

원래 구현 지시서 `CODEX_HANDOFF_NO_LDPC.md`는 결론과 실행 결과가
이 문서 및 상세 보고서에 모두 흡수됐다. 활성 지시로 오인하지 않도록
삭제하지 않고 `archive/`로 이동해 역사적 명세와 재현 맥락을
보존한다.
