# RSC source 경로 배선 및 수치 인터페이스 감사

## 종료 판정

**No-LDPC 갈래를 종료하고 5G LDPC 경로로 복귀한다.** 동일 payload
6,272 bit, 동일 `N=12,600`, 동일 Es/N0에서 C(RSC/BCJR + score + SPC)는
CRC-BLER `1e-2` knee가 `+3.662 dB`로, A(5G LDPC only, `-2.240 dB`)보다
**5.90 dB**, B(5G LDPC + score, `-2.661 dB`)보다 **6.32 dB** 뒤진다.
최종 성능 최선은 B다.

### 원인

전체 성능 차이의 지배 요인은 code strength다. RSC는 약한 code 때문에
실용 동작점이 약 `+2.6 dB`로 밀리고, 그 지점의 first-BCJR cavity는
bit 기준 약 `99.99%` 정확해 source가 기여할 잔여 불확실성이 거의 없다.
Hard disagreement 기준 교정:오염 기회는 RSC `1:3,675`, LDPC
`1:7.82`다. Source 유효성은 절대 SNR이 아니라 channel decoder의
잔여 BER과 image prior가 교정할 수 있는 오류 밀도로 결정된다.

### 최종 block 회계

과거 `14/64` first-BCJR 진단과 `2/256` score-off 실측은 같은 값이
아니었다.

- `14/64`: outer iteration 1의 **source/SPC 주입 전** BCJR extrinsic
  payload를 hard-decision한 component 진단.
- `2/256`: 두 outer pass와
  `IndependentBitCategoricalProvider + SourceSPCSISO`를 모두 거친
  **score-off + SPC 최종 출력**.
- 둘 다 true payload block error와 CRC-16 failure가 표본 내에서
  동일했으며 seed는 `20260730`으로 같다. 64 block은 256 paired plan의
  앞부분이다. 차이는 seed나 판정식이 아니라 decoder 단계와 source
  factor다.

따라서 `14/64`를 score-off BLER처럼 해석한 이전 비교는 무효다. 해당
first-pass tensor는 wiring, LLR, bit-plane, echo, local update 진단에는
유효하지만 최종 score-off 성능이나 `24 vs 2`의 block 원인을 설명하는
근거로는 사용할 수 없다.

조건을 완전히 정렬해 `+2.6 dB`, seed `20260730`, 동일 256
payload/noise에서 최종 score-off + SPC와 수정 전 waterfall에 사용한
unbounded canonical score-on + SPC를 paired 비교했다.

| 최종 block 상태 | 수 |
|---|---:|
| Score-off 성공 → score-on 실패 | **22** |
| Score-off 실패 → score-on 성공 | **0** |
| 둘 다 실패 | 2 |
| 둘 다 성공 | 232 |
| 순 BLER error-count 변화 | **+22** |

True payload와 CRC failure mask는 두 arm 모두 bit-exact하게 일치했다.
Score-off는 `2/256`, score-on은 `24/256`이고 bit error도 `6 -> 67`로
늘었다. 즉 실측의 `+22`가 `22 broken - 0 rescued = +22`와 정확히
일치한다. **최종 BLER 역전의 직접 원인은 source가 원래 성공하던
22개 block을 파괴하고 단 하나도 구제하지 못한 것**으로 확정한다.
First-pass local 총합에서 일부 bit가 교정돼 보인다는 사실은 최종
block 판정을 대표하지 못한다.

Marginal posterior bound 수정은 실제 interface bug를 고쳤지만 별도
paired 재측정에서 `24/256 -> 28/256`으로 개선되지 않았다. 위 exact
`24 vs 2` 회계는 당시 waterfall 조건을 맞추기 위해 수정 전 unbounded
provider를 명시적으로 재현했으며, bound 유무와 무관하게 score 경로가
해롭다는 판정은 같다.

### 기각된 가설

- payload/SPC/CRC 배선, LLR 부호, interleaver 방향, MSB/raster indexing 오류
- marginal bound 불일치가 주원인이라는 가설
  (불일치는 실제 bug였지만 성능 원인은 아님)
- `llr_clip=30` 상시 발동
- 첫 source 주입의 순오염
  (local 실효 교정:오염은 RSC에서 오히려 `14:1`)
- denoiser cavity echo 및 extrinsic double counting
- 두 번째 BCJR의 net bit-error 폭발

### LDPC 복귀 후 재사용할 계측

- Source posterior bit-plane 정합:
  LDPC `94.9 -> 63.5 -> 73.1%`의 U자형,
  RSC `95.6 -> 41.0%`의 대체로 감소하는 형상.
- Cavity 오류 위치의 조건부 source 정합과
  correction/contamination opportunity 회계.
- Source가 유효하려면 channel decoder의 잔여 오류가 image 공간에서
  보일 만큼 조밀해야 한다는 조건.
- Hard opportunity뿐 아니라 cavity `|LLR|`, 실제 sign flip, 최종
  block transition을 함께 보는 계측 절차.

### 미해결로 남기는 항목

LDPC purity 실험의 `minus < legacy (D<C)` 메커니즘은 여전히 열려 있다.
Cavity echo가 기각됐으므로 echo에 의한 유용한 자기강화로 설명할 수
없다. No-LDPC 종료 범위를 넘는 추가 탐색은 수행하지 않는다.

## 결론

`+2.6 dB`에서 관측된 score 경로 악화는 payload/SPC/CRC 인덱싱,
인터리버 방향, LLR 부호 또는 이미지 bit layout 오류 때문이 아니다.
Fashion-MNIST source 입력은 고SNR과 knee SNR에서 원본과 정렬된 정상
이미지였고 모든 round-trip/assert를 통과했다.

대신 LDPC wrapper와 no-LDPC categorical adapter 사이에서 실제 구현
불일치 하나를 발견했다. 동일 denoiser, checkpoint, systematic LLR에 대해
LDPC `SourcePriorDenoiser`는 bit probability를 `[1e-7, 1-1e-7]`로 제한한
뒤 posterior logit을 내보내지만, categorical adapter는 Gaussian pixel
PMF를 제한 없이 `logsumexp`하여 수백 LLR까지 내보냈다. 수정 전 두
extrinsic의 cosine은 `-0.099`, 평균 절대차는 `105.48`이었다.

동일 marginal probability bound를 categorical/SPC APP 뒤에 적용한 후
두 모듈의 extrinsic은 부호가 100% 일치하고 cosine `0.9999998`, 평균
절대차 `5.83e-4`로 정렬됐다. 그러나 동일 256 block 재측정은
`24/256 -> 28/256`으로 개선되지 않았다. 따라서 이 불일치는 수정
대상이지만 12배 역효과의 원인은 아니다.

## 실행 예산

조정 단계의 기본값은 다음으로 제한한다.

- 최초 paired screen: 32 block
- 살아남은 후보: 64--128 block
- 단일 확인점: 256 block
- 1,024--3,200 block: 최종 waterfall/결정 gate를 명시적으로 요청한 경우

이번 감사도 시각/배선 검증 4--16 block, 원인 분해 32--64 block,
수정 효과 확인 한 점만 256 block으로 수행했다.

## [1] 첫 outer pass 시각 검증

산출물:

- `results/source_wiring_diagnostic_fixed/source_first_pass_esn0_8dB.png`
- `results/source_wiring_diagnostic_fixed_16/source_first_pass_esn0_2.6dB.png`

`8 dB`에서는 source SISO hard 입력이 원본과 pixel 단위 100% 같고 soft
입력 MAE는 `2.29e-6/255`였다. `2.6 dB`, 16 block에서는 hard pixel
일치율 `99.976%`, soft 입력 MAE `0.026/255`였다. 두 PNG 모두 원본,
soft 입력, hard 입력이 같은 위치와 형태의 정상 Fashion-MNIST 이미지다.
denoiser 출력도 그럴듯한 동일 의류이지만 pixel MAE는 각각 `14.04`,
`10.41`로, 이미 거의 정확한 입력의 세부 grayscale 값을 오히려 바꾼다.

## [2] bit layout 및 부호 검증

다음 assert가 모두 통과했다.

- `deinterleave(interleave(x)) == x`
- encoded information의 앞 7,056 bit가 정확히 784개의
  `[8 systematic MSB-first bits, 1 SPC parity]`
- `decode_systematic(encode_spc(payload)) == payload`
- source provider에는 9-bit block 중 앞 8 bit만 `(784, 8)`로 전달
- 마지막 CRC-16 위치의 source feedback은 정확히 0
- payload는 `np.unpackbits`/`np.packbits(bitorder="big")`의 MSB-first
- pixel 순서는 Fashion-MNIST의 28x28 raster 순서
- `LLR > 0 -> bit 1`

첫 BCJR pass cavity hard-bit 정합률은 `2.6 dB`, 16 block에서
`99.997%`였다. 따라서 source 모듈이 받는 이미지는 parity/CRC가 섞인
scramble이나 반전 이미지가 아니다.

## [3] source 출력 방향과 feedback 위치

`SourceSPCSISO`가 반환한 extrinsic이 실제로
`posterior_llr - source_message`와 bit-exact하게 같음을 확인했다.
이 동일 tensor를 clip/damping한 뒤 CRC zero를 붙이고 forward
interleaver로 BCJR prior에 돌려보냈으며, 역변환 round-trip도 정확했다.

수정 후 첫 pass payload source-extrinsic 부호와 실제 bit 부호의
일치율은 다음과 같다.

| Es/N0 | blocks | cavity hard 정합 | source posterior 부호 정합 | source extrinsic 부호 정합 | payload clip 발동 |
|---:|---:|---:|---:|---:|---:|
| 8.0 dB | 4 | 100% | 69.26% | 0.00% | 100% |
| 2.6 dB | 16 | 99.997% | 75.06% | 2.24% | 19.17% |

핵심은 extrinsic의 이름이나 부호보다 입력 정보의 품질 차이다. 첫 BCJR
cavity는 사실상 정답인데 source posterior는 bit 기준으로 25--31%가
틀린다. 따라서 source prior는 새로운 정보가 아니라 더 부정확한 추정치를
거의 완벽한 cavity에 섞는 오염원이다. `posterior - cavity`가 반대
부호가 되는 현상은 이 품질 역전의 결과이지 별도의 원인으로 볼 필요가
없다.

## [4] LDPC 경로 직접 대조

`practical_sigma/denoiser.py`를 읽기 전용 import하고 같은 payload의 같은
첫-pass systematic LLR `(1, 6272)`를 두 source 모듈에 입력했다.
`source_prior.py`도 SHA-256이 동일했다.

| 비교 | 수정 전 | 수정 후 |
|---|---:|---:|
| extrinsic mean absolute difference | 105.484 | 5.83e-4 |
| extrinsic cosine | -0.09897 | 0.9999998 |
| extrinsic sign equality | 71.59% | 100% |
| posterior cosine | 0.66059 | 0.9999988 |

수정은 `ScorePriorCategoricalProvider`가 LDPC source posterior의 float32
probability floor를 선언하고 score 계열 `SourceCategoricalSISO` 및
`SourceSPCSISO`가 marginal APP에 그 범위를 적용하도록 했다.
Independent-bit SPC baseline의 exact APP에는 이 제한을 적용하지 않는다.

## 수정 후 paired 재측정

동일 seed `20260730`, 동일 256 payload/noise에서:

| 구현 | CRC/true block errors | BLER | Wilson 95% CI | bit errors |
|---|---:|---:|---:|---:|
| 수정 전 unbounded categorical APP | 24/256 | 0.09375 | [0.06381, 0.13570] | 67 |
| 수정 후 LDPC-compatible marginal bound | 28/256 | 0.10938 | [0.07676, 0.15354] | 80 |

CI는 크게 겹치며 개선은 없다. 이전 waterfall의 C 점은 수정 전 구현
결과이므로 다음 최종 비교가 필요할 때만 작은 screen부터 다시 생성한다.

## [5] 후속 원인 분해

### llr_clip

동일 64 block에서:

| llr_clip | errors/64 |
|---:|---:|
| 10 | 7 |
| 30 | 4 |
| 100 | 4 |

30과 100이 같고 10은 더 나빴다. 상시 clipping이 주원인이라는 증거는
없다.

### categorical + SPC 결합

32-block 탐색에서 no-SPC RSC는 score off/on이 각각 `2/32 -> 1/32`,
SPC RSC는 `0/32 -> 3/32`였다. 표본은 작아 성능 주장은 할 수 없지만,
score가 denoiser 단독보다 categorical/SPC 결합에서 문제를 만든다는
방향이다. 코드상 systematic cavity는 provider가 정확히 한 번만
소비하며 parity likelihood만 추가되므로 명시적 likelihood 이중계산은
발견되지 않았다.

### exact cavity 대 incomplete cavity

동일 64 block에서 SPC-only exact cavity `1/64`, score+SPC exact cavity
`4/64`, 이전 source prior를 포함한 legacy식 inclusive APP 입력
`5/64`였다. incomplete cavity가 개선하지 않았으므로 단순히
`L_APP-L_A`를 `L_APP`로 바꾸는 것으로 해결되지 않는다.

## [6] 저SNR source 가치 전환점 검증

가설은 source prior의 가치가 절대 SNR이 아니라 channel decoder의 잔여
불확실성으로 정해진다는 것이다. 직접 예측을 검증하기 위해 seed
`20260802`의 동일 payload/noise로 각 SNR에서 32 block을 측정했다.
살아남은 score-on 후보가 없었으므로 사전 실행 예산 원칙에 따라 64
block 확장은 하지 않았다.

| Es/N0 | 첫 BCJR cavity bit 정합 | source posterior bit 정합 | Score off BLER (Wilson 95%) | Score on BLER (Wilson 95%) |
|---:|---:|---:|---:|---:|
| +1.0 dB | 99.806% | 70.145% | 27/32 = 0.8438 `[0.6825, 0.9314]` | 30/32 = 0.9375 `[0.7985, 0.9827]` |
| +1.5 dB | 99.917% | 72.195% | 10/32 = 0.3125 `[0.1795, 0.4857]` | 18/32 = 0.5625 `[0.3933, 0.7183]` |
| +2.0 dB | 99.972% | 74.047% | 2/32 = 0.0625 `[0.0173, 0.2015]` | 6/32 = 0.1875 `[0.0889, 0.3531]` |

요청한 범위에서는 전환점이 없고 score-on은 세 점 모두 악화 방향이다.
중요한 관찰은 `+1.0 dB`에서 score-off BLER가 이미 84%인데도 cavity의
bit 정합률은 99.806%라는 점이다. 6,272-bit block에서는 약 0.2%의
잔여 BER만으로도 대부분의 block이 실패하므로 높은 BLER가 낮은
bit-정합률을 뜻하지 않는다. Source posterior의 70--74% 정합률과
교차하기 훨씬 전에 block 지표가 포화한다.

따라서 이 실험은 가설의 핵심을 지지하지만, 예상한 usable crossover는
부정한다. RSC waterfall 전 구간에서 channel cavity가 source posterior
보다 압도적으로 정확하며, source가 더 정확해지는 영역에 도달하기 전에
BLER가 1에 가까워진다. 현재 denoiser를 그대로 사용할 때 RSC 경로에서
source prior가 유효한 동작점은 관측되지 않았다.

## [7] 조건부 교정, 실패 이미지, bit-plane 및 LDPC 통제군

기존 first-BCJR/BP-100 출력과 frozen source module의 tensor를 그대로
읽는 계측 훅만 추가했다. Decoder update나 hard decision은 변경하지
않았다. RSC는 seed `20260730`으로 `+1.0/+2.0/+2.6 dB` 각 64 block,
LDPC는 `practical_sigma` read-only worktree에서 `-2.5 dB` 64 block을
측정했다.

### RSC 조건부 정확도

| Es/N0 | Cavity wrong bits | Cavity가 틀린 곳의 source 정합 | 유용한 교정 / 유해한 오염 | 교정:오염 | CRC 실패 block | 실패 block 오류 평균/중앙/최대 | 실패 이미지 PSNR 평균/중앙 | SSIM 평균 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +1.0 dB | 728 | 75.82% | 552 / 114,111 | 1:207 | 62/64 | 11.74 / 10 / 33 | 35.55 / 32.95 dB | 0.9682 |
| +2.0 dB | 105 | 76.19% | 80 / 113,218 | 1:1,415 | 29/64 | 3.62 / 3 / 10 | 43.70 / 40.87 dB | 0.9836 |
| +2.6 dB | 41 | 75.61% | 31 / 113,912 | 1:3,675 | 14/64 | 2.93 / 3 / 5 | 41.22 / 40.94 dB | 0.9918 |

첫 번째 예측인 “cavity가 틀린 곳에서 source 정합률이 50% 이하”는
성립하지 않았다. Source는 RSC가 틀린 위치에서는 약 76% 확률로 맞아
상보적 정보가 있다. 문제는 기회 수와 부작용의 비다. `+2.6 dB`에서
source가 실제 오류를 교정할 수 있는 disagreement는 31개뿐인데,
맞아 있던 cavity를 source가 반대로 주장하는 disagreement는
113,912개다. Hard posterior 기준 오염 기회가 교정 기회의 약
3,675배다.

이 count는 `alpha=0.1` damping 전 source posterior hard-decision의
진단값이므로 실제로 113,912 bit가 뒤집힌다는 의미는 아니다. 다만
RSC에서 source가 제공할 수 있는 유용한 수정은 극소수이고, 잘못된
방향으로 힘을 가할 위치는 압도적으로 많다는 기회 구조를 보여준다.

### RSC 실패 이미지와 오류 bit-plane

`+2.6 dB` first-BCJR CRC 실패 block은 평균 2.93 bit, 중앙값 3 bit만
틀렸고 PSNR 평균 41.22 dB, SSIM 평균 0.9918이었다. 대표 이미지는
`results/source_residual_images/rsc_failure_cavity_esn0_2.6dB.png`에
저장했다. 육안으로 원본과 cavity가 사실상 동일하고 error map에 몇
pixel만 나타난다.

실패 block의 cavity 오류 bit-plane count는 MSB부터
`[5, 4, 8, 5, 2, 3, 9, 5]`였다. 표본이 41 bit뿐이라 특정 plane 집중을
주장할 수 없지만, denoiser가 볼 수 있는 공간적 훼손이 거의 없다는
결론은 오류 수와 이미지 품질에서 직접 확인된다.

RSC source posterior의 전체 bit-plane 정합률은 다음과 같다.

| Es/N0 | b7(MSB) | b6 | b5 | b4 | b3 | b2 | b1 | b0(LSB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +1.0 dB | 94.46% | 85.97% | 77.09% | 75.10% | 73.45% | 65.69% | 56.25% | 44.22% |
| +2.0 dB | 95.16% | 85.54% | 78.50% | 75.95% | 75.36% | 68.46% | 55.39% | 39.95% |
| +2.6 dB | 95.57% | 86.92% | 79.39% | 75.84% | 74.01% | 66.18% | 54.04% | 41.01% |

예측대로 MSB는 약 95%지만 LSB는 40--44%다. Denoiser는 의류의
대략적인 밝기와 형상은 알지만 exact uint8 pixel의 하위 비트는 알지
못한다. Lossless bit recovery 관점에서 LSB source posterior는 순수
노이즈에 가깝다.

### LDPC BP-100 통제군

`practical_sigma`의 실제 CRC24A/5G-LDPC encoder와 BP-100 decoder를
호출하고, 그 pre-source payload posterior를 같은 worktree의 legacy
`SoftDenoiser`에 입력했다. Worktree 코드는 수정하지 않았다.

| 지표 | LDPC, -2.5 dB | RSC, +2.6 dB |
|---|---:|---:|
| Cavity bit 정합 | 95.23% | 99.9898% |
| Cavity wrong bits / 64 blocks | 19,161 | 41 |
| Cavity가 틀린 곳의 source 정합 | 68.42% | 75.61% |
| 유용한 교정 / 유해한 오염 | 13,110 / 102,530 | 31 / 113,912 |
| 교정:오염 | 1:7.82 | 1:3,675 |
| CRC 실패 blocks | 36/64 | 14/64 |
| 실패 block 오류 평균/중앙/최대 | 532.25 / 516 / 780 | 2.93 / 3 / 5 |
| 실패 이미지 PSNR 평균/중앙 | 15.35 / 15.02 dB | 41.22 / 40.94 dB |
| 실패 이미지 SSIM 평균 | 0.5636 | 0.9918 |

LDPC 실패 block은 RSC보다 평균 오류 bit가 약 182배 많고, PSNR은
25.9 dB 낮다. 대표 이미지는
`results/source_residual_images/ldpc_failure_cavity_esn0_-2.5dB.png`에
저장했으며 BP-100 cavity의 의류 형상이 육안으로 크게 훼손돼 있다.
LDPC 실패 block 오류는 bit-plane별
`[2445, 2327, 2361, 2440, 2461, 2416, 2352, 2359]`로 거의 균일하다.

LDPC source posterior bit-plane 정합률은
`[94.93, 81.37, 68.29, 67.73, 66.64, 63.50, 68.00, 73.14]%`였다.
RSC처럼 완전히 단조 감소하지는 않았지만, source가 exact bitstream보다
coarse image 정보를 주로 제공한다는 점은 동일하다.

### 사전 판정 기준에 대한 결론

제시된 세 기준 중 (a) LDPC 오류 bit 수가 훨씬 많음, (b) LDPC 이미지
PSNR이 훨씬 낮음은 강하게 성립했다. 그러나 (c) “LDPC에서 cavity 오류
위치의 source 조건부 정합률이 더 높음”은 반대로 나왔다
(`68.42% < 75.61%`). 따라서 사전의 세 조건을 문자 그대로 모두
충족하는 의미에서 설명이 완전 확증된 것은 아니다.

측정이 지지하는 수정된 설명은 다음과 같다. Source의 유효성은
조건부 정확도 하나가 아니라 **교정 가능한 오류의 prevalence와
오염 기회의 비**로 결정된다. LDPC에서는 source 조건부 정확도가 더
낮아도 교정 가능한 위치가 13,110개라 correction opportunity가
충분하고, 교정:오염 비가 RSC보다 약 470배 좋다. RSC에서는 source가
틀린 cavity bit를 잘 알아맞히더라도 그런 bit가 41개뿐이라 이득
기회가 사실상 없다. 반면 exact pixel LSB를 모르는 source posterior는
대부분의 정상 위치에서 계속 disagreement를 만든다.

즉 “source는 잔여 오류가 이미지 공간에서 보일 때 유효하다”는 시각적
설명은 오류 밀도와 PSNR 대조로 강하게 지지된다. 다만 그 메커니즘은
LDPC에서 source의 조건부 정확도가 더 높아서가 아니라, **눈에 보일
정도로 조밀한 오류에서는 source가 고칠 수 있는 절대 위치 수와
correction/contamination 비가 충분히 커지기 때문**이다.

## [8] |LLR| 가중 실효 교정과 LDPC 조건부 bit-plane

같은 64-block tensor에서 추가 계측만 수행했다. 유용한 교정 위치는
`cavity wrong ∩ source correct`, 오염 시도 위치는
`cavity correct ∩ source wrong`으로 정의했다. 실효 뒤집힘은 두 경로를
같은 local belief에서 비교하기 위해 다음 식으로 재계산했다.

```text
L_after = Q + 0.1 * clip(L_source_posterior - Q, -30, 30)
```

이는 RSC decoder가 각 source 호출 뒤 terminal payload decision에
사용하는 식과 정확히 같다. LDPC에서는 warm-start BP의 다음 비선형
전파까지 포함한 값이 아니라, 동일한 pre-source BP posterior `Q`에
source term만 더한 국소 효과를 분리한 값이다.

### Cavity |LLR| 분위수

| 경로/Es/N0 | 위치 | count | 평균 | p10 | p50 | p90 | p99 |
|---|---|---:|---:|---:|---:|---:|---:|
| RSC +1.0 dB | 교정 기회 | 552 | 1.74 | 0.21 | 1.18 | 4.35 | 7.39 |
| RSC +1.0 dB | 오염 시도 | 114,111 | 16.28 | 9.21 | 16.28 | 23.30 | 29.07 |
| RSC +2.0 dB | 교정 기회 | 80 | 2.04 | 0.23 | 1.26 | 4.65 | 6.50 |
| RSC +2.0 dB | 오염 시도 | 113,218 | 22.36 | 14.23 | 22.36 | 30.47 | 37.14 |
| RSC +2.6 dB | 교정 기회 | 31 | 1.32 | 0.12 | 0.95 | 3.27 | 4.63 |
| RSC +2.6 dB | 오염 시도 | 113,912 | 26.56 | 17.73 | 26.55 | 35.42 | 42.63 |
| LDPC -2.5 dB | 교정 기회 | 13,110 | 1.26 | 0.16 | 0.94 | 2.83 | 4.93 |
| LDPC -2.5 dB | 오염 시도 | 102,530 | 13.81 | 1.68 | 6.66 | 30.00 | 30.00 |

가중 가설의 첫 절반은 두 경로 모두에서 강하게 성립한다. 교정 기회의
median `|Q|`는 약 0.94--1.26인 반면 오염 시도 median은 RSC
16.28--26.55, LDPC 6.66이다. 즉 source가 틀리게 주장하는 정상 bit의
대부분은 cavity가 확신하므로 `α=0.1`로는 뒤집히지 않는다.

### 실제 local sign flip

| 경로/Es/N0 | 교정 성공 / 기회 | 오염 성공 / 시도 | 전체 sign flip | 실효 교정:오염 |
|---|---:|---:|---:|---:|
| RSC +1.0 dB | 213 / 552 (38.59%) | 25 / 114,111 (0.0219%) | 238 | 8.52:1 |
| RSC +2.0 dB | 26 / 80 (32.50%) | 9 / 113,218 (0.00795%) | 35 | 2.89:1 |
| RSC +2.6 dB | 14 / 31 (45.16%) | 1 / 113,912 (0.000878%) | 15 | 14.0:1 |
| LDPC -2.5 dB | 4,543 / 13,110 (34.65%) | 1,515 / 102,530 (1.478%) | 6,058 | 3.00:1 |

사전 예측은 **LDPC에서 실효 비가 1보다 크다**는 부분만 맞았다.
RSC에서 1보다 훨씬 작을 것이라는 예측은 명확히 틀렸다. 첫 source
호출 직후의 damped local decision은 RSC에서도 순교정이며, 오히려
측정 실효 비는 LDPC보다 높다. 따라서 hard opportunity count가
RSC 악화를 과장한다는 점은 확인됐지만, `|LLR|` 가중만으로
`score-on 24/256` 대 `score-off 2/256`의 최종 BLER 역전을 설명할
수는 없다.

이 반증은 원인 범위를 더 좁힌다. RSC의 첫 local injection 자체가
오염을 일으키는 것이 아니라, 그 prior가 두 번째 BCJR의 trellis
constraint를 통해 비국소적으로 전파된 뒤 생기는 cavity 변화 또는
두 번째 source/SPC terminal update에서 악화가 발생한다. 현재 계측은
새 성능 실험 없이 요청된 first-cavity tensor의 국소 효과만 분리했으므로,
어느 단계인지는 아직 분리되지 않았다. 다음 원인 후보는
`first local decision → second BCJR pre-source → second source terminal`
세 지점의 동일 bit별 transition 계측이다.

### LDPC source bit-plane calibration

| 범위 | b7(MSB) | b6 | b5 | b4 | b3 | b2 | b1 | b0(LSB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 전체 source 정합 | 94.93% | 81.37% | 68.29% | 67.73% | 66.64% | 63.50% | 68.00% | 73.14% |
| Cavity 오류 위치의 source 정합 | 95.83% | 81.95% | 61.12% | 62.34% | 59.24% | 55.96% | 61.22% | 69.78% |
| Cavity 오류 count | 2,445 | 2,327 | 2,361 | 2,440 | 2,461 | 2,416 | 2,352 | 2,359 |

LDPC에서는 하위 bit도 전체와 조건부 모두 50%를 넘었다. 특히 LSB는
전체 73.14%, cavity 오류 위치에서 69.78%다. 따라서 RSC에서 관측한
하위 2-bit의 50% 이하 정합을 LDPC에 일반화할 수 없으며, 현재
LDPC calibration 근거로 하위 bit를 차단해서는 안 된다. 가중 후보를
둔다면 `b2`가 조건부 55.96%로 가장 약하지만, 이 진단은 tuning이
아니므로 가중치 선택이나 성능 주장은 하지 않는다.

## [9] Denoiser cavity echo 판정

기존 seed `20260730`의 64 block만 다시 통과시키고 tensor 계측과
입력 masking만 추가했다. LDPC는 `-2.5 dB`, RSC는 `+2.6 dB`이며
채널 성능 sweep이나 parameter tuning은 수행하지 않았다.

### 입력 cavity 대 출력 posterior

| 경로 | 범위 | b7(MSB) | b6 | b5 | b4 | b3 | b2 | b1 | b0(LSB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LDPC | 입력 cavity | 95.13% | 95.36% | 95.29% | 95.14% | 95.10% | 95.18% | 95.31% | 95.30% |
| LDPC | 출력 posterior | 94.93% | 81.37% | 68.29% | 67.73% | 66.64% | 63.50% | 68.00% | 73.14% |
| LDPC | 출력−입력 | -0.20 pp | -14.00 pp | -27.01 pp | -27.41 pp | -28.45 pp | -31.68 pp | -27.31 pp | -22.16 pp |
| RSC | 입력 cavity | 99.99% | 99.99% | 99.98% | 99.99% | 100.00% | 99.99% | 99.98% | 99.99% |
| RSC | 출력 posterior | 95.57% | 86.92% | 79.39% | 75.84% | 74.01% | 66.18% | 54.04% | 41.01% |
| RSC | 출력−입력 | -4.42 pp | -13.07 pp | -20.59 pp | -24.15 pp | -25.99 pp | -33.81 pp | -45.94 pp | -58.98 pp |

LDPC LSB 출력 73.14%는 입력 95.30%를 그대로 보존한 값이 아니다.
22.16 percentage point를 잃은 결과다. MSB만 입력과 출력 정확도가
가깝지만, hard accuracy가 가깝다는 사실 자체는 causal echo를 뜻하지
않으므로 plane ablation으로 직접 확인했다.

### 입력 plane 제거

각 plane을 한 번에 하나씩 0 LLR로 만들었고, 요청된 하위 두 plane은
동시에 0으로 만드는 joint mask도 적용했다.

| 경로 | plane | 원 출력 | 해당 plane 단독 제거 | 하위 2-bit 동시 제거 | 동시 제거 변화 |
|---|---|---:|---:|---:|---:|
| LDPC | b1 | 68.00% | 67.99% | 67.99% | -0.01 pp |
| LDPC | LSB | 73.14% | 72.51% | 72.46% | -0.68 pp |
| RSC | b1 | 54.04% | 60.32% | 62.55% | +8.51 pp |
| RSC | LSB | 41.01% | 44.07% | 50.14% | +9.14 pp |

LDPC 하위 두 bit는 50%로 붕괴하지 않고 사실상 유지됐다. 따라서
73.14%/68.00%의 원인은 해당 cavity LLR의 통과가 아니다. 다른
bit-plane과 image/source prior만으로도 거의 같은 hard decision이
나온다. 다만 이 계측만으로 spatial image structure와 Fashion-MNIST의
강한 pixel/bit marginal bias를 분리할 수는 없으므로, 이를 “정확한
LSB 추론 능력”으로 확대 해석하지 않는다.

더 직접적으로, plane 입력 제거에 대한 출력 LLR 반응
`ΔL_post / ΔQ`의 최소제곱 gain은 LDPC b1 `-0.00095`, LSB
`-0.00023`이고 response RMS도 입력 RMS의 각각 `0.177%`, `0.043%`다.
identity echo라면 gain과 RMS ratio가 1에 가까워야 한다. RSC도 b1
`-0.00144`, LSB `-0.00071`로 같은 결론이다. **Cavity echo 가설은
기각된다.**

### Legacy 대 minus의 src_ext 잔존량

echo가 확인되지 않았지만 purity의 미해결 `D<C`와 연결되는지 확인하기
위해, 기존 purity 설정인 `[2]×15`, `α=0.1`, `β=0`,
`bp_post`/`minus_source_feedback`를 동일 64 block에서 호출하고
practical worktree 수정 없이 14개 source call의 입력과 출력을
기록했다.

| mode/전체 call | 지표 | b1 | LSB |
|---|---|---:|---:|
| legacy `bp_post` | posterior/input gain | 0.00598 | 0.00288 |
| legacy `bp_post` | src_ext/input gain | -0.99402 | -0.99712 |
| minus | posterior/input gain | 0.00539 | 0.00260 |
| minus | src_ext/input gain | -0.99461 | -0.99740 |
| legacy 마지막 call | posterior/input gain | 0.00457 | 0.00220 |
| minus 마지막 call | posterior/input gain | 0.00422 | 0.00204 |

하위 bit posterior에는 두 mode 모두 identity 성분이 거의 없고,
`src_ext = posterior - input`에는 약 `-1×input`이 남는다. 즉 구현은
입력 cavity를 source 정보로 재주입하는 것이 아니라 거의 정확히
제거한다. Legacy와 minus 차이도 b1/LSB gain에서 매우 작다.
따라서 cavity echo/double counting은 `minus`가 legacy보다 나쁜
현상을 설명하지 못한다. 오히려 기존의 “incomplete cavity가
source feedback을 음의 항으로 포함해 implicit self-damping을 만든다”
가능성과 정합하지만, 이번 계측만으로 그 효과를 성능 원인으로
확정하지는 않는다.

### RSC 두 번째 pass 위치

요청된 네 지점의 동일 bit transition은 다음과 같다.

| 단계 | wrong bits | wrong blocks | 직전 단계 교정 | 직전 단계 새 오류 |
|---|---:|---:|---:|---:|
| 첫 BCJR pre-source | 41 | 14 | — | — |
| 첫 source terminal | 28 | 13 | 14 | 1 |
| 두 번째 BCJR pre-source | 22 | 14 | 17 | 11 |
| 두 번째 source terminal | 11 | 4 | 13 | 2 |

두 번째 BCJR는 새 오류 11개를 만들었지만 17개를 교정해 net `-6`,
두 번째 source도 net `-11`이다. Wrong block 수는 BCJR에서
13→14로 한 block 늘지만 마지막 source에서 4로 감소한다. 따라서
“첫 주입의 한 오류가 trellis error event로 증폭돼 bit 오류가
폭발한다”는 가설은 이 paired sample에서 성립하지 않았다. 네 단계
모두의 net bit 방향은 개선이며, 이전 256-block 최종 BLER 역전을
이 단계별 bit count만으로 설명할 수 없다.

## 최종 원인 판정

확정적으로 배제된 항목은 이미지 layout, SPC/CRC 위치, interle버 방향,
LLR 부호, 다른 denoiser/checkpoint, 명시적 systematic likelihood
이중계산이다. 발견된 수치 인터페이스 불일치는 수정했지만 성능 원인은
아니었다.

RSC cavity는 usable waterfall 전체에서 99.8--100%의 bit 정합률이고
실패 block조차 평균 2.93 bit 오류, PSNR 41.22 dB다. Source는 그 소수의
오류 위치에서는 약 76% 정확하지만, exact pixel LSB를 모르기 때문에
정상 cavity와 disagreement하는 위치가 수십만 개다. 따라서 source
주입의 correction opportunity가 contamination opportunity보다
수천 배 작다.

LDPC 경로는 약 `-2.5 dB`에서 BP posterior의 잔여 불확실성이 크기 때문에
동일 source prior가 일부 약한 위치에서 보탬이 될 수 있다. 반면 RSC는
약한 코드 때문에 더 높은 Es/N0에서 동작하고, 그 지점의 BCJR bit
posterior는 이미 source model보다 훨씬 정확하다. 즉 source prior의
가치는 절대 SNR이나 marginal/conditional 정확도 하나가 아니라,
**channel decoder 잔여 오류가 이미지 prior가 볼 수 있을 만큼 조밀한가,
그리고 correction opportunity가 contamination opportunity를
상쇄하는가**로 결정된다. 현재 source model은 RSC 경로의 어느 실용
동작점에서도 최종 BLER 이득을 내지 못했다. 다만 이번 `|LLR|` 계측은
첫 local injection이 RSC에서도 순교정임을 보여 주므로, 최종 악화의
직접 메커니즘을 “첫 주입의 순오염”이나 두 번째 BCJR의 net bit-error
폭발로 설명할 수는 없다. Cavity echo/double counting 역시 기각됐다.

최종 조건 정렬 후에는 block 수준 원인이 확정됐다. First-BCJR
component 진단과 score-off 최종값을 섞었던 비교를 폐기하고, 동일
256 block의 full score-off + SPC와 score-on + SPC를 비교하자 source가
성공 block 22개를 실패로 바꾸고 실패 block은 0개 구제했다.
`22 - 0 = +22`는 `2/256 -> 24/256` 변화와 정확히 같다. 따라서
local/평균 bit 지표가 일부 개선돼도 성공 block 파괴가 최종 BLER를
지배한다.

## 검증 파일

- `experiments/source_wiring_diagnostic.py`
- `experiments/source_module_crosscheck.py`
- `experiments/source_message_mode_probe.py`
- `experiments/source_residual_diagnostic.py`
- `tests/test_source_spc_siso.py`
- `tests/test_source_residual_diagnostic.py`
- `results/source_wiring_diagnostic_fixed*/`
- `results/source_value_transition32/`
- `results/source_residual_{rsc,ldpc}64.json`
- `results/source_echo_{rsc,ldpc}64.json`
- `results/source_score_block_accounting256_unbounded.json`
- `results/source_residual_images/`
