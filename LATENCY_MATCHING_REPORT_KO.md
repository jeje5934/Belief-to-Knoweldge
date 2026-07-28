# Latency-매칭 4-system BLER 비교

## 최종 판정

가설은 **부분적으로만 성립**했다.

1. **송신 latency를 제외하면 ours가 유리한 실용 구간은 없다.**
   PixelCNN-MAX는 `-2.7/-2.9 dB` 모두 평균 BP `15~16` iteration에서
   `0/1024` 실패했고, 모든 `L_den∈{20,50,100}`에서 ours보다 먼저
   practical BLER에 도달했다.
2. **송신 latency를 포함하면 `-2.9 dB`에서 ours가 이기는 구간이
   생긴다.** PixelCNN의 실제 12-layer autoregressive critical path를
   포함한 모델에서, ours B100의 BLER `0.0703`이 WebP의 최선
   `0.1396`보다 낮다. 승리 구간은 다음과 같다.

   | L_den | ours가 유일한 최저 BLER인 latency 구간 | ours BLER |
   |---:|---:|---:|
   | 20 | `[303, 30,592)` | 0.0703 |
   | 50 | `[661, 30,592)` | 0.0703 |
   | 100 | `[1,257, 30,592)` | 0.0703 |

3. 이 구간은 BLER `10^-1` 이하이므로 사용 가능성은 있지만,
   `10^-2`에는 도달하지 못한다. **ours만 `10^-2`에 도달하는 latency
   구간은 하나도 없다.**
4. **저-latency 빠른 수렴 가설은 기각됐다.** `[10]×2`, `[5]×4`에서
   altproj와 EP의 강한 주입을 재탐색했지만 `-2.7/-2.9 dB` 모두 모든
   후보가 `512/512` 실패했다. EP는 altproj를 역전하지 못했다.
5. `-2.7 dB`에서는 송신 latency를 포함해도 WebP가 ours보다 먼저
   BLER `0.00195`에 도달한다. ours가 크게 이기는 구간은 없다.
6. PixelCNN의 784-symbol 순차성만 세고 내부 12-layer 깊이를 무시하는
   극단적 낙관 하한에서는 PixelCNN이 latency 약 `800`에 나타난다.
   이때 `-2.9 dB` ours 승리 구간은 `L_den=20`에서 `[303,800)`,
   `L_den=50`에서 `[661,800)`로 줄고, `L_den=100`에서는 사라진다.

따라서 교수 디스커션용 정직한 주장은 다음이다.

> 무제한 병렬 자원과 PixelCNN의 autoregressive 송신 지연을 함께
> 계산하면, knee SNR에서 우리 방식이 WebP보다 낮은 BLER을 제공하는
> 중간 latency 구간이 존재한다. 그러나 receiver-only, 저-latency,
> BLER `10^-2`, 또는 PixelCNN을 지나치게 낙관적으로 모델링한
> 보수 조건에서는 우위가 사라진다.

핵심 그림은 [latency_matching_key.png](latency_matching_key.png)이다.
`L_den=20/50/100 × 송신 포함/미포함` 전체 패널은
[latency_matching_m2.7.png](latency_matching_m2.7.png)와
[latency_matching_m2.9.png](latency_matching_m2.9.png)에 있다.

## 선행 커밋·push

요청대로 본 작업 전에 직전 sigma-post/budget/codec 결과를 네
커밋으로 나누고 `origin/practical_sigma`에 push했다.

- `4731053 diagnostics: close altproj sigma-post readout precision`
- `4ef19c1 experiments: retune source schemes by BP budget`
- `20e4c1e experiments: compare retuned altproj with equal-budget codecs`
- `e70a48a docs: report low-budget commercial codec duel`

`low_budget_codec_duel.png`는 `.gitignore`의 `*.png` 규칙을 넘겨
force-add했다.

## 1. 추상적 latency 모델

### 1.1 공통 단위와 병렬성

- 한 codeword의 평균 critical-path latency를 비교한다.
- codeword 간, 공간 위치 간, BP iteration 내부 노드 갱신 간 병렬성은
  무제한이라고 가정한다.
- BP flooding iteration은 iteration 사이에 의존하므로 1 iteration을
  latency `1` 단위로 둔다.
- throughput, batch 크기, 메모리 이동, kernel launch, inter-device copy는
  모델에서 제외한다. 따라서 실제 wall-clock 예측이 아니라 구조적
  sequential-depth 비교다.
- CRC early-stop은 각 block이 최초 CRC를 통과한 실제 chunk 시점을
  사용한다. 그래프의 x축은 block별 latency의 평균이다.
- 서로 다른 설정 사이 성능을 보간하지 않는다. 곡선은 “해당 평균
  latency 이하에서 실제 측정된 최저 BLER”의 계단형 frontier다.

### 1.2 BP와 denoiser

BP latency:

`L_BP = 실제 사용 BP iteration 수`.

현재 SongUNet은 `model_channels=64`, `channel_mult=(1,2,2)`,
`num_blocks=2`이며, forward 경로에 UNet block 21개와 attention block
4개가 있다. 이미지 입력 conv와 noise embedding MLP의 병렬 시작,
block당 두 spatial conv, attention의 qkv/projection, 최종 projection을
세면 architectural critical path는 약 `53` stage다.

이 수치를 중간값으로 보고 다음 민감도를 사용했다.

- 낙관: `L_den=20`
- 중간: `L_den=50`
- 보수: `L_den=100`

BP와 source step은 데이터 의존적으로 직렬이므로 ours의 latency는

`L_ours = mean(BP iters) + mean(source calls) × L_den`

이다. schedule이 `[5]×20`이고 최초 성공이 9번째 BP chunk라면, 성공
전에 끝난 source call은 8회다. 마지막 chunk 뒤에는 source call을
세지 않는다.

### 1.3 송신단 latency

두 버전을 모두 산출했다.

1. **receiver-only:** 모든 송신 latency를 0으로 둔다.
2. **TX + receiver:** 다음 값을 더한다.

| 시스템 | 송신 latency | 근거 |
|---|---:|---|
| ours raw | 0 | 압축 없음 |
| raw + BP | 0 | 압축 없음 |
| WebP-MAX | 20 | 최적화 library의 작은 고정 allowance; 실측값 아님 |
| PixelCNN-MAX | 30,576 | 784 sequential symbols × symbol당 39-stage critical path |

PixelCNN 구현은 12개 gated layer다. 각 픽셀의 longest path는 입력 1,
gated block당 3, 출력 2 stage로 `1+12×3+2=39`; arithmetic coding은
28×28 raster 순서를 따라 784번 network probability를 다시 평가한다.
따라서 `784×39=30,576`이다. arithmetic coder 자체와 memory cost를
빼었으므로 이 값도 낙관적이다.

반대로 각 PixelCNN forward를 단일 원자 연산으로 치는 절대 낙관
하한은 `784`다. 보고서의 기본 TX 그래프는 모든 신경망에 layer-depth
기준을 일관 적용한 `30,576`을 쓰고, 판정에는 `784` 하한 민감도도
별도로 병기한다. WebP allowance를 20에서 0으로 바꿔도 아래의 승패는
바뀌지 않고 WebP frontier가 20단위 왼쪽으로만 이동한다.

## 2. 비교 시스템과 측정 조건

공통 조건은 payload 6272 bit, `N=12600`, BPSK/AWGN, perfect CSI다.
대표 SNR은 ours의 waterfall knee 주변인 `-2.9 dB`와 high-reliability
쪽 `-2.7 dB`로 잡았다.

| 시스템 | 표현/LDPC | 설정 |
|---|---|---|
| ours | raw, LDPC rate 0.4997 | budget LUT altproj + 저-latency altproj/EP 탐색 |
| PixelCNN-MAX | 4872-bit container, rate 0.3886 | BP {20,30,50,100,200} |
| WebP-MAX | 5808-bit container, rate 0.4629 | BP {20,30,50,100,200} |
| raw BP | 6272-bit payload, rate 0.4997 | BP {20,30,50,100,200} |

- ours: 512 block, 같은 SNR 안에서 동일 이미지·noise seed로 후보 paired
- baseline: 1024 block, 세 시스템에 같은 stateless unit-noise realization
- 모든 행에 Wilson 95% CI 기록
- legacy `beta=0`, EP `beta_ep=1`, `sigma_post=3.0`
- production `decoder.py` 무수정

## 3. 측정 결과

### 3.1 Ours: 선택된 latency point

괄호는 `평균 BP iteration / 평균 source call`이다.

| SNR | 구성 | BLER [Wilson 95% CI] | BP / source call |
|---:|---|---:|---:|
| -2.7 | BP10, source off | 1.000 [.9926,1] | 10.00 / 0 |
| -2.7 | best [10]×2 | 1.000 [.9926,1] | 20.00 / 1 |
| -2.7 | best [5]×4 | 1.000 [.9926,1] | 20.00 / 3 |
| -2.7 | altproj [5]×6 | .6367 [.5942,.6772] | 29.74 / 4.95 |
| -2.7 | altproj [5]×10 | .0391 [.0254,.0596] | 36.46 / 6.29 |
| -2.7 | altproj [5]×20 | .0195 [.0106,.0356] | 47.16 / 8.43 |
| -2.9 | BP10, source off | 1.000 [.9926,1] | 10.00 / 0 |
| -2.9 | best [10]×2 | 1.000 [.9926,1] | 20.00 / 1 |
| -2.9 | best [5]×4 | 1.000 [.9926,1] | 20.00 / 3 |
| -2.9 | altproj [5]×6 | .9668 [.9475,.9792] | 30.00 / 5 |
| -2.9 | altproj [5]×10 | .2832 [.2459,.3237] | 44.68 / 7.94 |
| -2.9 | altproj [5]×20 | **.0703 [.0512,.0958]** | 64.62 / 11.92 |

`-2.7 dB` B100은 point estimate가 0.0195이고 CI 하한도 0.0106이므로
이 표본에서 BLER `10^-2` 달성을 주장할 수 없다.

### 3.2 EP 저-latency 재확인

`[10]×2`에서는 altproj `delta∈{.05,.1,.2}`, EP
`alpha_ep∈{.05,.1,.2}`, `sigma_end∈{.05,.02}`를 비교했다.
`[5]×4`에서는 altproj `delta∈{.1,.2}`와 EP
`alpha_ep∈{.1,.2}`를 같은 endpoint 범위에서 비교했다.

| SNR | 스케줄 | altproj 후보 | EP 후보 | 판정 |
|---:|---|---:|---:|---|
| -2.7 | [10]×2 | 전부 512/512 실패 | 전부 512/512 실패 | 동률·실용 불가 |
| -2.7 | [5]×4 | 전부 512/512 실패 | 전부 512/512 실패 | 동률·실용 불가 |
| -2.9 | [10]×2 | 전부 512/512 실패 | 전부 512/512 실패 | 동률·실용 불가 |
| -2.9 | [5]×4 | 전부 512/512 실패 | 전부 512/512 실패 | 동률·실용 불가 |

source call이 적으면 EP가 강한 site 주입으로 altproj를 역전할 것이라는
가설은 지지되지 않았다. 저-latency에서 ours가 형식상 단독 winner인
구간은 PixelCNN 첫 성공 latency 전의 `10~15/16`뿐인데, ours BLER는
1.0이므로 성능 우위가 아니라 다른 시스템 point가 아직 없는 공백이다.

### 3.3 BP baseline

각 셀은 BP budget 20/30/50/100/200의 BLER다.

| SNR | 시스템 | 20 | 30 | 50 | 100 | 200 |
|---:|---|---:|---:|---:|---:|---:|
| -2.7 | PixelCNN | 0 | 0 | 0 | 0 | 0 |
| -2.7 | WebP | 1.000 | .6152 | .03125 | .00293 | .00195 |
| -2.7 | raw BP | 1.000 | 1.000 | .9990 | .9785 | .9648 |
| -2.9 | PixelCNN | 0 | 0 | 0 | 0 | 0 |
| -2.9 | WebP | 1.000 | .9883 | .4844 | .1709 | .1396 |
| -2.9 | raw BP | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

PixelCNN의 `0/1024`는 true BLER 0이라는 뜻이 아니라 Wilson 95% 상한
`0.00374`다. 최초 CRC 통과 평균 BP latency는 `-2.7 dB`에서 15.06,
`-2.9 dB`에서 15.95였다.

WebP의 `-2.9 dB`, BP200 결과는 `143/1024=0.1396`, Wilson CI
`[0.1198,0.1622]`; ours B100은 `36/512=0.0703`, CI
`[0.0512,0.0958]`다. 두 CI가 분리되므로 TX-included 중간 latency
구간의 ours 우위는 단순 point-estimate 역전이 아니다.

## 4. Latency-BLER 판정

### 판정 1: 고정 latency 최저 BLER

- **receiver-only:** PixelCNN이 latency 약 15~16 이후 BLER 0/1024로
  항상 최저다. ours practical winner 구간 없음.
- **TX 포함, -2.7 dB:** WebP가 latency 약 56에서 BLER .00195에
  도달하며 PixelCNN 도착 전까지 최저다. ours winner 구간 없음.
- **TX 포함, -2.9 dB:** WebP는 latency 약 95에서 .1396에 머무르고,
  ours B100이 L_den에 따라 latency 303/661/1257에서 .0703으로
  내려가 PixelCNN 도착 전까지 최저가 된다.

### 판정 2: 저-latency 우위와 실용성

저-latency에서 source가 BP 수렴을 앞당기는 우위는 관측되지 않았다.
`[10]×2`와 `[5]×4`는 전부 BLER 1.0이었다. ours가 `BLER<0.1`에
도달하려면 `-2.7 dB`에서는 B50, `-2.9 dB`에서는 B100이 필요하며,
중간 `L_den=50` latency는 각각 약 351과 661이다. 같은 시점 이전에
PixelCNN은 receiver-only에서 이미 성공했고, TX 포함 `-2.7 dB`에서는
WebP도 BLER `10^-2` 아래다.

### 판정 3: 크게 이기는 조합

측정된 유일한 명확한 조합은 다음이다.

- Es/N0 `-2.9 dB`
- 송신 latency 포함
- PixelCNN architectural TX depth 사용
- ours B100 altproj `[5]×20`, `delta=.02`, `rho=.95`
- `L_den=20/50/100`에 따라 latency `303/661/1257` 이후
- ours .0703 vs WebP .1396, 약 2배 BLER 감소

다만 PixelCNN의 symbol-only 하한을 적용하면 승리 폭은 민감하다.

| PixelCNN TX 가정 | L_den=20 | L_den=50 | L_den=100 |
|---|---|---|---|
| 30,576 architectural | `[303,30592)` | `[661,30592)` | `[1257,30592)` |
| 784 symbol-only 하한 | `[303,800)` | `[661,800)` | 없음 |

즉 “병렬 자원이 풍족하면 항상 ours가 유리하다”가 아니라,
**denoiser critical path가 약 50 이하이고 PixelCNN의 autoregressive
network depth를 송신 지연에 포함할 때만 제한적 우위가 존재한다.**

## 5. FLOPs/에너지 단서

구조 hook으로 한 SongUNet forward의 dense work를 약 `1.271×10^9 MAC`
(약 2.54 GFLOP, MAC=2 FLOP convention)으로 셌다. 5G LDPC parity-check
graph는 edge 91,008개이며 한 flooding iteration은 최소 182,016개의
CN/VN edge-message update를 수행한다. 단순히 한 edge update를 한 MAC로
놓는 낙관 하한에서도 denoiser 한 번은 약 **6,982 BP iteration-equivalent
work**다.

이 비율은 정확한 hardware FLOPs 비교가 아니다. BP edge update에는
boxplus, memory access 등이 있어 한 MAC보다 비싸다. 그래도 denoiser가
BP 수천 iteration 규모의 일을 한다는 order는 명확하다. B100의 평균
source call이 8.4~11.9회이므로 work/energy 관점에서 ours가 가볍다고
주장할 수 없다.

따라서 latency 우위는 다음 강한 가정에만 의존한다.

- layer와 노드 내부 병렬 자원이 충분함
- bandwidth와 kernel-launch overhead가 무시 가능함
- 에너지/FLOPs 예산을 제한하지 않음
- 송신 PixelCNN의 raster autoregression을 end-to-end latency에 포함함

## 6. 자율 결정과 실행 기록

- 대표 SNR은 기존 ours knee `-2.908 dB`에 맞춰 `-2.9 dB`, 그보다
  좋은 신뢰도 영역으로 `-2.7 dB`를 선택했다.
- 조정 단계의 연산 절제를 유지하기 위해 ours 후보 screen은 512 block,
  최종 conventional baseline은 1024 block으로 제한했다.
- 저-latency 탐색은 요청된 `[10]×2`, `[5]×4`, `[10]×1`과 EP 강주입에
  한정했다. 전 후보 포화 후 다른 alpha/delta 가설을 추가하지 않았다.
- baseline은 한 warm BP trajectory에서 20/30/50/100/200 snapshot을
  동시에 얻어 중복 계산을 줄였다.
- 첫 baseline 실행에서 complex PAM output에 real random dtype을 직접
  사용해 실패했다. 기존 codec runner의 검증된 real/imag AWGN 생성과
  complex cast를 그대로 적용한 뒤 정상 재실행했다.
- PixelCNN TX의 내부 depth 포함/미포함이 결론을 크게 바꾸므로 기본
  architectural 값과 절대 낙관 하한을 모두 판정에 넣었다.
- WebP의 20-unit TX 비용은 실측이 아니라 명시적 abstract allowance다.
  이를 0으로 두어도 결론이 바뀌지 않는다.
- production `decoder.py`는 수정하지 않았다.

## 7. 산출물과 커밋 제안

- `latency_source_sweep.py`: ours/EP 저-latency·LUT point 계측
- `latency_baseline_sweep.py`: PixelCNN/WebP/raw BP20~200 계측
- `latency_architecture_profile.py`: depth·MAC·LDPC edge profile
- `plot_latency_matching.py`: step frontier, winner range, 전체/핵심 그래프
- `latency_matching_key.png`: 교수 디스커션용 핵심 그림
- `latency_matching_m2.7.png`: -2.7 dB 전체 6-panel 민감도
- `latency_matching_m2.9.png`: -2.9 dB 전체 6-panel 민감도
- `results/latency_source_sweep_512.json`
- `results/latency_baseline_sweep_1024.json`
- `results/latency_architecture_profile.json`
- `results/latency_matching_summary.json`

이번 결과는 요청대로 아직 커밋하지 않았다. 다음 커밋을 제안한다.

1. `experiments: add latency-matched source and compression sweeps`
2. `analysis: model neural and BP critical-path latency`
3. `docs: report latency-matched four-system comparison`

세 PNG는 `.gitignore` 대상이므로 커밋 시 force-add해야 한다.
