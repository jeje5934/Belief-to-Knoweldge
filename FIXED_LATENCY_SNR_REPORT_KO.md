# 고정 latency 예산 4-system SNR-BLER 비교

## 결론

이 비교는 특정 시스템의 전면 우위를 보이지 않는다. 수신기 latency만 계산하면
PixelCNN-MAX + BP가 측정한 모든 latency 예산과 SNR에서 지배적이다. 송신 latency를
포함하고 PixelCNN의 784-step autoregressive encoder에 네트워크 유효 깊이 39를
곱한 모델을 적용하면 PixelCNN은 이번 예산(`L=40/300/1100`)에 들어오지 않는다.
그때도 ours가 WebP-MAX를 이기는 영역은 넓지 않다.

- 기본값 `L_den=50`에서 `L=40`은 모든 raw/source 계열이 사실상 실패한다.
- `L=300`에서는 WebP-MAX + BP-200이 ours `[5]x6`보다 전 구간에서 낫다.
- `L=1100`에서는 두 곡선이 교차한다. WebP는 높은 SNR/낮은 BLER에서 낫고,
  ours `[5]x20`은 낮은 SNR의 BLER 0.1 부근과 그 아래 꼬리에서 낫다.
  BLER 0.1 knee는 ours `-2.925 dB`, WebP `-2.842 dB`로 ours가 `0.083 dB`
  앞서지만, BLER 0.01 knee는 ours `-2.660 dB`, WebP `-2.712 dB`로 WebP가
  `0.051 dB` 앞선다.
- 따라서 정직한 메시지는 “compression은 송신 지연을 대가로 수신 효율을 사고,
  ours는 송신 계산을 없애는 대신 수신 교대를 쓴다”이다.

교수 디스커션의 주 그림은 `fixed_latency_snr_lden50.png`다. `L_den=20/100`
그림은 민감도 부록이다.

## 기존 결과와 산출물

- 추상 latency 모델과 직전 latency-BLER 결과: [LATENCY_MATCHING_REPORT_KO.md](LATENCY_MATCHING_REPORT_KO.md)
- 예산별 ours LUT와 codec 기준선: [LOW_BUDGET_CODEC_DUEL_REPORT_KO.md](LOW_BUDGET_CODEC_DUEL_REPORT_KO.md)
- 기본 그림: `fixed_latency_snr_lden50.png`
- 민감도 그림: `fixed_latency_snr_lden20.png`, `fixed_latency_snr_lden100.png`
- 기계 판독 요약: `results/fixed_latency_snr_summary.json`

## 1. latency 모델과 구성 선택

단위는 무제한 병렬 자원 아래의 추상적 critical-path latency다.

1. BP 1 iteration은 1 latency 단위다. 한 iteration 안의 node/edge 갱신은
   완전히 병렬이라고 둔다.
2. denoiser 1회는 `L_den` 단위다. SongUNet의 유효 순차 깊이에서 얻은
   중간값 50을 기본으로 하고 20과 100을 민감도로 둔다.
3. ours는 `BP iterations + source_calls x L_den`이다. BP와 source step은
   데이터 의존이므로 직렬이다. 고정 예산 구성 선택에는 CRC early-stop의 평균값이
   아니라 worst-case nominal depth를 썼다.
4. 송신 latency는 ours/raw 0, WebP 20, PixelCNN
   `784 pixels x 39 critical stages = 30,576`으로 뒀다.
5. 각 예산에서 BLER을 보지 않고, 미리 확정한 LUT 중 nominal latency가 예산에
   들어오는 가장 높은 예산 구성을 선택했다. 기준선은 BP
   `{20,30,50,100,200}` 중 가능한 가장 큰 값을 택했다.

세 예산은 `L=40/300/1100`으로 정했다. 처음 예시의 `L=120`은 기본
`L_den=50`에서 저예산 구성을 반복할 뿐 새 운용점을 만들지 않았다. `L=300`은
budget-30 교대 구성을, `L=1100`은 검증된 budget-100 구성을 포함해 서로 다른
운용 영역을 보이도록 선택했다.

### Ours LUT

| 이름 | BP 스케줄 | source calls | altproj 설정 |
|---|---:|---:|---|
| `bp10` | `[10]` | 0 | source off |
| `10x2` | `[10]x2` | 1 | 저-latency focused sweep의 best envelope |
| `5x6` | `[5]x6` | 5 | delta=.10, rho=.90, sigma 0.3->0.05 |
| `5x10` | `[5]x10` | 9 | delta=.05, rho=.90, sigma 0.3->0.02 |
| `5x20` | `[5]x20` | 19 | delta=.02, rho=.95, sigma 0.3->0.02 |

`10x2`에서는 altproj와 EP를 함께 검사했지만 표시 구간의 모든 후보가
512/512 실패했다. 따라서 후보별 미세 선택은 결론에 영향을 주지 않는다.

### latency별 선택

| `L_den` | L=40 | L=300 | L=1100 |
|---:|---|---|---|
| 20 | `10x2`, L=40 | `5x10`, L=230 | `5x20`, L=480 |
| 50 | `bp10`, L=10 | `5x6`, L=280 | `5x20`, L=1050 |
| 100 | `bp10`, L=10 | `10x2`, L=120 | `5x10`, L=950 |

수신 전용 기준선은 `L=40`에서 BP-30, `L=300/1100`에서 BP-200이다.
송신 포함 시 WebP는 `L=40`에서 BP-20, 이후 BP-200이다. PixelCNN은 송신
30,576만으로 세 예산을 모두 넘는다.

## 2. 기본 결과: `L_den=50`

### BLER 0.1 / 0.01 knee

`< -3.2`는 가장 낮은 측정 SNR에서도 목표 BLER 아래였음을, `> -2.4`는 표시한
가장 높은 SNR에서도 목표에 도달하지 못했음을 뜻한다.

| latency/관점 | 시스템·선택 | knee @ 0.1 | knee @ 0.01 |
|---|---|---:|---:|
| L=40 RX | ours `bp10` | > -2.4 | > -2.4 |
|  | PixelCNN BP-30 | < -3.2 | < -3.2 |
|  | WebP BP-30 | -2.536 | -2.422 |
|  | raw BP-30 | > -2.4 | > -2.4 |
| L=300 RX/TX | ours `[5]x6` | -2.361 | -2.177 |
|  | PixelCNN BP-200, RX only | < -3.2 | < -3.2 |
|  | WebP BP-200 | -2.842 | -2.712 |
|  | raw BP-200 | > -2.4 | > -2.4 |
| L=1100 RX/TX | ours `[5]x20` | **-2.925** | -2.660 |
|  | PixelCNN BP-200, RX only | < -3.2 | < -3.2 |
|  | WebP BP-200 | -2.842 | **-2.712** |
|  | raw BP-200 | > -2.4 | > -2.4 |

송신 포함 `L=40`의 WebP BP-20 knee는 표시 구간 밖의 기존 점을 포함하면
BLER 0.1에서 약 `-2.096 dB`다. 표에는 표시 범위 안에서의 해석을 우선했다.

### 대표점과 Wilson 95% CI

| L | Es/N0 | ours | WebP | raw | PixelCNN RX-only |
|---:|---:|---:|---:|---:|---:|
| 40 | -2.7 | 1.000 [0.993,1] | RX: .615 [.585,.645], TX: 1.000 [.996,1] | 1.000 [.996,1] | 0/1024 [0,.00374] |
| 300 | -2.7 | .679 [.649,.707] | .00195 [.00054,.00709] | .965 [.952,.974] | 0/1024 [0,.00374] |
| 300 | -2.9 | .967 [.947,.979] | .140 [.120,.162] | 1.000 [.996,1] | 0/1024 [0,.00374] |
| 1100 | -2.7 | .0195 [.0106,.0356] | **.00195** [.00054,.00709] | .965 [.952,.974] | 0/1024 [0,.00374] |
| 1100 | -2.9 | **.0703** [.0512,.0958] | .140 [.120,.162] | 1.000 [.996,1] | 0/1024 [0,.00374] |
| 1100 | -3.2 | **.717** [.676,.754] | .949 [.934,.961] | 1.000 [.996,1] | 0/1024 [0,.00374] |

`L=1100, -2.9 dB`에서는 ours와 WebP의 Wilson CI가 분리되어 낮은 SNR
꼬리 우위가 확인된다. 반대로 `-2.7 dB`에서는 WebP의 CI가 분리되어 낮은
BLER 운용점에서 WebP가 낫다. 단일한 “동급” 결론보다 곡선 교차가 정확한
표현이다.

## 3. `L_den` 민감도

WebP BP-200의 BLER 0.1 knee는 `-2.842 dB`로 고정이다.

| `L_den` | L=300 ours 선택 / knee@0.1 | WebP 대비 | L=1100 ours 선택 / knee@0.1 | WebP 대비 |
|---:|---|---:|---|---:|
| 20 | `[5]x10`, -2.772 | -0.070 dB | `[5]x20`, **-2.925** | **+0.083 dB** |
| 50 | `[5]x6`, -2.361 | -0.482 dB | `[5]x20`, **-2.925** | **+0.083 dB** |
| 100 | `10x2`, 목표 미도달 | 열세 | `[5]x10`, -2.772 | -0.070 dB |

여기서 양수는 ours가 더 낮은 SNR에서 같은 BLER을 달성한다는 뜻이다. ours의
WebP 대비 BLER 0.1 우위는 `L=1100`이면서 `L_den<=50`일 때만 나타난다.
보수값 `L_den=100`에서는 사라진다. 따라서 latency 우위는 denoiser 병렬화
가정에 민감하며 FLOPs/에너지 우위로 해석할 수 없다. 직전 architecture profile의
SongUNet은 forward당 약 1.27 GMAC이고, BP edge-update 환산으로 약 6,982
iteration 상당이므로 FLOPs 축에서는 불리하다.

## 4. 송신 latency 해석

PixelCNN의 30,576은 784개 픽셀을 순차 생성하고 각 픽셀마다 유효 깊이 39가
직렬이라는 architecture-depth 모델이다. 이 모델에서는 송신 포함 세 예산에서
PixelCNN이 unavailable이다. 그러나 픽셀당 네트워크 깊이를 완전히 숨길 수 있다고
가정한 symbol-only 하한은 784다. 이 하한에서는 `L=1100`에 PixelCNN BP-200이
`784+200=984`로 들어오며, 측정 범위 전체에서 다시 지배한다. 따라서 “송신 포함
시 ours가 PixelCNN보다 낫다”는 결론은 full-depth autoregressive latency를 인정할
때만 성립한다. 하드웨어 실측 없이 보편적 우위로 주장하면 안 된다.

WebP 송신 latency 20 역시 최적화 라이브러리를 대표하는 추상 allowance이며 wall
clock 측정값이 아니다. 이 값은 L=300/1100의 BP 선택을 바꾸지 않지만 L=40에서는
BP-30을 BP-20으로 낮춘다.

## 5. 세 판정

1. **receiver-only:** PixelCNN-MAX가 지배한다. ours의 실용 우위 구간은 없다.
2. **TX+receiver:** full-depth PixelCNN 모델에서는 PixelCNN이 예산 밖이다. 그러나
   ours는 L=40에서 실용 BLER에 도달하지 못하고, L=300에서는 WebP에 진다.
   L=1100에서만 낮은 SNR/BLER 0.1 꼬리 우위가 있으며 낮은 BLER에서는 WebP가
   다시 낫다.
3. **가장 큰 ours 우위:** 기본 `L_den=50`, TX 포함, `L=1100`, Es/N0
   `-2.9 dB`에서 ours `.0703 [.0512,.0958]` 대 WebP
   `.1396 [.1198,.1622]`다. 절대 BLER도 0.1 아래라 실용 경계에 걸쳐 있다.
   반면 `L_den=100`에서는 같은 예산에서 이 우위가 사라진다.

## 6. graceful failure 보조축

추가 PSNR/SSIM 실험은 하지 않았다. 압축 arm의 CRC 실패는 이미지 복원이 아니라
bitstream erasure를 만들며, PSNR/SSIM을 계산하려면 검정 이미지, 평균 이미지,
직전 프레임 등 임의의 concealment 정책을 먼저 정해야 한다. 그런 정책 없이
“PixelCNN/WebP 실패 PSNR은 약 8 dB”라고 두는 것은 시스템 고유 성능이 아니다.

대신 현재 BLER 자체에서는 high-latency ours가 더 완만한 tail을 보인다.
`L=1100, -3.2 dB`에서 ours는 `.717 [.676,.754]`, WebP는
`.949 [.934,.961]`다. 이것은 channel-level graceful degradation의 증거지만,
실패 블록의 이미지 품질 우위로 확장해서 해석하지 않는다.

## 7. 측정 및 재현 메모

- 채널은 AWGN + perfect CSI, payload 6272, N=12600이다.
- 기존 latency/codec waterfall 점을 우선 재사용했다.
- 부족한 source 점은 구성당 512 block, 기준선 점은 1024 block으로 채웠다.
- 모든 표본 CI는 Wilson 95%다. 0 failure는 플롯에서 `0.5/n`에 표시했지만 CI와
  표에는 실제 0을 유지했다.
- 곡선 선분은 측정점 사이의 시각 가이드다. knee는 기존 보고서와 같은 linear
  BLER interpolation이며, 촘촘한 독립 waterfall 적합값으로 과해석하지 않는다.
- production `decoder.py`는 수정하지 않았다.

## 제안 커밋

이번 결과는 아직 커밋하지 않았다. 다음 세 논리 단위가 적절하다.

1. `experiments: fill fixed-latency SNR curves`
2. `analysis: plot fixed-latency four-system reliability map`
3. `docs: report fixed-latency SNR-BLER tradeoffs`

플롯은 ignore 규칙 때문에 커밋 시 `git add -f fixed_latency_snr_lden20.png
fixed_latency_snr_lden50.png fixed_latency_snr_lden100.png`가 필요하다.
