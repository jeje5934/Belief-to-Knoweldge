# 고정 latency 예산 4-system SNR-BLER 비교 — hard CRC 재생성

## 상태와 판정

soft logit을 `CRCDecoder`에 직접 넘긴 과거 그림은 무효로 유지하고, 이번에는
모든 CRC 입력을 `logit > 0 -> bit 1`로 hard decision한 데이터만 합쳐 다시
그렸다. 기존 hard-CRC sigma/저예산 codec 결과를 우선 재사용했고, 비어 있던
PixelCNN/WebP/raw BP 궤적과 source 곡선 끝점만 추가 측정했다.

soft-CRC에서 관찰한 정성 구도는 다음과 같이 유지된다.

1. **receiver-only:** PixelCNN-MAX가 측정 범위에서 계속 지배한다. 선택되는
   BP-30/BP-200은 `-2.4~-3.2 dB`에서 0 failure 또는 그에 준한다.
2. **TX+receiver:** PixelCNN의 architecture-depth 송신 latency `30,576`은
   `L={40,300,1100}`을 모두 넘으므로 세 패널에서 unavailable이다.
3. **L=1100 교차:** 기본 `L_den=50`에서 ours `[5]x20`과 WebP BP-200은
   계속 교차한다. BLER `0.1`에서는 ours가 **0.087 dB**, BLER `0.01`에서는
   WebP가 **0.101 dB** 앞선다.
4. **저 latency:** `L=40` TX 포함 패널에서는 표시 SNR 범위의 모든 available
   system이 BLER `0.1`에 도달하지 못한다. receiver-only에서는 PixelCNN이
   예외이므로 “전 시스템 비실용”은 TX 포함 조건으로만 표현한다.

교수 디스커션의 주 그림은 `fixed_latency_snr_hardcrc_lden50.png`이며,
`L_den=20/100`은 민감도 부록이다.

## 1. latency 모델과 선택

- BP 1 iteration = latency 1.
- denoiser 1회 = `L_den in {20,50,100}`, 기본 50.
- ours = `BP iterations + source calls * L_den`; BP/source 교대는 직렬.
- TX latency: ours/raw 0, WebP 20, PixelCNN `784*39=30,576`.
- 각 latency 예산에서 BLER을 보지 않고, 미리 확정된 LUT 중 worst-case
  nominal latency가 들어오는 가장 높은 구성을 고른다.
- hard-CRC LUT의 sigma endpoint는 budget 20/30/50/100 모두 `0.05`다.

| 이름 | BP schedule | calls | altproj |
|---|---:|---:|---|
| `bp10` | `[10]` | 0 | source off |
| `10x2` | `[10]x2` | 1 | delta=.10, rho=.90, end=.05 |
| `5x6` | `[5]x6` | 5 | delta=.10, rho=.90, end=.05 |
| `5x10` | `[5]x10` | 9 | delta=.05, rho=.90, end=.05 |
| `5x20` | `[5]x20` | 19 | delta=.02, rho=.95, end=.05 |

이 그림은 SNR robustness 감사를 시작하기 전에 동결한 LUT를 사용한다. 후속 감사에서
budget 50의 rho `.85`가 `.90`보다 강건했지만, sparse 독립-seed 점을 기존
full waterfall에 섞지 않기 위해 그림은 `.90`으로 유지했다. `.85` 적용 시에도
ours/WebP 교차라는 정성 결론은 유지된다(`LUT_ROBUSTNESS_REPORT_KO.md`).

| `L_den` | L=40 | L=300 | L=1100 |
|---:|---|---|---|
| 20 | `10x2` (40) | `5x10` (230) | `5x20` (480) |
| 50 | `bp10` (10) | `5x6` (280) | `5x20` (1050) |
| 100 | `bp10` (10) | `10x2` (120) | `5x10` (950) |

receiver-only baseline은 L=40에서 BP-30, L=300/1100에서 BP-200이다.
TX 포함 시 WebP는 각각 BP-20/BP-200/BP-200이며 PixelCNN은 unavailable이다.

## 2. 기본 결과 (`L_den=50`)

낮은 Es/N0가 좋다. `<-3.2`는 표시 범위의 가장 낮은 SNR에서도 목표 BLER
아래였고, `>-2.4`는 가장 높은 SNR에서도 목표에 도달하지 못했다는 뜻이다.

| latency/관점 | 시스템·선택 | knee@0.1 | knee@0.01 |
|---|---|---:|---:|
| L=40 RX | ours `bp10` | >-2.4 | >-2.4 |
|  | PixelCNN BP-30 | <-3.2 | <-3.2 |
|  | WebP BP-30 | -2.617 | -2.476 |
|  | raw BP-30 | >-2.4 | >-2.4 |
| L=40 TX | WebP BP-20 | -2.296 | -2.222 |
| L=300 RX/TX | ours `[5]x6` | -2.460 | -2.225 |
|  | PixelCNN BP-200, RX only | <-3.2 | <-3.2 |
|  | WebP BP-200 | **-2.852** | **-2.751** |
|  | raw BP-200 | >-2.4 | >-2.4 |
| L=1100 RX/TX | ours `[5]x20` | **-2.940** | -2.650 |
|  | PixelCNN BP-200, RX only | <-3.2 | <-3.2 |
|  | WebP BP-200 | -2.852 | **-2.751** |
|  | raw BP-200 | >-2.4 | >-2.4 |

L=1100에서 ours는 낮은 SNR/BLER 0.1 꼬리에서 유리하지만, 낮은 BLER
운용점에서는 WebP가 유리하다. hard CRC 교정은 이 교차를 없애지 않았다.

## 3. `L_den` 민감도

WebP BP-200의 hard-CRC knee는 `-2.852/-2.751 dB`(BLER 0.1/0.01)다.

| `L_den` | L=300 ours knee@0.1 | L=1100 ours knee@0.1 | L=1100 WebP 대비 |
|---:|---:|---:|---:|
| 20 | -2.802 (`5x10`) | **-2.940** (`5x20`) | ours +0.087 dB |
| 50 | -2.460 (`5x6`) | **-2.940** (`5x20`) | ours +0.087 dB |
| 100 | 목표 미도달 (`10x2`) | -2.802 (`5x10`) | WebP +0.050 dB |

따라서 ours의 BLER 0.1 우위는 `L=1100`이고 `L_den<=50`일 때만 남는다.
`L_den=100`에서는 사라진다. 이는 latency 결론이지 FLOPs/energy 우위가 아니다.

## 4. 송신 latency와 정직한 해석

PixelCNN의 `30,576`은 784개 픽셀의 순차 생성과 픽셀당 유효 깊이 39를 모두
critical path로 센 architecture model이다. 픽셀당 네트워크 깊이를 완전히 숨긴
symbol-only 하한은 784이고, 이 극단적 하한이면 L=1100에 PixelCNN BP-200이
들어와 다시 지배한다. 따라서 TX 포함 우위는 full-depth autoregressive latency를
인정하는 경우에만 성립한다. WebP TX=20도 wall-clock이 아닌 추상 allowance다.

핵심 trade-off는 다음 한 문장이다.

> compression은 송신 지연으로 수신 효율을 사고, ours는 송신 계산 제거의 대가로
> 수신 교대 latency를 쓴다.

## 5. 측정·산출물

- 채널: BPSK/AWGN/perfect CSI, payload 6272, N=12600.
- baseline 누락점: 1024 block; WebP BP-200 `-2.70/-2.75 dB`: 3200 block.
- source 누락점: 1024 block; knee 부근 기존 hard-CRC 3200-block 결과 재사용.
- CI: Wilson 95%; 0 failure는 플롯에서 `0.5/n`으로만 표시.
- production `decoder.py`: 수정하지 않음.

hard-CRC 산출물:

- `results/fixed_latency_snr_summary_hardcrc.json`
- `fixed_latency_snr_hardcrc_lden{20,50,100}.png`
- `results/fixed_latency_baseline_hardcrc_1024.json`
- `results/fixed_latency_webp_b200_hardcrc_3200.json`
- `results/fixed_latency_source_{low,edge}_hardcrc_1024.json`

과거 `results/fixed_latency_snr_summary.json`과
`fixed_latency_snr_lden{20,50,100}.png`는 soft-CRC 역사 자료이며 성능 근거로
사용하지 않는다.

## 제안 커밋

1. `experiments: regenerate fixed-latency curves with hard CRC`
2. `analysis: plot hard-CRC fixed-latency reliability map`
3. `docs: replace invalid fixed-latency conclusions`

PNG는 ignore 규칙 때문에 커밋 시 force-add가 필요하다.
