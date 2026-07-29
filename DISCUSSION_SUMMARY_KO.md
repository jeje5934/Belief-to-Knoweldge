# 교수 디스커션 1-page 요약 — hard CRC 기준

인쇄용 한 페이지 PDF:
[`output/pdf/PROFESSOR_DISCUSSION_ONEPAGE_KO.pdf`](output/pdf/PROFESSOR_DISCUSSION_ONEPAGE_KO.pdf)

## 한 줄 결론

**compression은 송신 지연으로 수신 효율을 사고, ours는 송신 계산 제거의 대가로
수신 교대 latency를 쓴다.** receiver-only reliability는 PixelCNN이 지배하고,
TX latency까지 full-depth로 세면 PixelCNN은 예산 밖으로 빠지지만 ours의 우위는
`L_den<=50`, 높은 latency, BLER 0.1 꼬리라는 제한된 영역에만 존재한다.

## 디스커션 핵심 그림

1. **고정 latency SNR-BLER, 기본 `L_den=50`:**
   [`fixed_latency_snr_hardcrc_lden50.png`](fixed_latency_snr_hardcrc_lden50.png)
   — receiver-only/TX 포함의 대비와 L=40/300/1100 영역을 한 장에서 보여준다.
2. **동일 BP 예산 codec 대결:**
   [`results/low_budget_codec_duel_hardcrc.png`](results/low_budget_codec_duel_hardcrc.png)
   — ours/WebP의 교차, gzip/PNG 대비 우세, budget 감소 시의 한계를 보여준다.

## hard-CRC 격차 분해

BP-100, BLER 0.1의 linear interpolation 기준이다.

| 시스템 | knee |
|---|---:|
| raw + BP-100 | -2.331 dB |
| ours altproj + BP budget 100 | -2.940 dB |
| PixelCNN-MAX + BP-100 | -3.826 dB |

- source prior 이득: raw -> ours = **0.608 dB**.
- compression/rate 이득: raw -> PixelCNN = **1.494 dB**.
- PixelCNN과 ours의 잔여 격차 = **0.886 dB**.

soft-CRC 당시의 `1.44 dB vs 0.50 dB`가 hard CRC에서 `1.49 dB vs 0.61 dB`로
바뀌었지만 결론은 같다. 학습 압축의 rate 여유가 source prior 이득보다 약
0.89 dB 크므로 같은 수신 BP 예산에서 정면 역전하기 어렵다.

## latency 지도

- **receiver-only:** PixelCNN BP-30/BP-200이 표시 범위 `-2.4~-3.2 dB`에서
  지배한다. ours의 실용 우위 영역은 없다.
- **TX 포함:** PixelCNN의 모델 latency는 `784*39=30,576`이라
  `L={40,300,1100}`에 진입하지 못한다. 단, 픽셀당 depth를 완전히 숨기는
  symbol-only 하한 784를 쓰면 L=1100에서 다시 지배하므로 하드웨어 가정에 민감하다.
- **L=1100, L_den=50:** ours/WebP가 교차한다. BLER 0.1에서는 ours가
  0.087 dB, BLER 0.01에서는 WebP가 0.101 dB 앞선다.
- **L=40 TX 포함:** 표시 SNR 범위에서 available system 모두 BLER 0.1에
  도달하지 못한다. 저-latency 수렴 우위가 곧 실용 reliability는 아니다.
- `L_den=100`이면 ours의 L=1100 우위도 사라진다. 이는 FLOPs/energy 우위가
  아니라 풍부한 병렬 자원을 둔 critical-path 결과다.

## 링크 적응 관점

ours가 이득을 내는 곳은 BP가 포화된 뒤 소수 잔여 실패가 남는 영역이다. 실제
adaptive link는 보통 이 구간에서 MCS를 강등하므로 기여를 “MCS 강등을 이긴다”로
표현하면 안 된다. 정확한 표현은 **“같은 MCS를 약 0.6 dB 더 낮은 SNR까지 유지해
강등을 유예한다”**이다. MCS 격자가 성기거나 빠른 피드백이 없는 broadcast/위성
링크에서는 이 유예가 실질 가치가 있다. 반면 MCS 강등과 정면 비교하면 code-rate
이득이 지배하므로 ours가 이기기 어렵고, 이는 PixelCNN compression 대결과 같은
구조다.

## LUT 강건성과 한계

- budget 100: `delta=.02`, `end=.05`, `rho=.95`는 세 SNR에서 CI 내 local
  plateau다. 독립 seed `-2.85 dB`는 CI가 경계에서만 겹쳐 knee 분산이 크다.
- budget 50: rho 탐색 범위가 좁았던 것이 확인됐다. `.85`가 `.90` 대비 독립
  seed에서 `0/8`, `0/33` 파괴/구제를 보여 권고값을 `.85`로 수정한다.
- budget 20: `.1/.9/.05`는 `.85`와 CI 내 local plateau고 delta `.2`는
  붕괴한다. 저예산 이득이 작은 것은 큰 숨은 튜닝 이득보다 source call
  `19 -> 3` 감소의 구조적 결과로 판정한다.
- budget-50 독립 seed 대결도 ours/WebP 교차를 재현했다: BLER 0.1은 ours
  +0.070 dB, BLER 0.01은 WebP +0.076 dB이며 측정점 CI는 겹친다.

fixed-latency 그림의 budget-50 곡선은 비교 도중 설정을 바꾸지 않기 위해 동결된
rho `.90` full waterfall을 사용한다. `.85`는 독립-seed sparse 검증으로만 보고하며,
full waterfall 갱신 전까지 그림 자체를 소급 최적화하지 않는다.

상세: [`LUT_ROBUSTNESS_REPORT_KO.md`](LUT_ROBUSTNESS_REPORT_KO.md),
[`FIXED_LATENCY_SNR_REPORT_KO.md`](FIXED_LATENCY_SNR_REPORT_KO.md).

## 철회·미해결

- ~~`syn=0 != 정답`, source가 valid-but-wrong codeword를 판별한다~~는 명제는
  soft-logit CRC 오용의 artifact로 철회했다. hard CRC 재현에서 해당 블록은 0개였다.
- CRC-ES의 근거는 valid-wrong 판별이 아니라 hard CRC가 full-graph syndrome보다
  먼저 통과한 164/512 block의 안전한 조기 종료다.
- fading 결과는 아직 soft-CRC 영향으로 무효 보류이며 이번 범위에서 재측정하지 않았다.
- purity 실험의 `minus < legacy (D<C)` 메커니즘은 여전히 열려 있다.
- 추상 latency 모델은 교수 디스커션용 지도이며 실제 wall-clock/energy 측정이 아니다.
