# 다중 단말 상관 활용: 복호 루프 내 주입 vs 압축 후 후처리 융합

> 상태: 완료 · 브랜치 `codex/multiview` · hard CRC · AWGN/perfect CSI  
> 핵심 그림: `multiview_loop_vs_post.png`  
> 기계 판독 요약: `results/multiview_loop_vs_post_summary.json`

## 1. 판정

핵심 주장 후보인 **“상관을 복호 루프 안에서 쓰는 것이 복호 후 융합보다
항상 낫다”는 전역 명제로는 성립하지 않는다.** 측정으로 방어되는 범위는 다음과
같다.

1. **WebP-MAX 상대, 낮은 SNR/높은 실패율 구간에서는 C가 E보다 우월하다.**
   BLER 0.1 knee는 C `-3.023 dB`, WebP D/E `-2.891 dB`로 C가
   `0.132 dB` 앞선다. `-3.20/-3.05/-2.85 dB`에서 BLER Wilson CI도
   분리된다. 이 구간에서는 평균 MSE도 C가 E보다 낮다.
2. **BLER 0.01 부근에서는 C와 WebP-E가 동급이다.** `-2.75 dB`,
   1,024 block에서 C `12/1024`, WebP D/E `11/1024`이고 Wilson CI가
   크게 겹친다. C knee는 `-2.727 dB`; WebP는 `-2.75 dB`의 점추정이
   이미 0.0107이고 `-2.65 dB`에서 `0/512`이므로 knee는 두 점 사이로만
   제한한다. 소표본 0을 이용한 허위 정밀 보간은 하지 않았다.
3. **E는 distortion은 고치지만 bit-exact BLER은 고치지 못했다.** 압축
   bitstream CRC가 실패한 뒤 side image로 대체해도 원 bitstream의 CRC/정확
   복원은 회복되지 않는다. 따라서 D와 E의 BLER은 전 점에서 같다. 반면 WebP
   실패 블록의 평균 MSE는 side 직접 대체로 약 9.7--10.5배 감소했다.
4. **PixelCNN-MAX는 C보다 우월하다.** 이번 `-3.20~-2.65 dB` 범위에서
   전 점 `0` failure였고, 기존 hard-CRC BLER 0.1 knee는 `-3.826 dB`다.
   side 후처리가 개입할 실패도 없었다. 따라서 learned compression까지 포함한
   보편 우위 주장은 기각한다.
5. **side의 추가 이득은 존재하지만 이용률은 낮다.** estimated registration
   조건에서 B 대비 C 이득은 BLER 0.1/0.01에서 `0.0667/0.0741 dB`다.
   pixelwise 상호정보량을 기존 rate--dB 결과로 환산한 휴리스틱 가용량 대비
   실현율은 약 `12.5%/13.8%`다.

따라서 논문용으로 가능한 정확한 표현은 다음이다.

> 송신단을 바꾸지 않는 raw+5G-LDPC 링크에서, 수신단의 상관 view를 반복 복호
> 루프에 넣으면 독립 WebP 전송의 post-decode concealment보다 낮은 SNR의
> bit-exact 복구를 개선한다. 그러나 높은 신뢰도 영역에서는 동급이며,
> PixelCNN learned compression의 rate 이점은 넘지 못한다.

## 2. 공정 비교 구성

| arm | 구성 | BP/source 예산 | 상관 사용 위치 |
|---|---|---:|---|
| A | raw 6,272 bit + CRC24A + 5G LDPC, BP only | BP 100 | 없음 |
| B | raw + altproj (`delta=0.02`, `rho=0.95`, geom sigma `0.3->0.05`) | `[5]x20`, source 19회 | 없음 |
| C | B + side pull (`delta_side=0.01`, `sigma>=0.1` gate, source endpoint `0.02`) | `[5]x20`, source 19회 | LDPC loop 내부 |
| D | WebP-MAX 또는 PixelCNN-MAX + CRC24A + LDPC | BP 100 | 없음 |
| E | D 복호 완료 후 CRC 실패 image를 aligned side 또는 side+denoiser로 대체 | BP 100, denoiser 0/1/4/8/19회 허용 | loop 외부 |

공통 규약은 BPSK/AWGN/perfect CSI, `N=12,600`, hard CRC, 동일 SNR,
동일 source index, 가능한 범위의 동일 stateless noise다. raw arm도 codec archive와
짝이 맞는 Fashion-MNIST test 첫 3,200개로 제한했다. WebP/PixelCNN은 각 UE가
독립 압축하며 view 간 상관은 송신단에서 쓰지 않는다.

side는 **receiver에 무손실·추가 채널 비용 0으로 존재**한다고 가정했다. 이는
낙관적 상한이다. 정합은 oracle이 아니라 raw BP-5 hard image와 side 사이의
receiver grid estimate를 썼다. 압축 실패 수신기는 이 raw BP-5 image를 자체적으로
만들 수 없으므로 E에도 같은 정합 결과를 그대로 제공했다. 즉 정합 조건은 E에
유리하다.

## 3. 5-arm BLER 결과

표의 값은 `failure/n [Wilson 95%]`다. D와 E는 bit-exact BLER이 같으므로 codec별
한 열로 합쳤다.

| Es/N0 | A raw BP | B no-side | C side-in-loop | WebP D=E | PixelCNN D=E |
|---:|---:|---:|---:|---:|---:|
| -3.20 | 512/512 [0.9926,1] | 277/512 [0.4977,0.5837] | **131/512 [0.2200,0.2954]** | 479/512 [0.9109,0.9537] | **0/512 [0,0.00745]** |
| -3.05 | 512/512 [0.9926,1] | 112/512 [0.1851,0.2566] | **67/512 [0.1044,0.1628]** | 318/512 [0.5783,0.6621] | **0/512 [0,0.00745]** |
| -2.85 | 512/512 [0.9926,1] | 21/512 [0.0270,0.0619] | **9/512 [0.00927,0.0331]** | 32/512 [0.0446,0.0869] | **0/512 [0,0.00745]** |
| -2.75 | 1008/1024 [0.9748,0.9904] | 22/1024 [0.0142,0.0323] | 12/1024 [0.00672,0.0204] | **11/1024 [0.00601,0.0191]** | **0/1024 [0,0.00374]** |
| -2.65 | 480/512 [0.9131,0.9554] | 5/512 [0.00418,0.0227] | 3/512 [0.00199,0.0171] | **0/512 [0,0.00745]** | **0/512 [0,0.00745]** |

### Knee

| 시스템 | BLER 0.1 knee | BLER 0.01 knee | 해석 |
|---|---:|---:|---|
| A raw BP-100 | 측정 범위 밖 (`>-2.65`) | 범위 밖 | source 없는 raw baseline |
| B raw altproj | -2.956 | -2.653 | side의 순수 추가 이득 기준 |
| C side-in-loop | **-3.023** | **-2.727** | B 대비 0.0667/0.0741 dB |
| WebP D/E | -2.891 | `(-2.75,-2.65)` bracket | C가 0.1에서 0.132 dB 우세, 0.01에서 동급 |
| PixelCNN D/E | 기존 hard-CRC -3.826 | 이번 격자보다 낮음 | 이번 전 점 0 failure |

Knee는 양의 BLER 두 점 사이에서 log-BLER 선형 보간했다. WebP 0.01은 한쪽이
0 failure라 점추정 대신 bracket만 썼다.

## 4. 블록 단위 파괴/구제 회계

| Es/N0 | B->C 구제 | B->C 파괴 | 순구제 | C만 성공 / WebP만 성공 |
|---:|---:|---:|---:|---:|
| -3.20 | 153 | 7 | +146 | 353 / 5 |
| -3.05 | 52 | 7 | +45 | 267 / 16 |
| -2.85 | 17 | 5 | +12 | 30 / 7 |
| -2.75 | 13 | 3 | +10 | 11 / 12 |
| -2.65 | 3 | 1 | +2 | 0 / 3 |

A에서 C로는 다섯 점 모두 파괴 0이고 각각 381/445/503/996/477 block을
구제했다. C는 low-SNR에서 B/WebP보다 확실한 순구제를 만들지만 SNR이 오르면
기회가 빠르게 사라지고 WebP와 교차한다.

E는 D의 compressed bitstream을 바꾸지 않으므로 **bit-exact 구제 0, 파괴 0**다.
이 차이가 loop 내부 활용의 구조적 BLER 이점이다. 다만 E의 목적을 distortion
concealment로 바꾸면 아래와 같이 강하다.

## 5. MSE/PSNR 축

CRC 실패도 포함한 전체 block 평균 MSE(0--1 scale)다. D는 실패 시
all-zero/global mean/mean image/(WebP) best-effort decode 중 SNR별로 가장 낮은
정책을 택한 baseline-favouring lower envelope다. E의 주 정책은 독립 64-block
screen에서 고정한 `side_direct`다.

| Es/N0 | A | B | C | WebP D | WebP E | PixelCNN D/E |
|---:|---:|---:|---:|---:|---:|---:|
| -3.20 | 6.044e-2 | 5.042e-3 | **3.409e-3** | 8.099e-2 | 8.375e-3 | 0 |
| -3.05 | 5.658e-2 | 2.638e-3 | **1.793e-3** | 5.451e-2 | 5.747e-3 | 0 |
| -2.85 | 4.918e-2 | 7.247e-4 | **2.545e-4** | 5.600e-3 | 5.332e-4 | 0 |
| -2.75 | 4.314e-2 | 3.941e-4 | 2.264e-4 | 1.056e-3 | **1.186e-4** | 0 |
| -2.65 | 3.567e-2 | 1.744e-4 | 1.126e-4 | **0** | **0** | 0 |

WebP-E의 low-SNR MSE 개선은 D 대비 `9.7x`(-3.20), `9.5x`(-3.05),
`10.5x`(-2.85), `8.9x`(-2.75)다. 그러나 C는 처음 세 점에서 E보다 다시
`2.46x/3.20x/2.10x` 낮다. `-2.75 dB`부터 압축 stream이 거의 모두 성공하면서
E가 C를 앞선다.

### C와 WebP-E의 tail

| Es/N0 | 시스템 | p50 | p90 | p99 | aggregate PSNR |
|---:|---|---:|---:|---:|---:|
| -3.20 | C | 0 | 1.310e-2 | 2.660e-2 | 24.68 dB |
|  | WebP-E | 7.651e-3 | 1.478e-2 | 2.353e-2 | 20.77 dB |
| -3.05 | C | 0 | 7.286e-3 | 2.424e-2 | 27.46 dB |
|  | WebP-E | 4.997e-3 | 1.455e-2 | 2.083e-2 | 22.41 dB |
| -2.85 | C | 0 | 0 | 1.229e-2 | 35.94 dB |
|  | WebP-E | 0 | 0 | 1.148e-2 | 32.73 dB |
| -2.75 | C | 0 | 0 | 7.543e-3 | 36.45 dB |
|  | WebP-E | 0 | 0 | 4.920e-3 | 39.26 dB |

`-2.65 dB`에서는 양쪽 p99가 0이다. MSE가 일부 SNR에서 완전히 단조롭지 않은
것은 평균이 희소한 대형 실패에 지배되고 각 SNR seed가 독립이기 때문이다.

### E 후처리 후보를 약하게 만들지 않았는가

독립 screen(`-3.05 dB`, 64 block)에서 WebP 실패 block에 다음을 모두 허용했다.

| E 후보 | denoiser 호출 | 평균 MSE |
|---|---:|---:|
| side 직접 대체 | 0 | **0.004989** |
| side denoise, sigma=0.10 | 1 | 0.005386 |
| side denoise, sigma=0.05 | 1 | 0.005056 |
| side denoise, sigma=0.02 | 1 | 0.005001 |
| 반복 denoise | 4 / 8 / 19 | 0.007959 / 0.008979 / 0.009554 |

C와 같은 최대 19회 denoiser budget을 E에도 열어 두었지만 모두 악화했다.
따라서 독립 screen의 최선인 side 직접 대체를 고정해 512/1024-block 검증에 썼다.
이 결과는 현 score model이 정합된 실제 관측을 후처리하는 데에도 prior bias를
더한다는 기존 sigma 진단과 정합한다.

## 6. 상관량과 실현율

Fashion-MNIST test 3,200장, oracle inverse registration 후 8-bit로 재양자화해
256x256 pixel joint histogram의 plug-in entropy를 계산했다.

| strength | H(X) | H(X|Y) | I(X;Y) | I/H(X) | bit agreement |
|---|---:|---:|---:|---:|---:|
| strong | 4.924 | 3.146 | **1.778 bpp** | 36.1% | 79.67% |
| medium | 4.924 | 3.189 | 1.735 bpp | 35.2% | 79.67% |
| weak | 4.924 | 3.308 | 1.616 bpp | 32.8% | 79.45% |

Strong의 bit-plane agreement는 MSB부터
`96.92/90.48/82.85/78.53/75.14/72.79/71.05/69.62%`다. byte exact rate는
43.85%다. `strong/medium/weak`가 oracle 정합 뒤 비슷한 이유는 이 knob가 주로
가역 affine/brightness 크기를 바꾸며, 역정합 뒤에는 경계 crop과 interpolation
손실만 남기 때문이다. 즉 현재 knob는 **receiver registration 난이도**는 바꾸지만
intrinsic conditional entropy를 크게 바꾸는 knob는 아니다. 향후에는 occlusion,
viewpoint-dependent content, sensor noise를 넣어야 상관 강도가 실질적으로 갈린다.

PixelCNN 실제 stream 평균은 3.036 bpp다. 이는 spatial context를 쓰는 operational
proxy이고 위 pixelwise H(X|Y)와 estimator class가 다르므로 둘을 직접 빼지 않았다.

실현율은 정보이론적 정리가 아니라 다음 휴리스틱 cross-calibration으로만 제시한다.

- 기존 raw->PixelCNN rate 이득: `1.494 dB`
- bpp 차이: `8 - 3.036 = 4.964 bpp`
- 환산: `0.301 dB/bpp`
- strong I(X;Y) `1.778 bpp`의 가용 이득 proxy: `0.535 dB`
- 실제 C-B gain: `0.0667/0.0741 dB` @ BLER `0.1/0.01`
- **실현율 proxy: 12.5%/13.8%**

이는 side 상관 자체가 부족해서가 아니라 현재 fixed Gaussian pseudo-observation과
estimated registration이 가용 상관의 작은 일부만 LDPC loop에 전달한다는 근거다.

## 7. 핵심 주장에 대한 최종 답

### C > E인가

- **WebP, BLER 0.1 및 그보다 낮은 SNR:** 예. CI 분리, knee +0.132 dB.
- **WebP, BLER 0.01:** 아니라고도 예라고도 할 수 없다. 1,024 block에서
  12 vs 11 failure로 동급이다.
- **WebP, MSE:** low-SNR에서는 C, near-reliable 구간에서는 E.
- **PixelCNN:** 아니오. D/E가 전 구간 bit-exact이고 MSE 0이다.

따라서 loop 내부 상관 활용의 확인된 장점은 **postprocessing으로는 바꿀 수 없는
compressed-stream BLER를 parity graph 전파로 실제 구제한다는 것**이다. 하지만
강한 learned source coding의 rate 이점을 대체하지는 못한다.

## 8. 자율 결정 및 환경 우회

1. E는 압축 실패 시 자체 raw BP image가 없지만 C와 같은 registration을 제공했다.
   E를 약하게 만들지 않기 위한 보수적 결정이다.
2. E 후보는 64-block 독립 screen에서 선택하고 512/1024-block seed에서는 고정했다.
   SNR별 oracle policy 선택은 주 판정에 쓰지 않았다. D concealment만 요청대로
   lower envelope를 썼다.
3. `-2.75 dB` 한 점만 1,024 block으로 확대했다. C/WebP가 0.01 부근에서
   교차했기 때문이며 다른 점은 512 block으로 유지했다.
4. 혼합 TensorFlow/PyTorch 프로세스에서 SciPy가 간헐적으로 `f32` dtype 이름을
   거부했다. experiment-only registration을 vectorized NumPy bilinear sampler로
   교체했다. 동일 64-block seed에서 기존/신규 BLER는 모두 10/64였고 MSE 차이는
   `1.94e-5`였다. production `decoder.py`는 수정하지 않았다.
5. NPZ lazy 반복 읽기에서 일시 CRC-32 I/O 오류가 발생해 archive를 시작 시
   read-only memory로 eager load했다. 재실행 완료 결과만 채택했다.
6. 3-view는 수행하지 않았다. 2-view C-vs-E가 이미 PixelCNN에는 패하고 WebP에는
   조건부로만 이겼으며, 현재 strength knob도 intrinsic I를 충분히 변화시키지
   못했다. 여러 side pull의 합산/정규화와 상관 중복 제거 없이 3-view 숫자를
   추가하면 해석보다 confound가 커진다고 판단했다.

## 9. 다음 단계 권고

1. **조건부 score `p(X|Y)` 또는 learned virtual-channel variance**를 구현해 현재
   12--14% 실현율을 먼저 높인다. 단순 fixed Gaussian side pull의 추가 sweep은
   우선순위가 낮다.
2. 상관 강도 knob를 occlusion/viewpoint/sensor noise로 다시 정의하고,
   estimated registration 오차까지 포함한 H(X|Y) proxy를 만든다.
3. side도 채널을 거치는 2단 복호를 측정한다. 현재 zero-cost lossless side는
   시스템 상한이지 현실 최종 수치가 아니다.
4. 주장을 WebP-class 송신 경량 baseline에 한정할지, PixelCNN을 넘기 위해
   conditional learned model까지 확장할지 먼저 결정한다. 후자 없이는 learned
   compression 대비 positive 논문 주장은 어렵다.

## 10. 재현 파일

- `multiview_altproj_study.py`: A/B/C, estimated registration, paired pool
- `multiview_postdecode_fusion.py`: D/E, hard CRC, concealment 및 후처리 sweep
- `multiview_correlation_entropy.py`: H(X|Y), I(X;Y), bit-plane 계측
- `analyze_multiview_loop_vs_post.py`: 병합, knee, 회계, 그림
- `results/multiview_*512*.json`, `results/multiview_*1024*.json`: 원자료
- `results/multiview_correlation_entropy_3200.json`: 상관량
- `results/multiview_loop_vs_post_summary.json`: 통합 요약
- `multiview_loop_vs_post.png`: 핵심 그래프
