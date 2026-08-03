# BP-50 MSE 기준 파라미터 최적화 — 소표본 견적

## 결론

`codex/mse-optimization` 브랜치에서 BP 예산을 정확히 50회,
스케줄을 `[10]×5`로 고정하고 이미지 MSE 기준으로 다시 훑었다.
32-block 축차 스크리닝의 최적점은

> **altproj, δ=0.05, ρ=0.9, geom σ 0.3→0.1, sigma_post=3.0**

이었다. 독립 validation에서는 `ρ=0.85`가 −2.85/−2.7 dB에서 근소하게
앞섰지만 CI가 크게 겹치므로 현 단계의 견적은 **ρ=0.85~0.9 plateau**로 보는
것이 맞다. BLER 기준 BP-50 LUT의 `σ_end=0.05`와 달리, MSE 기준에서는
`σ_end=0.1`이 선택됐다. `sigma_post=3.0`은 그대로 최적이었다.

WebP+BP50과의 비교는 운용점에 따라 역전된다.

- −3.0 dB: ours가 WebP보다 평균 MSE 약 **10~12배 낮다**.
- −2.85 dB: ours가 WebP보다 평균 MSE 약 **4배 낮다**.
- −2.7 dB: WebP가 이번 64개를 전부 정확히 복원해 **측정 MSE=0**으로 이긴다.

즉 이 갈래의 positive 영역은 CRC 실패가 남는 저SNR이다. 우리 출력은 실패해도
형태가 남지만, WebP는 압축 스트림 오류가 이미지 전체 손실로 이어진다. 반대로
WebP가 LDPC로 완전히 보호되는 SNR에서는 lossless codec이므로 반드시 유리하다.

이 결과는 **견적**이다. screen 32 block, validation 64 block이므로 canonical
승격이나 논문 수치로 사용하려면 독립 seed와 더 큰 표본이 필요하다.

## 산출물

- `mse_budget50_study.py`: source/WebP 분리 실행, 축차 sweep, plot
- `results/mse_budget50_source_estimate.json`: source 스윕 원자료
- `results/mse_budget50_webp_estimate.json`: WebP baseline 원자료
- `results/mse_budget50_parameter_sweep.png`: 파라미터 스크리닝
- `results/mse_budget50_vs_webp.png`: 다중 SNR MSE 비교

## 1. 공통 조건과 MSE 정의

| 항목 | ours | WebP baseline |
|---|---:|---:|
| source | raw Fashion-MNIST 6272 bit | lossless WebP MAX container 5808 bit |
| LDPC information bits | 6272+CRC24=6296 | 5808+CRC24=5832 |
| N | 12600 | 12600 |
| LDPC rate | 0.4997 | 0.4629 |
| BP | 정확히 50회, `[10]×5` | 정확히 50회, `[10]×5` |
| early stop | off | off |
| channel | BPSK/AWGN/perfect CSI | 동일 |
| pairing | 동일 이미지 index와 동일 unit-noise realization | 동일 |

주 지표는 최종 payload LLR을 hard decision한 뒤 MSB-first 8-bit pixel로 복원한
`mean((x_hat-x0)^2)/255^2`다. MSE 최적 readout의 가능성을 보기 위해 LLR의
bit 확률로 만든 posterior-mean pixel MSE도 병기했다. 최종 선택은 더 보수적인
hard 8-bit MSE로 했다.

WebP는 lossless라 정상 stream이면 MSE가 0이지만, stream decode 실패 시에는
이미지가 정의되지 않는다. 따라서 다음 정책을 모두 기록했다.

1. **best effort + mean fallback**: 깨진 stream도 먼저 실제 WebP decode를 시도하고,
   불가능하면 paired pool의 per-pixel 평균영상을 출력한다. 보고서의 주 baseline.
2. **CRC gated + mean fallback**: CRC 실패면 decode 결과를 쓰지 않고 평균영상 출력.
3. 위 두 정책의 zero-image fallback 민감도.

평균영상 fallback은 수신기가 source 분포를 안다는 낙관적 baseline이며, 임의로
큰 failure penalty를 주어 우리 방식에 유리하게 만든 비교가 아니다.

## 2. 스킴과 주입 강도

공통 `geom σ 0.3→0.05`, `sigma_post=3`, −2.85 dB, 32 block에서 각 스킴의
최적점은 다음과 같다.

| 스킴 | 최적 주입 | hard MSE/255² | soft MSE/255² | CRC-BLER |
|---|---:|---:|---:|---:|
| altproj | δ=0.05, ρ=0.9 | **0.002348** | 0.001840 | 0.4375 |
| legacy | α=0.1 | 0.007229 | 0.006370 | 0.5625 |
| EP | α_ep=0.05, β_ep=1 | 0.014765 | 0.012845 | 0.8438 |
| raw BP-50 | α=0, β=0 | 0.049104 | 약 0.043 | 1.0000 |

추가로 훑은 범위는 legacy `α={0.02,0.05,0.1,0.2,0.3,0.5}`,
EP `α_ep={0.005,0.01,0.02,0.05,0.1}`, altproj
`δ={0.01,0.02,0.05,0.1,0.2} × ρ={0.85,0.9,0.95,1.0}`다.
altproj의 `δ=0.05`가 분명했고, `ρ=0.85~0.95`는 소표본에서 거의 plateau였다.

legacy `β=0`은 추가 BP-extrinsic boost를 끄는 공정 기준이고, EP `β_ep=1`은
코드 팩터 site를 유지하기 위한 정의상 값이므로 sweep하지 않았다. 두 파라미터는
이름만 비슷하고 의미가 반대다.

## 3. denoiser noise σ sweep

altproj `δ=0.05, ρ=0.9, sigma_post=3`에서:

| σ 경로 | hard MSE/255² | bootstrap mean 95% | CRC-BLER |
|---|---:|---:|---:|
| geom 0.3→0.1 | **0.001175** | [0.000399, 0.002222] | 0.3125 |
| geom 0.3→0.2 | 0.001610 | [0.000653, 0.002751] | 0.3438 |
| geom 0.3→0.05 | 0.002348 | [0.001008, 0.004009] | 0.4375 |
| geom 0.5→0.05 | 0.002654 | [0.001225, 0.004355] | 0.4062 |
| fixed 0.3 | 0.003755 | [0.001480, 0.006191] | 0.3750 |
| geom 0.3→0.02 | 0.004811 | [0.002733, 0.007126] | 0.6875 |
| geom 0.3→0.01 | 0.007243 | [0.004895, 0.009711] | 0.7812 |
| fixed 0.1 | 0.011336 | [0.008173, 0.014528] | 0.9062 |
| fixed 0.05 | 0.025746 | [0.022534, 0.028791] | 1.0000 |
| fixed 0.02 | 0.031608 | [0.028760, 0.034381] | 1.0000 |

단순히 σ를 작게 두는 것은 실패했다. 초반에는 `0.3`의 prior 교정력이 필요하고,
4번의 source call 동안 `0.1`까지만 완만하게 낮추는 것이 MSE에 가장 좋았다.
BP-50 `[10]×5`는 source call이 4회뿐이므로, budget-100에서 좋았던 낮은 endpoint를
그대로 옮기면 너무 빠른 신뢰 상승이 된다.

## 4. sigma_post sweep

altproj `δ=0.05, ρ=0.9, geom σ 0.3→0.1`에서:

| sigma_post | hard MSE/255² | bootstrap mean 95% | CRC-BLER |
|---:|---:|---:|---:|
| 3.0 | **0.001175** | [0.000402, 0.002116] | 0.3125 |
| 6.0 | 0.001372 | [0.000496, 0.002445] | 0.4688 |
| 1.5 | 0.003221 | [0.001477, 0.005065] | 0.4062 |
| 0.75 | 0.005310 | [0.002960, 0.007859] | 0.5312 |
| 12.0 | 0.007373 | [0.004970, 0.009995] | 0.9375 |
| 24.0 | 0.028202 | [0.026233, 0.030273] | 1.0000 |

MSE 목적에서도 `sigma_post=3`이 중심이다. 너무 작으면 과신한 bit LLR이
오염을 만들고, 너무 크면 source 정보가 사실상 사라진다.

## 5. 독립 validation과 WebP 비교

아래 ours는 세 점에 공통으로 쓸 수 있는 `δ=0.05, ρ=0.85,
geom σ 0.3→0.1, sigma_post=3`이다. 각 점 64 block이며 screen과 seed가 다르다.

| Es/N0 | ours hard MSE/255² [boot 95%] | ours BLER | WebP best-effort+mean MSE [boot 95%] | WebP BLER | 판정 |
|---:|---:|---:|---:|---:|---|
| −3.00 | 0.006324 [0.004851,0.007906] | 0.8750 | 0.062466 [0.050763,0.074556] | 0.7656 | ours 약 9.9× 낮음 |
| −2.85 | 0.002633 [0.001498,0.004068] | 0.4375 | 0.011610 [0.005189,0.018976] | 0.1719 | ours 약 4.4× 낮음 |
| −2.70 | 0.000143 [0.000015,0.000327] | 0.0938 | **0 (64/64 exact)** | 0 | WebP 우위 |

−3.0 dB 전용 후보 `ρ=0.85, geom 0.3→0.2, sigma_post=6`은
MSE `0.005068 [0.003916,0.006369]`로 공통 후보보다 더 좋았다. 즉 SNR이 더
낮아질수록 σ와 `sigma_post`를 높여 source prior를 강하게 유지하는 방향성이
보인다. 다만 64-block 관측이므로 SNR별 LUT로 확정할 단계는 아니다.

−2.85 dB에서 WebP의 CRC-gated+mean 정책은 MSE 0.010824로 best-effort보다
조금 좋았지만, 여전히 ours보다 약 4.1배 높다. −3.0 dB에서는 best-effort가
CRC-gated보다 조금 좋았다. 이 차이까지 함께 보존한 이유는 corrupted WebP를
표시할지 폐기할지가 응용 정책이기 때문이다.

## 6. BLER와 MSE가 갈라지는 이유

ours의 CRC 성공 블록은 모두 hard image MSE=0이었다. 실패 블록만 보면:

| Es/N0 | ours 실패 수 | 실패 블록 조건부 MSE/255² |
|---:|---:|---:|
| −3.00 | 56/64 | 0.00723 |
| −2.85 | 28/64 | 0.00602 |
| −2.70 | 6/64 | 0.00153 |

WebP는 −2.85 dB에서 11/64만 CRC 실패했지만 전체 평균 MSE가 0.0116이었다.
반면 ours는 28/64가 CRC 실패해도 평균 MSE가 0.00263이다. 따라서 이 비교에서
ours의 이득은 더 많은 block을 bit-exact하게 만드는 것이 아니라, **실패 block의
왜곡을 국소화하여 graceful reconstruction을 남기는 것**이다.

soft posterior-mean readout은 공통 후보의 hard MSE를 −2.85 dB에서
`0.002633→0.002077`로 약 21% 더 낮췄고 순위도 거의 유지했다. 향후 MSE 전용
시스템이라면 bit hard decision보다 posterior-mean pixel 출력을 정식 readout으로
삼을 근거가 있다.

## 7. 자율 결정과 보류

- SNR은 기존 BP-50 튜닝점 −2.85 dB를 screen 중심으로 잡고, 양옆 −3.0/−2.7을
  독립 seed로 확인했다.
- 계산량을 제한하기 위해 전격자 곱 대신 `scheme/strength → σ → sigma_post →
  상위 3×3×3 interaction` 순으로 축차 축소했다.
- production `decoder.py`는 수정하지 않았다.
- `llr_max=30`, 학습 시 `sigma_data`, 네트워크 구조는 runtime source-injection
  파라미터가 아니므로 이번 견적에서 변경하지 않았다.
- 현 결과로 정식 canonical을 바꾸지 않는다. 다음 단계가 필요하면 `ρ={0.85,0.9}`,
  `σ_end={0.1,0.2}`, `sigma_post={3,6}`만 256~512 block 독립 seed로 재검증하면 된다.
