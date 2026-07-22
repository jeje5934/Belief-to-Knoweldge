# no-LDPC 구현 진행 보고서

## 1단계: 저장소 감사

- 작업 브랜치: `no-LDPC`
- 기존 `practical_sigma` 및 LDPC 구현은 수정하지 않았다.
- Fashion-MNIST 표현은 `28 x 28`, `uint8`, 픽셀당 8비트이며 비트 순서는 MSB-first다.
- 원본 payload는 6,272비트이고 기존 고정 채널 길이는 12,600비트다.
- LLR 부호는 `log P(1)/P(0)`이며 양수가 bit 1이다.
- 기존 CRC는 CRC24A지만 새 경로는 명세대로 CRC-16-CCITT-FALSE를 사용한다.
- 기존 채널은 BPSK/AWGN과 QPSK fast Rayleigh fading, perfect/imperfect CSI를 제공한다.
- pretrained score 모델은 그대로 동결하며 첫 구현은 one-step EDM 호출만 사용한다.
- 저장소에서 lossless neural-compression baseline 구현은 발견되지 않았다.

추가/수정 파일:

- `CODEX_HANDOFF_NO_LDPC.md`
- `NO_LDPC_DESIGN.md`
- `NO_LDPC_IMPLEMENTATION_REPORT_KO.md`

미해결 위험:

- RSC-only/no-SPC rate-1/2 출력은 12,582비트라 목표보다 18비트 짧다. 이 충돌은 명시적 `allow_parity_repetition` 정책으로 해결했다. 모든 parity를 한 번 전송하고 균일한 18개 위치를 반복하며, 수신기에서 반복 LLR을 합산한다.
- pretrained score 모델은 현재 `M=8`, `28 x 28` 구조에 묶여 있다.
- neural-compression baseline은 별도 구현 또는 외부 자원이 필요하다.

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

- 총 33개 `unittest`가 로컬 임시 환경에서 통과했으며, 원격 재검증 명령도 같은 테스트 모음을 사용한다.
- noiseless payload exact recovery 및 CRC pass 통과.
- `alpha=0`이 source module 없는 BCJR-only 결과와 일치.
- CRC source-extrinsic 0 검증 통과.
- 기본 자원 길이 12,600 검증 통과.
- 최종 hard decision도 raw source posterior가 아니라
  `Q + alpha * clip(L_S_to_C)`를 사용하도록 검증했다.
- no-SPC baseline의 18 parity 반복, LLR 합산, exact `N=12,600`, noiseless CRC recovery를 검증했다.
- 기존 5G LDPC-only, LDPC warm-score, RSC/no-SPC, RSC/SPC,
  RSC/score, RSC/score+SPC의 6-arm paired matrix 진입점을 1블록으로 실행 검증했다.
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
python3 experiments/no_ldpc_plot.py results/no_ldpc_smoke_stageA.json \
  --output results/no_ldpc_smoke_stageA_bler.png
```

다음 단계:

1. 실제 6,272비트 프레임의 소규모 paired AWGN smoke 결과를 저장한다.
2. frozen score checkpoint를 연결한 1블록 smoke에서 shape/finite-LLR를 검증한다.
3. 그 후에만 outer iteration/alpha/log-MAP ablation을 확장한다.
4. 기존 LDPC 2개 arm을 동일 CRC-16/paired-noise 실험에 연결한다.
5. neural-compression baseline 자원을 확보한 뒤 마지막 비교 arm을 구현한다.
