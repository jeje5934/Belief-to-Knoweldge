# 동일 자원 3-arm BLER waterfall 최종 보고서

> 종료 판정의 단일 진입점은
> [`NO_LDPC_SUMMARY.md`](NO_LDPC_SUMMARY.md)다. 이 문서는 waterfall
> 원자료, CI와 보조 실험의 상세 근거로 보존한다.

## 1. 결론

**C(RSC/BCJR + score + SPC)는 A(5G LDPC only)와 B(5G LDPC + score)를
모두 큰 차이로 이기지 못했다. 현재 의사결정은 LDPC 경로 복귀다.**

CRC-detected BLER knee의 Es/N0는 다음과 같다.

| arm | BLER 1e-1 | BLER 1e-2 |
| --- | ---: | ---: |
| A. 5G LDPC only, BP-100 | -2.351 dB | -2.240 dB |
| B. 5G LDPC + score, `[5]x20` | **-2.906 dB** | **-2.661 dB** |
| C. RSC/BCJR + score + SPC | +2.630 dB | +3.662 dB |

따라서 C의 요구 Es/N0는 A보다 1e-1에서 **4.98 dB**, 1e-2에서
**5.90 dB** 높고, B보다 각각 **5.54 dB**, **6.32 dB** 높다. CI 폭보다
수십 배 큰 간격이며 곡선 자체가 분리돼 있다. conventional 기준선을
넘지 못했고 아키텍처 전환을 정당화하지 못했다.

최종 산출물:

- `results/three_arm_waterfall_final.png`
- `results/three_arm_waterfall_final.json`
- `results/three_arm_waterfall_final.csv`
- `results/three_arm_C_alpha_sweep512.json`
- `results/three_arm_aux_decomposition.png`

## 2. 저장소 지형 감사

`git branch -a`에서 확인한 관련 branch는 `no-LDPC`, `practical_sigma`,
`pure-EP_ada-sigma`였다. 정확히 `ada-sigma`라는 branch는 없지만 지정된
우선 위치인 `practical_sigma`가 존재하므로 이를 사용했다.

| arm | 실제 위치 | 사용 상태 |
| --- | --- | --- |
| A | `practical_sigma:decoder.py` 및 Sionna `LDPC5GDecoder` | read-only 호출 |
| B | `practical_sigma:decoder.py`, `ep_mode=False` | read-only 호출 |
| C | `no-LDPC:coding/`, `decoders/`, `experiments/` | 작업 기준 |

`practical_sigma` commit은 `0d54696`이며
`/home/LJH/onlyextrinsic_ada_sigma_practical_sigma` worktree에 체크아웃했다.
실험 후 `git status`는 clean으로, A/B 소스 변경은 없다.

요청에는 `COMPUTE_LESSONS.md`가 root 문서처럼 적혀 있었지만 실제 위치는
`docs/COMPUTE_LESSONS.md`였다. `docs/EP_RESEARCH_SUMMARY.md` §F와 함께
읽었다. 문서의 기존 축은 Eb/N0였고 본 실험은 rate 변환값을 전달하지 않고
명시적 BPSK Es/N0를 직접 생성했다. 기존 Eb/N0 0.5 dB가 내부 LDPC rate로
대략 Es/N0 -2.51 dB가 되는 관계와 측정 knee가 일치한다.

## 3. 공통 규약

- Fashion-MNIST payload: 6,272 bits, MSB-first.
- 전송 길이: N=12,600.
- 공통 payload rate: 6,272/12,600 = 0.4977778.
- channel: real BPSK, bit 0 -> -1, bit 1 -> +1, AWGN, perfect known channel.
- LLR: `log P(1)/P(0)`, 양수 -> bit 1.
- Es/N0: 세 프로세스가 같은 식 `N0=10^(-EsN0/10)`을 사용.
- payload index와 12,600-length 표준 정규 noise는 `(seed, Es/N0)`로
  결정해 가능한 모든 동일 점에서 arm 간 paired.
- A/B: CRC24A, LDPC information K=6,296, internal coderate 0.499683.
- C: CRC-16-CCITT-FALSE, SPC+CRC+tail을 RSC rate matching 안에서 처리.
- 주 BLER: CRC fail을 block error로 정의. true payload BLER, BER,
  undetected error를 별도 계측.

CRC 길이 차이는 유지했다. C의 이론적 undetected 확률은 2^-16으로 A/B의
2^-24보다 높지만 최종 측정 전 점에서 실제 undetected error는 0이었다.
A/B 일부 점에는 payload가 맞고 CRC bit만 틀린 `CRC fail / payload correct`가
있어 주 CRC-BLER가 true BLER보다 소폭 높았다.

## 4. 최종 waterfall 표

괄호 안은 Wilson 95% CI다. `true`는 CRC 판정과 무관한 payload block
error다.

### A. 5G LDPC only, BP-100

| Es/N0 | blocks | CRC-BLER | true BLER | BER |
| ---: | ---: | ---: | ---: | ---: |
| -2.40 | 512 | 87/512 = 0.1699 (0.1399--0.2049) | 84/512 | 1.294e-2 |
| -2.35 | 1024 | 101/1024 = 0.0986 (0.0818--0.1184) | 99/1024 | 6.802e-3 |
| -2.30 | 1024 | 41/1024 = 0.0400 (0.0297--0.0539) | 40/1024 | 2.763e-3 |
| -2.25 | 1024 | 12/1024 = 0.0117 (0.0067--0.0204) | 12/1024 | 8.688e-4 |
| -2.20 | 3200 | 17/3200 = 0.00531 (0.00332--0.00849) | 17/3200 | 4.308e-4 |
| -2.15 | 3200 | 2/3200 = 0.000625 (0.000171--0.002276) | 2/3200 | 7.623e-5 |

### B. 5G LDPC + score, legacy `[5]x20`, alpha=beta=0.1

| Es/N0 | blocks | CRC-BLER | true BLER | BER |
| ---: | ---: | ---: | ---: | ---: |
| -3.00 | 512 | 107/512 = 0.2090 (0.1760--0.2463) | 105/512 | 1.238e-2 |
| -2.90 | 1024 | 98/1024 = 0.0957 (0.0792--0.1153) | 95/1024 | 5.066e-3 |
| -2.80 | 1024 | 32/1024 = 0.0313 (0.0222--0.0438) | 32/1024 | 1.664e-3 |
| -2.75 | 1024 | 29/1024 = 0.0283 (0.0198--0.0404) | 29/1024 | 1.560e-3 |
| -2.70 | 3200 | 50/3200 = 0.0156 (0.0119--0.0205) | 46/3200 | 7.438e-4 |
| -2.60 | 3200 | 16/3200 = 0.00500 (0.00308--0.00811) | 13/3200 | 1.830e-4 |
| -2.50 | 3200 | 1/3200 = 0.000313 (0.000055--0.001768) | 1/3200 | 6.427e-6 |

### C. RSC/BCJR + score + SPC, canonical

| Es/N0 | blocks | CRC-BLER | true BLER | BER |
| ---: | ---: | ---: | ---: | ---: |
| +2.40 | 512 | 98/512 = 0.1914 (0.1597--0.2277) | 98/512 | 7.536e-5 |
| +2.60 | 1024 | 110/1024 = 0.1074 (0.0899--0.1279) | 110/1024 | 4.360e-5 |
| +3.00 | 1024 | 42/1024 = 0.0410 (0.0305--0.0550) | 42/1024 | 1.479e-5 |
| +3.40 | 1024 | 23/1024 = 0.0225 (0.0150--0.0335) | 23/1024 | 9.186e-6 |
| +3.60 | 3200 | 34/3200 = 0.0106 (0.00761--0.0148) | 34/3200 | 3.836e-6 |
| +3.80 | 3200 | 28/3200 = 0.00875 (0.00606--0.0126) | 28/3200 | 3.637e-6 |
| +4.00 | 3200 | 10/3200 = 0.00313 (0.00170--0.00574) | 10/3200 | 1.196e-6 |
| +4.20 | 3200 | 8/3200 = 0.00250 (0.00127--0.00493) | 8/3200 | 9.965e-7 |
| +4.50 | 3200 | 1/3200 = 0.000313 (0.000055--0.001768) | 1/3200 | 9.965e-8 |

같은 Es/N0 직접 anchor에서도 결과가 일치한다. -2.5 dB에서 C는
128/128 fail, A는 59/128, B는 1/3200이었다. +2.6 dB에서 A/B는 각각
0/128 (Wilson upper 0.0291), C는 110/1024 (lower 0.0899)로 CI가 분리됐다.

## 5. 판정

### C가 A를 이기는가

아니다. C는 CRC-BLER 1e-1/1e-2에서 A보다 4.98/5.90 dB 더 필요하다.
동일 payload와 N에서 conventional BP-100보다 현저히 열세다.

### C가 B를 이기는가

아니다. B는 세 arm 중 가장 낮은 knee이며 C보다 5.54/6.32 dB 앞선다.
아키텍처 전환의 성능 근거가 없고, conventional 5G 호환성까지 고려하면
LDPC 경로 복귀가 합리적이다.

### A와 B

B의 source feedback은 A 대비 1e-1에서 약 0.55 dB, 1e-2에서 약
0.42 dB 이득이다. 즉 이 데이터에서는 source prior 자체가 무가치한 것이
아니라 **LDPC의 강한 code constraint와 gentle warm-start schedule 안에서만
유효**했다.

## 6. C alpha sweep

C의 1e-1 knee인 2.6 dB, independent seed `20260731`, 512 paired block이다.

| alpha | CRC-BLER | Wilson 95% CI | BER |
| ---: | ---: | ---: | ---: |
| 0.1 | 49/512 = 0.0957 | 0.0731--0.1243 | 3.893e-5 |
| 0.2 | 396/512 = 0.7734 | 0.7352--0.8076 | 1.007e-3 |
| 0.3 | 512/512 = 1.0 | 0.9926--1.0 | 4.470e-2 |
| 0.5 | 512/512 = 1.0 | 0.9926--1.0 | 1.279e-1 |
| 1.0 | 512/512 = 1.0 | 0.9926--1.0 | 4.223e-1 |

단조 악화와 붕괴가 강하게 재현됐다. 0.1과 0.2의 CI부터 완전히 분리된다.
legacy/EP/altproj에 이어 네 번째 구조에서도 나타났으므로 gentle injection
요구는 특정 code architecture보다 현재 source model의 과신 성질에
가깝다는 증거가 강화됐다.

## 7. 보조 RSC 분해

256 paired block의 가벼운 분해다.

| arm | 2.6 dB | 3.0 dB | 3.4 dB |
| --- | ---: | ---: | ---: |
| RSC/BCJR only | 2/256 | 3/256 | 0/256 |
| RSC/BCJR + SPC | 2/256 | 1/256 | 0/256 |
| RSC/BCJR + score + SPC | **24/256** | **6/256** | **7/256** |

2.6 dB에서 score+SPC CI는 0.0638--0.1357, 두 score-off arm은
0.00215--0.0280으로 분리된다. 이 표본에서는 SPC 자체보다 score 주입이
성능을 크게 악화시켰다. 따라서 C의 약한 결과를 RSC code 자체의 한계로만
설명할 수 없다. 현재 score-SISO 결합이 knee에서 해로운 것이 직접 보인다.

## 8. 자율 판단과 실행 우회

1. 각 arm을 128--256 block으로 먼저 스캔하고, 1e-1 부근은 512--1024,
   1e-2와 1e-3 부근은 1024--3200 block으로 확대했다.
2. TF/torch CUDA 충돌 이력에 따라 A, B, C를 별도 프로세스로 실행했다.
   B는 문서 recipe대로 TF를 CPU에 숨기고 torch denoiser만 CUDA를 썼다.
3. Sionna Mapper는 내부적으로 bit 0 -> +1인 반면 C는 bit 0 -> -1이었다.
   전역 부호는 등가지만 exact pairing을 위해 공통 runner에서 bit 0 -> -1,
   positive logit -> bit 1을 명시적으로 생성해 양 decoder에 직접 넣었다.
4. C 장기 실행에서 Python scalar 변환 관련 CRC 오류가 반복돼 NumPy scalar
   비교로 동일 CRC 연산을 수행하도록 우회했다. known vector와 전체
   unittest를 통과했다.
5. 4.5 dB C 실행에서 GPU process가 한 번 segmentation fault로 종료됐다.
   GPU memory/온도를 확인한 뒤 동일 명령을 새 프로세스로 재실행해
   1/3200 결과를 정상 저장했다.
6. 최초 scan의 독립 SNR 표본에서 일부 비단조 점이 보여 최종 그림에는
   사전 규칙대로 각 관심 구간의 대표본 결과만 사용했다. 작은 표본을
   유리한 방향으로 선택하지 않았다.

## 9. 검증과 변경 상태

- `practical_sigma` worktree: clean, A/B 코드 변경 0.
- no-LDPC 전체 `unittest`: 48개 통과.
- `git diff --check`: 통과.
- plot/JSON/CSV 생성 및 시각 검증 완료.
- 결과 파일은 기존 정책대로 `results/` 아래 ignore 상태다.
- commit은 생성하지 않았다.
