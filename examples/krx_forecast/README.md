# KRX × Kronos-base Forecast

`krx_all_codes.json` 의 KRX 2,565종목 (KOSPI 838 + KOSDAQ 1,727)을
HuggingFace `NeoQuasar/Kronos-base` 모델로 일봉 예측하는 파이프라인.

- **lookback**: 512일 · **pred_len**: 60 영업일
- **Backtest**: 마지막 60일을 holdout으로 두고 예측 → MAE/RMSE/MAPE/방향정확도
- **Forward**: 최신 데이터 기반 향후 60 영업일 예측

## 1. 환경 준비

```bash
conda activate kronos
pip install -r requirements.txt
python -c "from model import Kronos, KronosTokenizer, KronosPredictor; import pykrx; print('ok')"
```

## 2. 스모크 테스트 (20종목)

```bash
cd D:/01_Career/Kronos/examples/krx_forecast
python fetch_krx.py --codes-json ../../krx_all_codes.json --limit 20 --years 5
python run_forecast.py --mode both --limit 20 --batch-size 8
```

산출물:
- `data/{code}.csv` — 캐시된 일봉 OHLCV + 거래대금
- `output/backtest/{code}.csv` — 예측 vs 실제
- `output/forward/{code}.csv` — 미래 60영업일 예측
- `output/metrics.csv` — 종목별 backtest 지표
- `output/run_skipped.csv`, `output/fetch_skipped.csv` — 스킵 목록

## 3. 전체 실행 (2,565종목)

```bash
python fetch_krx.py --codes-json ../../krx_all_codes.json --years 5
python run_forecast.py --mode both --batch-size 8
```

예상 시간:
- 데이터 페치: 네트워크 바운드, ~1–2시간 (sleep 0.2s 기본)
- 추론: GPU 기준 수 시간, CPU 기준 매우 오래 걸림 → GPU 권장

## 4. 주요 옵션

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--mode` | `both` | `backtest` / `forward` / `both` |
| `--lookback` | 512 | Kronos-base max_context와 동일 |
| `--pred-len` | 60 | 예측 영업일 수 |
| `--batch-size` | 8 | GPU OOM 시 4/2로 감소 |
| `--sample-count` | 1 | 2 이상이면 내부에서 평균 (노이즈 완화) |
| `--temperature` | 1.0 | 샘플링 temperature |
| `--top-p` | 0.9 | nucleus sampling |
| `--seed` | 0 | 0 외 지정 시 `torch.manual_seed` |
| `--refresh` | off | 캐시 무시하고 재다운 (fetch_krx.py) |

## 5. GPU 설정 (권장)

현재 `kronos` conda env는 `torch==2.11.0+cpu` 가 설치되어 있음.
CPU 기준 20종목이 약 15분 걸렸으므로 전체 2,565종목은 ~30시간 소요 예상.
CUDA GPU가 있다면 성능을 위해 GPU torch로 교체 권장:

```bash
conda activate kronos
pip uninstall -y torch
# CUDA 12.x 기준 (버전에 맞춰 조정):
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available())"  # -> True
```

## 6. 트러블슈팅

- **pykrx rate limit / 네트워크 오류**: `--sleep 0.5` 로 간격 늘리기. `fetch_skipped.csv` 확인 후 재실행하면 캐시된 것은 건너뜀.
- **GPU OOM**: `--batch-size 4` 또는 `2` 로 내림. 그래도 OOM이면 `--device cpu`.
- **상장 기간 부족**: lookback(512) + pred_len(60) = 572 영업일 미만 종목은 자동 스킵. `run_skipped.csv` 에 기록.
- **재현성**: 매번 다른 결과가 나오면 `--seed 42` 등으로 고정. 다만 `predict_batch` 내부 sampling 때문에 완전 일치하지 않을 수 있음.
