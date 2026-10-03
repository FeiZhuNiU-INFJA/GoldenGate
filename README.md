# Extreme Quant

A 股、恒生、标普 500、纳斯达克 100 的日线研究。行情在 `dataset/{cn,hk,us,ndx}/`，不进 git。纳指里已经属于标普的股票共用 `dataset/us/bars/`。

- **分类**：共享 Transformer。5 日 / 20 日相对基准的超额，分成买入、中性、卖出。
- **排序**：每个市场、每个期限一个 LightGBM LambdaRank。标普和纳指每天记三个种子各自前 5 名的交集。

笔记页面：<https://feizhuniu-infja.github.io/GoldenGate/>（仓库里是 [`docs/live/intersection.html`](docs/live/intersection.html)）。

## 环境

Python 3.12，conda 环境 `extreme_quant`。`environment.yml` 走清华源，不含 LightGBM，排序还要装 `requirements.txt`。

```bash
conda env create -f environment.yml
conda activate extreme_quant
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

已有环境：`conda env update -f environment.yml --prune`，再跑上面的 pip。

Apple Silicon 上 PyTorch 走 MPS，NVIDIA 走 CUDA，否则 CPU。东财若报 `ProxyError`，不要设置 `EXTREME_QUANT_KEEP_PROXY`。

## 每日美股

只刷新标普和纳指，不重训。`--end` 用最近一个已经收盘的美股交易日（美东 16:00 之后；周末停在上周五）。

```bash
EQ_SKIP_YFINANCE=1 python scripts/update_data.py --markets us ndx --end YYYYMMDD
python scripts/update_us_recommendations.py
```

| 文件 | 内容 |
|------|------|
| `docs/live/us-intersection.md` | 标普，从 2026-10-01 起 |
| `docs/live/ndx-intersection.md` | 纳指，从 2026-10-02 起 |
| `docs/live/intersection.html` | 页面，可切换市场和 5 / 10 / 20 日 |
| `docs/live/intersection-data.js` | 页面读的数字，每次运行重写 |

手改 markdown 会被下一次运行覆盖。页面不重写。

## 数据

A 股 Baostock，港股 / 美股 / 纳指 yfinance，失败时回退东财或新浪。已有 parquet 默认跳过。

```bash
python scripts/download_data.py --markets cn hk us ndx --workers 8
python scripts/update_data.py --markets cn hk us ndx --end YYYYMMDD
python scripts/build_labels.py --markets cn hk us ndx --summary
```

前复权在增量窗口之外变了，用 `download_data.py --force` 重下。

## 训练

分类按标签日切分：训练到 2022-12-31，验证 2023，测试从 2024-01-01 起。5 日超额 ±1.5%、20 日 ±4% 为买入或卖出，其余中性。检查点 `checkpoints/best_val_loss.pt`。

```bash
python scripts/train.py --epochs 20
python scripts/evaluate.py --checkpoint checkpoints/best_val_loss.pt --split val
```

排序用 `t < 2024-01-01` 拟合，2024-01 至 2024-05 选树，再在 `t < 2024-06-01` 上重训。`t ≥ 2024-06-01` 只出报告。

```bash
python scripts/compare_rank_horizons.py --markets cn hk us ndx
python scripts/compare_rank_ensemble.py --markets us ndx
```

口径和数字在 `docs/experiments/`。

## 测试

```bash
pytest tests/
```

离线冒烟不访问网络：`python scripts/make_synthetic_data.py`，再对少量股票跑 `build_labels.py --max-symbols 5` 和 `train.py --epochs 2 --max-symbols 5`。
