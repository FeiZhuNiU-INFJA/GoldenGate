# Extreme Quant

A 股、恒生、标普 500、纳斯达克 100 的日线研究。行情在 `dataset/{cn,hk,us,ndx}/`。纳斯达克 100 里已经属于标普 500 的股票共用 `dataset/us/bars/`，`ndx` 只另存指数和标普里没有的名字。

两条模型：

- **分类基线**：共享 Transformer，5 日 / 20 日相对基准的超额收益，分成买入、中性、卖出。
- **截面排序**：每个市场、每个期限一个 LightGBM LambdaRank。标普和纳指的每日推荐，是三个种子各自前 5 名的交集。

## 环境

Python **3.12**。

```bash
conda env create -f environment.yml
conda activate extreme_quant
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

`environment.yml` 里没有 LightGBM。排序和每日推荐依赖 `requirements.txt`。

Apple Silicon 上 PyTorch 走 MPS，NVIDIA 走 CUDA，否则用 CPU：

```bash
python -c "import torch; print(torch.__version__, 'mps=', torch.backends.mps.is_available(), 'cuda=', torch.cuda.is_available())"
```

东财请求若报 `ProxyError`，保持 `EXTREME_QUANT_KEEP_PROXY` 未设置。客户端默认清掉 `http_proxy` / `https_proxy`。

## 每日美股推荐

只刷新标普 500 和纳斯达克 100，不重训，也不重建 `dataset/panels/rank_multi.parquet`。`--end` 用最近一个已经收盘的美股交易日（美东 16:00 之后；周末停在上周五）。

```bash
EQ_SKIP_YFINANCE=1 python scripts/update_data.py --markets us ndx --end YYYYMMDD
python scripts/update_us_recommendations.py
```

`update_us_recommendations.py` 会补上账本里还没有的交易日，并写出页面用的数字：

| 文件 | 内容 |
|------|------|
| `docs/live/us-intersection.md` | 标普，从 2026-10-01 起 |
| `docs/live/ndx-intersection.md` | 纳指，从 2026-10-02 起 |
| `docs/live/intersection.html` | 固定页面，可切换市场和 5 / 10 / 20 日 |
| `docs/live/intersection-data.js` | 上面那个页面读的数字，每次运行重写 |

模型在 `checkpoints/ensemble/ranker_{us,ndx}_h{5,10,20}_s{0,1,2}.txt`。每个期限三个种子各取前 5，交集才是当天推荐；交集为空记为空仓。当天名字太少（标普少于 450、纳指少于 80）不记。手改 markdown 会被下一次运行覆盖。页面本身不重写。要留一句话，写在对应 JSON 信号的 `note` 字段。

## 数据

来源：A 股 Baostock，港股 / 美股 / 纳指 yfinance，失败时回退东财或新浪。已有 parquet 默认跳过；`--force` 整段重下。

```bash
python scripts/download_data.py --markets cn hk us ndx --workers 8
python scripts/update_data.py --markets cn hk us ndx --end YYYYMMDD
python scripts/build_labels.py --markets cn hk us ndx --summary
```

增量更新从每只股票最后一根 K 线往前重叠几天再合并。前复权在这个窗口之外变了，用 `download_data.py --force` 重下。

离线冒烟（不访问网络）：

```bash
python scripts/make_synthetic_data.py
python scripts/build_labels.py --markets cn hk us --max-symbols 5 --force --summary
python scripts/train.py --epochs 2 --max-symbols 5 --batch-size 64
```

`dataset/` 不进 git。

## 分类基线

按标签日切分：训练 2015-01-01 至 2022-12-31，验证 2023，测试 2024-01-01 起。输入只用 `[t-127, t]`。

相对基准的前向超额。基准：沪深 300、恒生、标普 500、纳斯达克 100。

| 期限 | 买入 `1` | 卖出 `2` | 其余 |
|------|----------|----------|------|
| 5 日 | 超额 ≥ +1.5% | 超额 ≤ −1.5% | 中性 `0` |
| 20 日 | 超额 ≥ +4% | 超额 ≤ −4% | 中性 `0` |

推理分数：`0.4 * (P_buy − P_sell)_5d + 0.6 * (P_buy − P_sell)_20d`。检查点 `checkpoints/best_val_loss.pt`。

```bash
python scripts/train.py --epochs 20
python scripts/evaluate.py --checkpoint checkpoints/best_val_loss.pt --split val
python scripts/emit_signals.py --date 2024-06-28 --markets cn hk us ndx
```

记录：`docs/experiments/2026-10-02-baseline-train.md`。

## 截面排序

特征只看到标签日及之前：`ret_5`、`ret_20`、`ret_60`、`vol_20`、`activity_ratio_20`、`range_pct`、`close_loc`。同一市场、同一天少于 20 只股票的日期丢掉。相关度按当天名次分三档：前 5、第 6–15、其余。

`t < 2024-01-01` 用来拟合，2024-01-01 至 2024-05-31 用来选树和分数正负号，然后在 `t < 2024-06-01` 上重训。`t ≥ 2024-06-01` 只出报告。

```bash
python scripts/compare_rank_horizons.py --markets cn hk us ndx
python scripts/compare_rank_ensemble.py --markets us ndx
python scripts/train_nasdaq_ranker.py
```

单模型写到 `checkpoints/ranker_{market}_h{5,10,20}.txt`。`compare_rank_ensemble.py` 把三个种子写到 `checkpoints/ensemble/`，不覆盖单模型。

`train_ranker.py --horizon 20` 仍按分类那套切分（训练到 2022、验证 2023、测试 2024 起），不走上面的调参窗口。

记录：

- `docs/experiments/2026-10-02-cross-section-rank.md`
- `docs/experiments/2026-10-02-rank-horizons.md`
- `docs/experiments/2026-10-03-nasdaq-100.md`

## 目录

- `config/` — 路径、分类阈值、日期切分、模型宽度
- `data/` — 下载、增量更新、日历、parquet
- `labels/` — 超额收益三分类，以及截面名次
- `models/` — 可替换的 encoder 和分市场分类头
- `train/` — 分类数据集、排序特征和日期协议
- `eval/` — 分类指标、回测、推荐账本和 HTML
- `scripts/` — 命令行入口
- `docs/live/` — 标普和纳指的每日交集
- `docs/experiments/` — 各次训练的数字和口径

## 测试

```bash
pytest tests/
```
