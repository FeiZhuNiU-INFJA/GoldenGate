# 实验记录：20 日截面排序（2026-10-02）

这次实验把基线的固定阈值三分类换成按日截面排序，并用 LightGBM LambdaRank 选股。没有做 MASTER 的市场门控和股票间注意力，也没有换成 Alpha158 因子。

## 依据

不是某一篇论文的复现。标签和损失各有来源，模型是按日分组的梯度提升树。

截面标签沿用 Qlib 和 MASTER 的做法：投资要的是当天谁更强，所以把未来收益在截面上标准化，用名次而不是绝对涨跌当目标。

- Yang, Xiao, Weiqing Liu, Dong Zhou, Jiang Bian, and Tie-Yan Liu. 2020. “Qlib: An AI-Oriented Quantitative Investment Platform.” arXiv:2009.11189.
- Li, Tong, Zhaoyang Liu, Yanyan Shen, Xue Wang, Haokun Chen, and Sen Huang. 2024. “MASTER: Market-Guided Stock Transformer for Stock Price Forecasting.” AAAI. 文中把未来收益在股票集合上做当日 z-score，并注明沿用 Yang et al. 2020。

排序损失用 LambdaRank。它按日期把股票收成一组，直接惩罚排错的名次，而不是拟合一个绝对收益数字。用到截面选股上的代表工作是 Poh 等人的 LambdaMART。

- Burges, Christopher J. C. 2010. “From RankNet to LambdaRank to LambdaMART: An Overview.” Microsoft Research MSR-TR-2010-82.
- Poh, Daniel, Bryan Lim, Stefan Zohren, and Stephen Roberts. 2021. “Building Cross-Sectional Systematic Strategies by Learning to Rank.” Journal of Financial Data Science 3 (2): 70–86.

Kwiatkowski and Chudziak（arXiv:2510.14156，2025）在 MASTER 式 Transformer 上比较了 MSE、Margin、ListNet 和 BPR。那篇用来决定先改标签和损失。本次训练用的是 LambdaRank，不是他们表里夏普最高的 Margin。

验证集 RankIC 为负时把分数乘 −1，是这次实验自己加的规则，不是上述论文里的步骤。

## 数据和标签

数据与基线实验相同：`dataset/{cn,hk,us}/labeled/`，行情到 2026-09-30。面板缓存在 `dataset/panels/rank20.parquet`。

原始目标仍是相对基准的 20 日超额收益（A 股沪深 300、港股恒生、美股标普 500）。同一市场、同一天内：

1. 当天股票数少于 20 只的日期丢掉。
2. 对超额收益做 z-score。
3. 按分位切成 5 档整数相关度，0 为最差，4 为最好。LambdaRank 需要这种整数档，同一档里 +1.6% 和 +15% 不再被当成同一个类，但档内的差距被抹平。

特征只用到标签日及之前的行情：`ret_5`、`ret_20`、`ret_60`、`vol_20`、`activity_ratio_20`（成交额相对 20 日均值；美股成交额全为 0，改用成交量）、`range_pct`、`close_loc`。

| 市场 | 训练 2015–2022 | 验证 2023 | 测试 2024 起 |
|------|----------------|-----------|----------------|
| A 股 | 6,281,067 | 1,165,521 | 3,269,538 |
| 港股 | 127,121 | 18,709 | 49,521 |
| 美股 | 885,863 | 120,591 | 330,094 |

## 模型

每个市场一个 `LGBMRanker`，`objective=lambdarank`，按交易日分组。学习率 0.05，最多 400 棵树，叶子数 63，在验证集 NDCG@5 上早停（40 轮无提升）。A 股和港股停在第 12 棵，美股停在第 24 棵。

| 市场 | 文件 | 分数符号 |
|------|------|----------|
| A 股 | `checkpoints/ranker_cn_h20.txt` | +1 |
| 港股 | `checkpoints/ranker_hk_h20.txt` | −1 |
| 美股 | `checkpoints/ranker_us_h20.txt` | +1 |

训练脚本：`python scripts/train_ranker.py --markets cn hk us --horizon 20`。数值摘要在 `reports/rank20_summary.json`。

## 截面排序结果

RankIC 是每个交易日上，分数与 20 日超额收益的 Spearman 相关，再按日平均。多空价差是当天分数最高 20% 与最低 20% 的 20 日超额之差，再按日平均。ICIR = 日 RankIC 的均值 / 标准差，没有年化。隔 20 个交易日抽一天的 RankIC 用来减轻标签重叠。

| 市场 | 验证 RankIC | 验证价差 | 测试 RankIC | 测试价差 | 测试隔 20 日 RankIC |
|------|-------------|----------|-------------|----------|----------------------|
| A 股 | 0.048 | +0.93% | 0.071 | +1.72% | 0.062 |
| 港股 | 0.097 | +1.83% | 0.045 | −0.63% | 0.013 |
| 美股 | 0.041 | +2.09% | 0.046 | +2.11% | 0.038 |

A 股验证集 242 天里 71% 的日子 RankIC 为正，测试集 646 天里 76% 为正。

港股原始分数在训练期 RankIC 为 +0.07，2023 年变成 −0.10，所以使用时乘了 −1。乘完以后验证和测试的全体 RankIC 为正，但测试集最高 20% 减最低 20% 的 20 日超额仍是 −0.63%。模型主要靠 `vol_20` 和 `range_pct`，其次是 `ret_60`、`ret_20`。2024 年之后，乘完 −1 的高分组过去 20 日平均跌 0.3%，低分组过去 20 日平均涨 4.0%；低分组后来的 20 日超额更高（+2.1% 对 +0.4%）。全体名次相关为正、两头收益差为负，在大约 77 只股票上可以同时出现。

## 2024 年之后，最高 5 只和最低 5 只

每天取分数最高的 5 只和最低的 5 只。收益是之后 1、5、20 个交易日的收盘涨跌幅，先在组内平均，再对交易日平均。当天或未来收盘价不是正数的记录去掉（美股 `NEM` 在 2023 年出现过负价格；本表只含 2024 年之后）。完整数字在 `reports/rank20_forward_test_top5.json`。

| 市场 | 1 日高 / 低 | 5 日高 / 低 | 20 日高 / 低 |
|------|-------------|-------------|--------------|
| A 股 | +0.22% / −0.40% | +0.79% / −2.11% | +2.53% / −3.85% |
| 港股 | +0.08% / +0.17% | +0.49% / +0.94% | +2.00% / +3.63% |
| 美股 | +2.14% / +0.06% | +8.56% / +0.22% | +19.4% / +2.22% |

A 股三个期限都是高分组更高。港股三个期限都是低分组更高，和上面的尾部价差一致。美股高分组的日均被少数大波动股票拉高：1 日、5 日、20 日收益的中位数分别是 +0.61%、+2.36%、+11.3%。

## 结论

把 20 日超额从固定 ±4% 三分类改成当日截面排序之后，A 股和美股在 2024 年之后仍有大约 0.05–0.07 的日 RankIC，最高 5 只的后续收益高于最低 5 只。港股全体名次能排正，极端的 5 只不能。这一版还没有做行业中性，也没有上 MASTER 的截面注意力。
