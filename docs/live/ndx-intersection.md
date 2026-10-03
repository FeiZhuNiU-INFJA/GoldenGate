# 纳斯达克推荐跟踪：三个模型前 5 名交集

从 2026-10-02 收盘开始记。标签分三档：前 5、第 6–15、其余。每个期限用三个已经训好的模型（`checkpoints/ensemble/ranker_ndx_h{5,10,20}_s{0,1,2}.txt`，分数符号都是 +1）各自取前 5 名，三个名单的交集就是当天的推荐。交集为空也记一行，当天没有持仓。5 日、10 日、20 日各自一本账。早于 2026-10-02 的交易日不补。

行情更新之后运行：

```bash
python scripts/update_us_recommendations.py
```

脚本把账本里还没有的新交易日补上，并按本地复权收盘重算下面的涨跌幅。名单在 `docs/live/ndx-intersection.json`。同一轮把数字写进 `docs/live/intersection-data.js`，页面 `docs/live/intersection.html` 不重写，也会更新另一本账（标普 500 与纳斯达克 100）。手改本页会被下一次运行覆盖；想留一句话，写在对应信号的 `note` 字段。

## 口径

入场价是信号日收盘。持有 n 个交易日按纳斯达克 100 自己的交易日往后数，涨跌幅 = 当天收盘 / 信号日收盘 − 1。组合是推荐名单等权。超额 = 组合涨跌幅 − 同期纳斯达克 100 涨跌幅。

名单里有一只当天没有可用收盘，这一行就不写。持有天数仍按纳斯达克 100 的那一天计，所以后面的行不会把缺的那一天算进持有期。5 / 10 / 20 日三列就是持有天数走到 5、10、20 的那一行；还没走到写「未到期」。表记到持有 20 日为止。

收盘价不是正数、相对前后约 11 日中位数偏离超过 5 倍、或单日涨跌超过 2.5 倍的打印，视为没有收盘。


## 5 日模型

### 总表

| 信号日 | 名单 | 已持有 | 5 日组合 | 5 日大盘 | 5 日超额 | 10 日组合 | 10 日大盘 | 10 日超额 | 20 日组合 | 20 日大盘 | 20 日超额 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | APP、ARM、INTC、MSTR | 0 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |


### 2026-10-02

种子 0 前 5：MSTR +0.108 MicroStrategy、APP +0.097 AppLovin、WDC +0.071 Western Digital、ARM +0.071 Arm Holdings、INTC +0.068 Intel
种子 1 前 5：ALAB +0.636 Astera Labs、INTC +0.419 Intel、MSTR +0.373 MicroStrategy、ARM +0.331 Arm Holdings、APP +0.247 AppLovin
种子 2 前 5：ALAB +0.262 Astera Labs、ARM +0.225 Arm Holdings、APP +0.215 AppLovin、MSTR +0.202 MicroStrategy、INTC +0.171 Intel

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| APP | AppLovin | 268.22 | +0.097 | +0.247 | +0.215 |
| ARM | Arm Holdings | 307.49 | +0.071 | +0.331 | +0.225 |
| INTC | Intel | 119.33 | +0.068 | +0.419 | +0.171 |
| MSTR | MicroStrategy | 160.01 | +0.108 | +0.373 | +0.202 |

累计涨跌幅：

| 日期 | 持有 | APP | ARM | INTC | MSTR | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |


## 10 日模型

### 总表

| 信号日 | 名单 | 已持有 | 5 日组合 | 5 日大盘 | 5 日超额 | 10 日组合 | 10 日大盘 | 10 日超额 | 20 日组合 | 20 日大盘 | 20 日超额 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | ALAB、ARM、INTC、MSTR | 0 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |


### 2026-10-02

种子 0 前 5：ALAB +0.726 Astera Labs、ARM +0.527 Arm Holdings、INTC +0.405 Intel、MPWR +0.223 Monolithic Power Systems、MSTR +0.207 MicroStrategy
种子 1 前 5：ALAB +0.487 Astera Labs、MSTR +0.343 MicroStrategy、ARM +0.280 Arm Holdings、INTC +0.279 Intel、HONA +0.217 Honeywell Aerospace
种子 2 前 5：ALAB +0.370 Astera Labs、MSTR +0.315 MicroStrategy、INTC +0.281 Intel、ARM +0.201 Arm Holdings、TER +0.172 Teradyne

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| ALAB | Astera Labs | 350.33 | +0.726 | +0.487 | +0.370 |
| ARM | Arm Holdings | 307.49 | +0.527 | +0.280 | +0.201 |
| INTC | Intel | 119.33 | +0.405 | +0.279 | +0.281 |
| MSTR | MicroStrategy | 160.01 | +0.207 | +0.343 | +0.315 |

累计涨跌幅：

| 日期 | 持有 | ALAB | ARM | INTC | MSTR | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |


## 20 日模型

### 总表

| 信号日 | 名单 | 已持有 | 5 日组合 | 5 日大盘 | 5 日超额 | 10 日组合 | 10 日大盘 | 10 日超额 | 20 日组合 | 20 日大盘 | 20 日超额 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | ADBE、ALAB、MSTR | 0 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |


### 2026-10-02

种子 0 前 5：MSTR +0.499 MicroStrategy、ALAB +0.438 Astera Labs、ARM +0.370 Arm Holdings、ADBE +0.293 Adobe Inc.、INTC +0.269 Intel
种子 1 前 5：ALAB +0.373 Astera Labs、MSTR +0.367 MicroStrategy、ADBE +0.242 Adobe Inc.、ARM +0.196 Arm Holdings、AXON +0.187 Axon Enterprise
种子 2 前 5：MSTR +0.349 MicroStrategy、ALAB +0.341 Astera Labs、WDC +0.260 Western Digital、ADBE +0.241 Adobe Inc.、PLTR +0.218 Palantir Technologies

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| ADBE | Adobe Inc. | 118.84 | +0.293 | +0.242 | +0.241 |
| ALAB | Astera Labs | 350.33 | +0.438 | +0.373 | +0.341 |
| MSTR | MicroStrategy | 160.01 | +0.499 | +0.367 | +0.349 |

累计涨跌幅：

| 日期 | 持有 | ADBE | ALAB | MSTR | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|
| 2026-10-02 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |

