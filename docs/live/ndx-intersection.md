# 纳斯达克笔记：三个模型前 5 名交集

从 2026-10-02 收盘开始记。标签分三档：前 5、第 6–15、其余。每个期限用三个已经训好的模型（`checkpoints/ensemble/ranker_ndx_h{5,10,20}_s{0,1,2}.txt`，分数符号都是 +1）各自取前 5 名，三个名单的交集就是当天记下的名单。交集为空也记一行，当天没有持仓。5 日、10 日、20 日各自一本账。早于 2026-10-02 的交易日不补。

行情更新之后运行：

```bash
python scripts/update_us_recommendations.py
```

脚本把账本里还没有的新交易日补上，并按本地复权收盘重算下面的涨跌幅。名单在 `docs/live/ndx-intersection.json`。同一轮把数字写进 `docs/live/intersection-data.js`，页面 `docs/live/intersection.html` 不重写，也会更新另一本账（标普 500 与纳斯达克 100）。手改本页会被下一次运行覆盖；想留一句话，写在对应信号的 `note` 字段。

## 口径

入场价是信号日收盘。持有 n 个交易日按纳斯达克 100 自己的交易日往后数，涨跌幅 = 当天收盘 / 信号日收盘 − 1。组合是记下的名单等权。超额 = 组合涨跌幅 − 同期纳斯达克 100 涨跌幅。

名单里有一只当天没有可用收盘，这一行就不写。持有天数仍按纳斯达克 100 的那一天计，所以后面的行不会把缺的那一天算进持有期。5 / 10 / 20 日三列就是持有天数走到 5、10、20 的那一行；还没走到写「未到期」。表记到持有 20 日为止。

收盘价不是正数、相对前后约 11 日中位数偏离超过 5 倍、或单日涨跌超过 2.5 倍的打印，视为没有收盘。


## 5 日模型

### 总表

| 信号日 | 名单 | 已持有 | 5 日组合 | 5 日大盘 | 5 日超额 | 10 日组合 | 10 日大盘 | 10 日超额 | 20 日组合 | 20 日大盘 | 20 日超额 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | APP、ARM、INTC、MSTR | 2 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-05 | APP、SNPS | 1 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-06 | ALAB、NBIS | 0 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |


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
| 2026-10-05 | 1 | +5.13% | -1.49% | -2.63% | +2.76% | +0.94% | +0.87% | +0.07% |
| 2026-10-06 | 2 | +3.94% | -1.60% | -5.72% | +2.84% | -0.14% | +1.35% | -1.49% |


### 2026-10-05

种子 0 前 5：APP +0.152 AppLovin、MSTR +0.135 MicroStrategy、LITE +0.101 Lumentum、CRWD +0.083 CrowdStrike、SNPS +0.074 Synopsys
种子 1 前 5：APP +0.381 AppLovin、SNPS +0.351 Synopsys、ALAB +0.325 Astera Labs、NBIS +0.282 Nebius Group、WDC +0.209 Western Digital
种子 2 前 5：APP +0.311 AppLovin、SNPS +0.183 Synopsys、MSTR +0.168 MicroStrategy、ALAB +0.142 Astera Labs、SHOP +0.118 Shopify

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| APP | AppLovin | 281.97 | +0.152 | +0.381 | +0.311 |
| SNPS | Synopsys | 488.47 | +0.074 | +0.351 | +0.183 |

累计涨跌幅：

| 日期 | 持有 | APP | SNPS | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|
| 2026-10-05 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-06 | 1 | -1.13% | +3.42% | +1.14% | +0.48% | +0.67% |


### 2026-10-06

种子 0 前 5：MSTR +0.135 MicroStrategy、ALAB +0.115 Astera Labs、NBIS +0.084 Nebius Group、AMD +0.078 Advanced Micro Devices、PANW +0.078 Palo Alto Networks
种子 1 前 5：ALAB +0.394 Astera Labs、AMD +0.385 Advanced Micro Devices、PANW +0.298 Palo Alto Networks、NBIS +0.287 Nebius Group、CRWV +0.280 CoreWeave
种子 2 前 5：ALAB +0.237 Astera Labs、NBIS +0.208 Nebius Group、SBUX +0.206 Starbucks、MSTR +0.196 MicroStrategy、CEG +0.187 Constellation Energy

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| ALAB | Astera Labs | 389.80 | +0.115 | +0.394 | +0.237 |
| NBIS | Nebius Group | 249.87 | +0.084 | +0.287 | +0.208 |

累计涨跌幅：

| 日期 | 持有 | ALAB | NBIS | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|
| 2026-10-06 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |


## 10 日模型

### 总表

| 信号日 | 名单 | 已持有 | 5 日组合 | 5 日大盘 | 5 日超额 | 10 日组合 | 10 日大盘 | 10 日超额 | 20 日组合 | 20 日大盘 | 20 日超额 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | ALAB、ARM、INTC、MSTR | 2 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-05 | APP、LITE、MSTR | 1 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-06 | ALAB、INTC、MSTR、NBIS | 0 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |


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
| 2026-10-05 | 1 | +3.43% | -1.49% | -2.63% | +2.76% | +0.52% | +0.87% | -0.35% |
| 2026-10-06 | 2 | +11.27% | -1.60% | -5.72% | +2.84% | +1.69% | +1.35% | +0.34% |


### 2026-10-05

种子 0 前 5：APP +0.445 AppLovin、MSTR +0.431 MicroStrategy、LITE +0.369 Lumentum、ALAB +0.369 Astera Labs、WDC +0.293 Western Digital
种子 1 前 5：MSTR +0.459 MicroStrategy、LITE +0.241 Lumentum、APP +0.237 AppLovin、NBIS +0.219 Nebius Group、SNPS +0.198 Synopsys
种子 2 前 5：MSTR +0.324 MicroStrategy、APP +0.275 AppLovin、LITE +0.247 Lumentum、NBIS +0.170 Nebius Group、ALAB +0.141 Astera Labs

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| APP | AppLovin | 281.97 | +0.445 | +0.237 | +0.275 |
| LITE | Lumentum | 1091.67 | +0.369 | +0.241 | +0.247 |
| MSTR | MicroStrategy | 164.43 | +0.431 | +0.459 | +0.324 |

累计涨跌幅：

| 日期 | 持有 | APP | LITE | MSTR | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|
| 2026-10-05 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-06 | 1 | -1.13% | +3.82% | +0.07% | +0.92% | +0.48% | +0.45% |


### 2026-10-06

种子 0 前 5：ALAB +0.384 Astera Labs、MSTR +0.287 MicroStrategy、INTC +0.242 Intel、APP +0.226 AppLovin、NBIS +0.224 Nebius Group
种子 1 前 5：MSTR +0.337 MicroStrategy、ALAB +0.329 Astera Labs、NBIS +0.245 Nebius Group、LITE +0.207 Lumentum、INTC +0.201 Intel
种子 2 前 5：ALAB +0.246 Astera Labs、MSTR +0.210 MicroStrategy、NBIS +0.187 Nebius Group、LITE +0.156 Lumentum、INTC +0.143 Intel

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| ALAB | Astera Labs | 389.80 | +0.384 | +0.329 | +0.246 |
| INTC | Intel | 112.50 | +0.242 | +0.201 | +0.143 |
| MSTR | MicroStrategy | 164.55 | +0.287 | +0.337 | +0.210 |
| NBIS | Nebius Group | 249.87 | +0.224 | +0.245 | +0.187 |

累计涨跌幅：

| 日期 | 持有 | ALAB | INTC | MSTR | NBIS | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|---|
| 2026-10-06 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |


## 20 日模型

### 总表

| 信号日 | 名单 | 已持有 | 5 日组合 | 5 日大盘 | 5 日超额 | 10 日组合 | 10 日大盘 | 10 日超额 | 20 日组合 | 20 日大盘 | 20 日超额 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | ADBE、ALAB、MSTR | 2 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-05 | LITE、PLTR | 1 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-06 | ALAB、LITE、STX | 0 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |


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
| 2026-10-05 | 1 | +0.46% | +3.43% | +2.76% | +2.22% | +0.87% | +1.35% |
| 2026-10-06 | 2 | +0.18% | +11.27% | +2.84% | +4.76% | +1.35% | +3.41% |


### 2026-10-05

种子 0 前 5：APP +0.451 AppLovin、LITE +0.419 Lumentum、PLTR +0.287 Palantir Technologies、CRWV +0.255 CoreWeave、WDAY +0.228 Workday, Inc.
种子 1 前 5：LITE +0.224 Lumentum、PLTR +0.215 Palantir Technologies、ALAB +0.186 Astera Labs、CRWV +0.156 CoreWeave、MSTR +0.153 MicroStrategy
种子 2 前 5：APP +0.355 AppLovin、LITE +0.211 Lumentum、PLTR +0.196 Palantir Technologies、SBUX +0.186 Starbucks、MSTR +0.175 MicroStrategy

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| LITE | Lumentum | 1091.67 | +0.419 | +0.224 | +0.211 |
| PLTR | Palantir Technologies | 189.40 | +0.287 | +0.215 | +0.196 |

累计涨跌幅：

| 日期 | 持有 | LITE | PLTR | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|
| 2026-10-05 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-06 | 1 | +3.82% | +1.41% | +2.62% | +0.48% | +2.14% |


### 2026-10-06

种子 0 前 5：ALAB +0.415 Astera Labs、LITE +0.304 Lumentum、INTC +0.295 Intel、TER +0.287 Teradyne、STX +0.247 Seagate Technology
种子 1 前 5：ALAB +0.238 Astera Labs、STX +0.233 Seagate Technology、SBUX +0.197 Starbucks、LITE +0.178 Lumentum、WDC +0.175 Western Digital
种子 2 前 5：ALAB +0.227 Astera Labs、SBUX +0.206 Starbucks、LITE +0.174 Lumentum、INTC +0.166 Intel、STX +0.163 Seagate Technology

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| ALAB | Astera Labs | 389.80 | +0.415 | +0.238 | +0.227 |
| LITE | Lumentum | 1133.40 | +0.304 | +0.178 | +0.174 |
| STX | Seagate Technology | 805.63 | +0.247 | +0.233 | +0.163 |

累计涨跌幅：

| 日期 | 持有 | ALAB | LITE | STX | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|
| 2026-10-06 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |

