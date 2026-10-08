# 美股笔记：三个模型前 5 名交集

从 2026-10-01 收盘开始记。标签分三档：前 5、第 6–15、其余。每个期限用三个已经训好的模型（`checkpoints/ensemble/ranker_us_h{5,10,20}_s{0,1,2}.txt`，分数符号都是 +1）各自取前 5 名，三个名单的交集就是当天记下的名单。交集为空也记一行，当天没有持仓。5 日、10 日、20 日各自一本账。早于 2026-10-01 的交易日不补。

行情更新之后运行：

```bash
python scripts/update_us_recommendations.py
```

脚本把账本里还没有的新交易日补上，并按本地复权收盘重算下面的涨跌幅。名单在 `docs/live/us-intersection.json`。同一轮把数字写进 `docs/live/intersection-data.js`，页面 `docs/live/intersection.html` 不重写，也会更新另一本账（标普 500 与纳斯达克 100）。手改本页会被下一次运行覆盖；想留一句话，写在对应信号的 `note` 字段。

## 口径

入场价是信号日收盘。持有 n 个交易日按标普 500 自己的交易日往后数，涨跌幅 = 当天收盘 / 信号日收盘 − 1。组合是记下的名单等权。超额 = 组合涨跌幅 − 同期标普 500 涨跌幅。

名单里有一只当天没有可用收盘，这一行就不写。持有天数仍按标普 500 的那一天计，所以后面的行不会把缺的那一天算进持有期。5 / 10 / 20 日三列就是持有天数走到 5、10、20 的那一行；还没走到写「未到期」。表记到持有 20 日为止。

收盘价不是正数、相对前后约 11 日中位数偏离超过 5 倍、或单日涨跌超过 2.5 倍的打印，视为没有收盘。


## 5 日模型

### 总表

| 信号日 | 名单 | 已持有 | 5 日组合 | 5 日大盘 | 5 日超额 | 10 日组合 | 10 日大盘 | 10 日超额 | 20 日组合 | 20 日大盘 | 20 日超额 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-01 | COHR、FICO、MRNA | 4 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-02 | FICO、MRNA、P | 3 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-05 | CTVA、FICO、ILMN、MRNA | 2 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-06 | CTVA、MRNA、SWKS | 1 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-07 | CTVA、MRNA、P、SMCI | 0 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |


### 2026-10-01

2026-10-02 记下的第一笔，依据 2026-10-01 美股收盘。

种子 0 前 5：FICO +0.232 Fair Isaac、LITE +0.227 Lumentum、COHR +0.222 Coherent Corp.、MRNA +0.221 Moderna、ACN +0.182 Accenture
种子 1 前 5：COHR +0.879 Coherent Corp.、FICO +0.768 Fair Isaac、P +0.686 Everpure、MRNA +0.554 Moderna、CIEN +0.519 Ciena
种子 2 前 5：COHR +0.503 Coherent Corp.、P +0.447 Everpure、MRNA +0.422 Moderna、ACN +0.349 Accenture、FICO +0.339 Fair Isaac

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| COHR | Coherent Corp. | 319.19 | +0.222 | +0.879 | +0.503 |
| FICO | Fair Isaac | 661.75 | +0.232 | +0.768 | +0.339 |
| MRNA | Moderna | 188.94 | +0.221 | +0.554 | +0.422 |

累计涨跌幅：

| 日期 | 持有 | COHR | FICO | MRNA | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|
| 2026-10-01 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-02 | 1 | +5.59% | -0.08% | +0.57% | +2.03% | +0.73% | +1.29% |
| 2026-10-05 | 2 | +4.53% | +4.21% | +7.55% | +5.43% | +1.40% | +4.03% |
| 2026-10-06 | 3 | +6.01% | +5.09% | -0.78% | +3.44% | +1.99% | +1.45% |
| 2026-10-07 | 4 | +4.82% | +3.03% | +3.99% | +3.94% | +1.77% | +2.18% |


### 2026-10-02

种子 0 前 5：P +0.233 Everpure、FICO +0.229 Fair Isaac、MRNA +0.220 Moderna、SMCI +0.171 Supermicro、INTC +0.163 Intel
种子 1 前 5：FICO +0.836 Fair Isaac、MRNA +0.715 Moderna、P +0.715 Everpure、COIN +0.557 Coinbase、HPE +0.524 Hewlett Packard Enterprise
种子 2 前 5：FICO +0.523 Fair Isaac、P +0.504 Everpure、MRNA +0.418 Moderna、HPE +0.381 Hewlett Packard Enterprise、COIN +0.371 Coinbase

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| FICO | Fair Isaac | 661.25 | +0.229 | +0.836 | +0.523 |
| MRNA | Moderna | 190.01 | +0.220 | +0.715 | +0.418 |
| P | Everpure | 140.14 | +0.233 | +0.715 | +0.504 |

累计涨跌幅：

| 日期 | 持有 | FICO | MRNA | P | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|
| 2026-10-02 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-05 | 1 | +4.29% | +6.95% | +2.67% | +4.63% | +0.66% | +3.97% |
| 2026-10-06 | 2 | +5.17% | -1.34% | +5.04% | +2.96% | +1.25% | +1.71% |
| 2026-10-07 | 3 | +3.10% | +3.41% | +8.93% | +5.15% | +1.02% | +4.12% |


### 2026-10-05

种子 0 前 5：CTVA +0.303 Corteva、MRNA +0.241 Moderna、ILMN +0.208 Illumina, Inc.、FICO +0.168 Fair Isaac、SWKS +0.134 Skyworks Solutions
种子 1 前 5：CTVA +1.237 Corteva、MRNA +0.641 Moderna、FICO +0.629 Fair Isaac、ILMN +0.296 Illumina, Inc.、COHR +0.243 Coherent Corp.
种子 2 前 5：CTVA +0.567 Corteva、MRNA +0.512 Moderna、ILMN +0.365 Illumina, Inc.、FICO +0.344 Fair Isaac、SWKS +0.337 Skyworks Solutions

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| CTVA | Corteva | 12.39 | +0.303 | +1.237 | +0.567 |
| FICO | Fair Isaac | 689.61 | +0.168 | +0.629 | +0.344 |
| ILMN | Illumina, Inc. | 146.84 | +0.208 | +0.296 | +0.365 |
| MRNA | Moderna | 203.21 | +0.241 | +0.641 | +0.512 |

累计涨跌幅：

| 日期 | 持有 | CTVA | FICO | ILMN | MRNA | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|---|
| 2026-10-05 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-06 | 1 | +12.27% | +0.85% | -6.86% | -7.75% | -0.37% | +0.58% | -0.95% |
| 2026-10-07 | 2 | +16.63% | -1.14% | -8.83% | -3.31% | +0.84% | +0.36% | +0.48% |


### 2026-10-06

种子 0 前 5：CTVA +0.337 Corteva、MRNA +0.241 Moderna、SWKS +0.188 Skyworks Solutions、P +0.187 Everpure、SMCI +0.182 Supermicro
种子 1 前 5：CTVA +1.443 Corteva、SWKS +0.692 Skyworks Solutions、MRNA +0.569 Moderna、ILMN +0.456 Illumina, Inc.、STX +0.378 Seagate Technology
种子 2 前 5：CTVA +0.589 Corteva、MRNA +0.435 Moderna、ILMN +0.427 Illumina, Inc.、SWKS +0.396 Skyworks Solutions、CIEN +0.384 Ciena

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| CTVA | Corteva | 13.91 | +0.337 | +1.443 | +0.589 |
| MRNA | Moderna | 187.46 | +0.241 | +0.569 | +0.435 |
| SWKS | Skyworks Solutions | 82.79 | +0.188 | +0.692 | +0.396 |

累计涨跌幅：

| 日期 | 持有 | CTVA | MRNA | SWKS | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|
| 2026-10-06 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-07 | 1 | +3.88% | +4.81% | +0.77% | +3.16% | -0.22% | +3.38% |


### 2026-10-07

种子 0 前 5：MRNA +0.243 Moderna、SMCI +0.238 Supermicro、CTVA +0.223 Corteva、P +0.216 Everpure、SWKS +0.171 Skyworks Solutions
种子 1 前 5：CTVA +0.969 Corteva、P +0.841 Everpure、MRNA +0.824 Moderna、SWKS +0.674 Skyworks Solutions、SMCI +0.642 Supermicro
种子 2 前 5：P +0.532 Everpure、MRNA +0.494 Moderna、SMCI +0.456 Supermicro、CTVA +0.434 Corteva、FICO +0.371 Fair Isaac

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| CTVA | Corteva | 14.45 | +0.223 | +0.969 | +0.434 |
| MRNA | Moderna | 196.48 | +0.243 | +0.824 | +0.494 |
| P | Everpure | 152.66 | +0.216 | +0.841 | +0.532 |
| SMCI | Supermicro | 44.94 | +0.238 | +0.642 | +0.456 |

累计涨跌幅：

| 日期 | 持有 | CTVA | MRNA | P | SMCI | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|---|
| 2026-10-07 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |


## 10 日模型

### 总表

| 信号日 | 名单 | 已持有 | 5 日组合 | 5 日大盘 | 5 日超额 | 10 日组合 | 10 日大盘 | 10 日超额 | 20 日组合 | 20 日大盘 | 20 日超额 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-01 | COHR、LITE、MRNA | 4 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-02 | 空仓 | — | — | — | — | — | — | — | — | — | — |
| 2026-10-05 | ILMN、MRNA | 2 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-06 | ILMN、MRNA | 1 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-07 | MRNA、SMCI | 0 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |


### 2026-10-01

2026-10-02 记下的第一笔，依据 2026-10-01 美股收盘。

种子 0 前 5：MRNA +0.521 Moderna、COHR +0.485 Coherent Corp.、LITE +0.433 Lumentum、HPE +0.433 Hewlett Packard Enterprise、FICO +0.416 Fair Isaac
种子 1 前 5：COHR +0.968 Coherent Corp.、LITE +0.618 Lumentum、FICO +0.544 Fair Isaac、MRNA +0.505 Moderna、P +0.489 Everpure
种子 2 前 5：COHR +0.142 Coherent Corp.、P +0.116 Everpure、MRNA +0.113 Moderna、MPC +0.110 Marathon Petroleum、LITE +0.098 Lumentum

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| COHR | Coherent Corp. | 319.19 | +0.485 | +0.968 | +0.142 |
| LITE | Lumentum | 1045.78 | +0.433 | +0.618 | +0.098 |
| MRNA | Moderna | 188.94 | +0.521 | +0.505 | +0.113 |

累计涨跌幅：

| 日期 | 持有 | COHR | LITE | MRNA | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|
| 2026-10-01 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-02 | 1 | +5.59% | +3.79% | +0.57% | +3.32% | +0.73% | +2.58% |
| 2026-10-05 | 2 | +4.53% | +4.39% | +7.55% | +5.49% | +1.40% | +4.09% |
| 2026-10-06 | 3 | +6.01% | +8.38% | -0.78% | +4.53% | +1.99% | +2.54% |
| 2026-10-07 | 4 | +4.82% | +6.24% | +3.99% | +5.02% | +1.77% | +3.25% |


### 2026-10-02

种子 0 前 5：FICO +0.569 Fair Isaac、MRNA +0.449 Moderna、HPE +0.374 Hewlett Packard Enterprise、P +0.374 Everpure、COHR +0.368 Coherent Corp.
种子 1 前 5：FICO +0.982 Fair Isaac、HPE +0.709 Hewlett Packard Enterprise、COIN +0.635 Coinbase、COHR +0.593 Coherent Corp.、SWKS +0.449 Skyworks Solutions
种子 2 前 5：MRNA +0.131 Moderna、P +0.131 Everpure、SMCI +0.095 Supermicro、FDS +0.088 FactSet、BR +0.086 Broadridge Financial Solutions

这一天三个模型的前 5 名没有交集，不建仓。


### 2026-10-05

种子 0 前 5：MRNA +0.533 Moderna、CTVA +0.514 Corteva、COHR +0.359 Coherent Corp.、FICO +0.348 Fair Isaac、ILMN +0.304 Illumina, Inc.
种子 1 前 5：CTVA +1.164 Corteva、MRNA +1.006 Moderna、FICO +0.547 Fair Isaac、COHR +0.510 Coherent Corp.、ILMN +0.272 Illumina, Inc.
种子 2 前 5：ILMN +0.133 Illumina, Inc.、MRNA +0.111 Moderna、NEM +0.086 Newmont、P +0.086 Everpure、LITE +0.082 Lumentum

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| ILMN | Illumina, Inc. | 146.84 | +0.304 | +0.272 | +0.133 |
| MRNA | Moderna | 203.21 | +0.533 | +1.006 | +0.111 |

累计涨跌幅：

| 日期 | 持有 | ILMN | MRNA | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|
| 2026-10-05 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-06 | 1 | -6.86% | -7.75% | -7.31% | +0.58% | -7.88% |
| 2026-10-07 | 2 | -8.83% | -3.31% | -6.07% | +0.36% | -6.43% |


### 2026-10-06

种子 0 前 5：CTVA +0.685 Corteva、MRNA +0.518 Moderna、SWKS +0.434 Skyworks Solutions、ILMN +0.399 Illumina, Inc.、BE +0.371 Bloom Energy
种子 1 前 5：CTVA +1.420 Corteva、MRNA +0.899 Moderna、SWKS +0.836 Skyworks Solutions、ILMN +0.606 Illumina, Inc.、FICO +0.396 Fair Isaac
种子 2 前 5：MRNA +0.156 Moderna、ILMN +0.134 Illumina, Inc.、LII +0.112 Lennox International、CIEN +0.110 Ciena、CRL +0.098 Charles River Laboratories

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| ILMN | Illumina, Inc. | 136.77 | +0.399 | +0.606 | +0.134 |
| MRNA | Moderna | 187.46 | +0.518 | +0.899 | +0.156 |

累计涨跌幅：

| 日期 | 持有 | ILMN | MRNA | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|
| 2026-10-06 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-07 | 1 | -2.11% | +4.81% | +1.35% | -0.22% | +1.57% |


### 2026-10-07

种子 0 前 5：MRNA +0.530 Moderna、SMCI +0.518 Supermicro、SWKS +0.446 Skyworks Solutions、P +0.440 Everpure、COHR +0.432 Coherent Corp.
种子 1 前 5：CTVA +0.978 Corteva、SWKS +0.872 Skyworks Solutions、SMCI +0.735 Supermicro、MRNA +0.715 Moderna、COHR +0.626 Coherent Corp.
种子 2 前 5：P +0.131 Everpure、MRNA +0.113 Moderna、VST +0.098 Vistra Corp.、SMCI +0.095 Supermicro、NEM +0.086 Newmont

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| MRNA | Moderna | 196.48 | +0.530 | +0.715 | +0.113 |
| SMCI | Supermicro | 44.94 | +0.518 | +0.735 | +0.095 |

累计涨跌幅：

| 日期 | 持有 | MRNA | SMCI | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|
| 2026-10-07 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |


## 20 日模型

### 总表

| 信号日 | 名单 | 已持有 | 5 日组合 | 5 日大盘 | 5 日超额 | 10 日组合 | 10 日大盘 | 10 日超额 | 20 日组合 | 20 日大盘 | 20 日超额 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-01 | COHR、MRNA | 4 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-02 | COIN、MRNA、P | 3 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-05 | ILMN、MRNA | 2 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-06 | MRNA | 1 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-07 | MRNA、P、SMCI | 0 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |


### 2026-10-01

2026-10-02 记下的第一笔，依据 2026-10-01 美股收盘。

种子 0 前 5：MRNA +0.224 Moderna、SWKS +0.190 Skyworks Solutions、HPE +0.187 Hewlett Packard Enterprise、ACN +0.179 Accenture、COHR +0.174 Coherent Corp.
种子 1 前 5：HPE +0.280 Hewlett Packard Enterprise、COHR +0.277 Coherent Corp.、LITE +0.264 Lumentum、FICO +0.262 Fair Isaac、MRNA +0.262 Moderna
种子 2 前 5：MRNA +0.132 Moderna、COHR +0.113 Coherent Corp.、LITE +0.113 Lumentum、CIEN +0.112 Ciena、P +0.106 Everpure

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| COHR | Coherent Corp. | 319.19 | +0.174 | +0.277 | +0.113 |
| MRNA | Moderna | 188.94 | +0.224 | +0.262 | +0.132 |

累计涨跌幅：

| 日期 | 持有 | COHR | MRNA | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|
| 2026-10-01 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-02 | 1 | +5.59% | +0.57% | +3.08% | +0.73% | +2.35% |
| 2026-10-05 | 2 | +4.53% | +7.55% | +6.04% | +1.40% | +4.64% |
| 2026-10-06 | 3 | +6.01% | -0.78% | +2.61% | +1.99% | +0.62% |
| 2026-10-07 | 4 | +4.82% | +3.99% | +4.40% | +1.77% | +2.64% |


### 2026-10-02

种子 0 前 5：MRNA +0.212 Moderna、P +0.199 Everpure、COIN +0.172 Coinbase、FICO +0.165 Fair Isaac、SMCI +0.164 Supermicro
种子 1 前 5：MRNA +0.301 Moderna、FICO +0.278 Fair Isaac、P +0.253 Everpure、BE +0.244 Bloom Energy、COIN +0.244 Coinbase
种子 2 前 5：MRNA +0.127 Moderna、P +0.127 Everpure、COIN +0.094 Coinbase、BR +0.087 Broadridge Financial Solutions、NEM +0.087 Newmont

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| COIN | Coinbase | 183.00 | +0.172 | +0.244 | +0.094 |
| MRNA | Moderna | 190.01 | +0.212 | +0.301 | +0.127 |
| P | Everpure | 140.14 | +0.199 | +0.253 | +0.127 |

累计涨跌幅：

| 日期 | 持有 | COIN | MRNA | P | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|
| 2026-10-02 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-05 | 1 | +2.85% | +6.95% | +2.67% | +4.16% | +0.66% | +3.49% |
| 2026-10-06 | 2 | +1.50% | -1.34% | +5.04% | +1.73% | +1.25% | +0.49% |
| 2026-10-07 | 3 | -2.49% | +3.41% | +8.93% | +3.28% | +1.02% | +2.26% |


### 2026-10-05

种子 0 前 5：CTVA +0.250 Corteva、MRNA +0.197 Moderna、ILMN +0.176 Illumina, Inc.、P +0.152 Everpure、NEM +0.148 Newmont
种子 1 前 5：CTVA +0.333 Corteva、MRNA +0.264 Moderna、ILMN +0.234 Illumina, Inc.、LITE +0.229 Lumentum、FICO +0.203 Fair Isaac
种子 2 前 5：MRNA +0.124 Moderna、ILMN +0.112 Illumina, Inc.、FICO +0.108 Fair Isaac、NEM +0.087 Newmont、P +0.087 Everpure

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| ILMN | Illumina, Inc. | 146.84 | +0.176 | +0.234 | +0.112 |
| MRNA | Moderna | 203.21 | +0.197 | +0.264 | +0.124 |

累计涨跌幅：

| 日期 | 持有 | ILMN | MRNA | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|
| 2026-10-05 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-06 | 1 | -6.86% | -7.75% | -7.31% | +0.58% | -7.88% |
| 2026-10-07 | 2 | -8.83% | -3.31% | -6.07% | +0.36% | -6.43% |


### 2026-10-06

种子 0 前 5：CTVA +0.246 Corteva、MRNA +0.246 Moderna、SWKS +0.176 Skyworks Solutions、FICO +0.158 Fair Isaac、P +0.152 Everpure
种子 1 前 5：CTVA +0.333 Corteva、MRNA +0.284 Moderna、ILMN +0.269 Illumina, Inc.、BE +0.244 Bloom Energy、GNRC +0.227 Generac
种子 2 前 5：MRNA +0.162 Moderna、ILMN +0.152 Illumina, Inc.、RVTY +0.118 Revvity、CRL +0.113 Charles River Laboratories、PTC +0.106 PTC Inc.

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| MRNA | Moderna | 187.46 | +0.246 | +0.284 | +0.162 |

累计涨跌幅：

| 日期 | 持有 | MRNA | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|
| 2026-10-06 | 0 | +0.00% | +0.00% | +0.00% | +0.00% |
| 2026-10-07 | 1 | +4.81% | +4.81% | -0.22% | +5.03% |


### 2026-10-07

种子 0 前 5：CTVA +0.250 Corteva、MRNA +0.224 Moderna、SMCI +0.209 Supermicro、P +0.199 Everpure、SWKS +0.193 Skyworks Solutions
种子 1 前 5：SMCI +0.309 Supermicro、MRNA +0.284 Moderna、COHR +0.262 Coherent Corp.、P +0.259 Everpure、CTVA +0.248 Corteva
种子 2 前 5：MRNA +0.127 Moderna、P +0.127 Everpure、SMCI +0.127 Supermicro、PTC +0.106 PTC Inc.、ILMN +0.097 Illumina, Inc.

入场：

| 代码 | 名称 | 入场收盘 | 种子 0 | 种子 1 | 种子 2 |
|---|---|---:|---:|---:|---:|
| MRNA | Moderna | 196.48 | +0.224 | +0.284 | +0.127 |
| P | Everpure | 152.66 | +0.199 | +0.259 | +0.127 |
| SMCI | Supermicro | 44.94 | +0.209 | +0.309 | +0.127 |

累计涨跌幅：

| 日期 | 持有 | MRNA | P | SMCI | 组合 | 大盘 | 超额 |
|---|---|---|---|---|---|---|---|
| 2026-10-07 | 0 | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% |

