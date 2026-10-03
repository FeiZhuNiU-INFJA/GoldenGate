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
| 2026-10-01 | COHR、FICO、MRNA | 1 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-02 | FICO、MRNA、P | 0 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |


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


## 10 日模型

### 总表

| 信号日 | 名单 | 已持有 | 5 日组合 | 5 日大盘 | 5 日超额 | 10 日组合 | 10 日大盘 | 10 日超额 | 20 日组合 | 20 日大盘 | 20 日超额 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-01 | COHR、LITE、MRNA | 1 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-02 | 空仓 | — | — | — | — | — | — | — | — | — | — |


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


### 2026-10-02

种子 0 前 5：FICO +0.569 Fair Isaac、MRNA +0.449 Moderna、HPE +0.374 Hewlett Packard Enterprise、P +0.374 Everpure、COHR +0.368 Coherent Corp.
种子 1 前 5：FICO +0.982 Fair Isaac、HPE +0.709 Hewlett Packard Enterprise、COIN +0.635 Coinbase、COHR +0.593 Coherent Corp.、SWKS +0.449 Skyworks Solutions
种子 2 前 5：MRNA +0.131 Moderna、P +0.131 Everpure、SMCI +0.095 Supermicro、FDS +0.088 FactSet、BR +0.086 Broadridge Financial Solutions

这一天三个模型的前 5 名没有交集，不建仓。


## 20 日模型

### 总表

| 信号日 | 名单 | 已持有 | 5 日组合 | 5 日大盘 | 5 日超额 | 10 日组合 | 10 日大盘 | 10 日超额 | 20 日组合 | 20 日大盘 | 20 日超额 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-01 | COHR、MRNA | 1 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |
| 2026-10-02 | COIN、MRNA、P | 0 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 | 未到期 |


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

