## 安装环境
```
pip install -r requirements.txt
```

## 下载数据、生成标签
```py
python data/initialization.py 
```

## 训练
```
accelerate config
accelerate launch training.py
```
## 输出某一天的买卖信号
1. 更新hubs.py中的模型路径
2. 修改portfolio2.py中的日期
```
python strategy/portfolio2.py
```


