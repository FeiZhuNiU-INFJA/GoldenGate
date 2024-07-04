## 环境安装
```py
conda create -n extreme python=3.10 -y
conda activate extreme
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

```

## 下载数据
```py
python data/initialization.py 
```
## 生成标签
```py
python data/annotation.py 
```
##### dataset  数据保存的地方，永久的标签可以存在这里


