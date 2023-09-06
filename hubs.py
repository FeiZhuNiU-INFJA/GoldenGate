from config import DEVICE
from strategy.model import MyLSTM, MyTransformer
from data.annotation import BuySellPointAnnotation
import torch

# 标签
anno1 = BuySellPointAnnotation(quote_change=0.2, soft_percent=0.02, soft_eta=0.9, min_gap=5)
anno2 = BuySellPointAnnotation(quote_change=0.5, soft_percent=0.03, soft_eta=0.9, min_gap=5)
# anno3 = BuySellPointAnnotation(quote_change=0.5, soft_percent=0.03, soft_eta=0.9, min_gap=5)
anno3 = BuySellPointAnnotation(quote_change=0.1, soft_percent=0.01, soft_eta=0.9, min_gap=2)


# model1 = MyLSTM(
#     input_size=6,
#     hidden_size=16,
#     num_layers=2,
#     dropout_prob=0.1,
#     directions=2,
#     is_classification=True,
#     n_classes=3,
#     use_bceloss=False,
#     device=DEVICE,
#     seq_length=64,
#     weight="mylstm_64.pt",
#     # weight="mylstm_last.pt",
# ).to(DEVICE)
# model1.eval()


# model2 = MyLSTM(
#     input_size=6,
#     hidden_size=16,
#     num_layers=2,
#     dropout_prob=0.1,
#     directions=2,
#     is_classification=True,
#     n_classes=3,
#     use_bceloss=False,
#     device=DEVICE,
#     seq_length=128,
#     weight="mylstm_128.pt",
# ).to(DEVICE)
# model2.eval()

model3 = MyTransformer(seq_len=128, input_dim=8).to(DEVICE)  # 三分类模型， BASE_FEATURES = ["open", "high", "low", "close", 'vol', 'amount', 'turnover_rate', 'volume_ratio']
# model3.load_state_dict(torch.load("/data/home/eric/workspace/extreme_quant/checkpoints/mytransformer_best_17.pt", map_location=DEVICE))
model3.load_state_dict(torch.load("/data/home/eric/workspace/extreme_quant/checkpoints/mytransformer_epoch_7.pt", map_location=DEVICE))
model3.eval()