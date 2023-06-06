from config import DEVICE
from strategy.model import MyLSTM
from data.annotation import BuySellPointAnnotation


# 标签
anno1 = BuySellPointAnnotation(quote_change=0.2, soft_percent=0.02, soft_eta=0.9, min_gap=5)
anno2 = BuySellPointAnnotation(quote_change=0.5, soft_percent=0.03, soft_eta=0.9, min_gap=5)


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