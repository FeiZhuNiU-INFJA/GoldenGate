
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader
from pathlib import Path
import tqdm
import glob
import sys

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[1]  # ninja_pro
# print(FILE.parents[1])
#
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to
from strategy.model import MyLSTM
from data.annotation import BuySellPointAnnotation
from data.dataset import SingleSymbolDataset


if __name__ == '__main__':
    device = torch.device("cpu")
    # HEADER_TARGET = "Y"
    SEQ_LENGTH = 64
    ANNOTATION_CLASS = BuySellPointAnnotation
    IS_CLASSIFICATION = True
    BATCH_SIZE = 128
    # NUM_FEATURES = 7  # TODO
    HIDDEN_SIZE = 16
    NUM_LAYERS = 2
    DROPOUT = 0.1
    DIRECTIONS = 2
    # N_CLASSES = 3
    LEARNING_RATE = 0.0005
    RESUME = False
    EPOCHS = 10

    TRAIN_START_DATE = "20000101"
    TRAIN_END_DATE = "20211231"
    VAL_START_DATE = "20220101"
    VAL_END_DATE = "20220630"

    FEATURES_HEAD = ["open", "high", "low", "close"]
    LABEL_HEAD = "BuySellPoint_qc_0.2_sp_0.02"

    train_dl_params = {'batch_size': BATCH_SIZE,
                       'shuffle': True,  # TODO
                       'drop_last': True,  # Disregard last incomplete batch
                       'num_workers': 8}

    val_dl_params = {'batch_size': 1,
                     'shuffle': False,
                     'drop_last': False,
                     'num_workers': 8}

    training_datasets = []
    validation_datasets = []

    for f_sse in tqdm.tqdm(glob.glob("/Users/yulin/workspace/extreme_quant/dataset/stock_sse/history/*.csv"),
                           desc="loading sse data"):
        training_datasets.append(
            SingleSymbolDataset(
                f_hist=f_sse,
                label_head=LABEL_HEAD,
                features_head=FEATURES_HEAD,
                start_date=TRAIN_START_DATE,
                end_date=TRAIN_END_DATE,
                seq_len=SEQ_LENGTH,
                is_classification=True,
                is_one_hot_label=False,
                n_classes=BuySellPointAnnotation.n_class(),
                func_label_to_class=BuySellPointAnnotation.label_to_class,
            )
        )
        validation_datasets.append(
            SingleSymbolDataset(
                f_hist=f_sse,
                label_head=LABEL_HEAD,
                features_head=FEATURES_HEAD,
                start_date=VAL_START_DATE,
                end_date=VAL_END_DATE,
                seq_len=SEQ_LENGTH,
                is_classification=True,
                is_one_hot_label=False,
                n_classes=BuySellPointAnnotation.n_class(),
                func_label_to_class=BuySellPointAnnotation.label_to_class,
            )
        )

    for f_szse in tqdm.tqdm(glob.glob("/Users/yulin/workspace/extreme_quant/dataset/stock_szse/history/*.csv"),
                            desc="loading szse data"):
        training_datasets.append(
            SingleSymbolDataset(
                f_hist=f_szse,
                label_head=LABEL_HEAD,
                features_head=FEATURES_HEAD,
                start_date=TRAIN_START_DATE,
                end_date=TRAIN_END_DATE,
                seq_len=SEQ_LENGTH,
                is_classification=True,
                is_one_hot_label=False,
                n_classes=BuySellPointAnnotation.n_class(),
                func_label_to_class=BuySellPointAnnotation.label_to_class,
            )
        )
        validation_datasets.append(
            SingleSymbolDataset(
                f_hist=f_szse,
                label_head=LABEL_HEAD,
                features_head=FEATURES_HEAD,
                start_date=VAL_START_DATE,
                end_date=VAL_END_DATE,
                seq_len=SEQ_LENGTH,
                is_classification=True,
                is_one_hot_label=False,
                n_classes=BuySellPointAnnotation.n_class(),
                func_label_to_class=BuySellPointAnnotation.label_to_class,
            )
        )

    training_dl = DataLoader(ConcatDataset(training_datasets), pin_memory=True, **train_dl_params)
    validation_dl = DataLoader(ConcatDataset(validation_datasets), pin_memory=True, **val_dl_params)

    model = MyLSTM(
        input_size=len(FEATURES_HEAD),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_prob=DROPOUT,
        directions=DIRECTIONS,
        is_classification=IS_CLASSIFICATION,
        n_classes=ANNOTATION_CLASS.n_class(),
        use_bceloss=False,
        device=device,
        model_name="LSTM202301",
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    if IS_CLASSIFICATION:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 25., 25.]).float()).to(device)
    else:
        criterion = nn.MSELoss().to(device)

    model.train_model(
        training_dl=training_dl,
        validation_dl=validation_dl,
        optimizer=optimizer,
        criterion=criterion,
        epochs=EPOCHS,
        batch_size=train_dl_params["batch_size"],
        validate_batch_size=val_dl_params["batch_size"],
        validate_every_n_epoch=1
    )
