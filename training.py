
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader
import tqdm
import glob
from torch.optim.lr_scheduler import StepLR
from strategy.model import MyLSTM
from data.dataset import SingleSymbolDataset
from config import BASE_FEATURES, DIR_DATA_HIST_CN
from hubs import anno1, anno2

if __name__ == '__main__':
    device = torch.device("cpu")
    # HEADER_TARGET = "Y"
    SEQ_LENGTH = 128
    IS_CLASSIFICATION = True
    BATCH_SIZE = 64
    HIDDEN_SIZE = 16
    NUM_LAYERS = 2
    DROPOUT = 0.1
    DIRECTIONS = 2
    LEARNING_RATE = 0.0008
    RESUME = False
    EPOCHS = 30

    TRAIN_START_DATE = "20000101"
    TRAIN_END_DATE = "20221231"
    VAL_START_DATE = "20220101"
    VAL_END_DATE = "20230101"

    anno = anno1
    feature_head = BASE_FEATURES
    LABEL_HEAD = anno.head_label

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

    for f_hist_csv in tqdm.tqdm(list(glob.glob(f"{DIR_DATA_HIST_CN}/*.csv"))[:10],
                                desc="loading hist data"):
        training_datasets.append(
            SingleSymbolDataset(
                f_hist=f_hist_csv,
                label_head=LABEL_HEAD,
                features_head=feature_head,
                start_date=TRAIN_START_DATE,
                end_date=TRAIN_END_DATE,
                seq_len=SEQ_LENGTH,
                is_classification=True,
                is_one_hot_label=False,
                n_classes=anno.n_class(),
                func_label_to_class=anno.label_to_class,
                with_aug=True,
            )
        )
        validation_datasets.append(
            SingleSymbolDataset(
                f_hist=f_hist_csv,
                label_head=LABEL_HEAD,
                features_head=feature_head,
                start_date=VAL_START_DATE,
                end_date=VAL_END_DATE,
                seq_len=SEQ_LENGTH,
                is_classification=True,
                is_one_hot_label=False,
                n_classes=anno.n_class(),
                func_label_to_class=anno.label_to_class,
                with_aug=False,
            )
        )
    validation_datasets = validation_datasets[::40]
    training_dl = DataLoader(ConcatDataset(training_datasets), pin_memory=True, **train_dl_params)
    validation_dl = DataLoader(ConcatDataset(validation_datasets), pin_memory=True, **val_dl_params)
    print(f"training data: {len(training_dl) * BATCH_SIZE}, validation data: {len(validation_dl)}")

    model = MyLSTM(
        input_size=len(feature_head),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_prob=DROPOUT,
        directions=DIRECTIONS,
        is_classification=IS_CLASSIFICATION,
        n_classes=anno.n_class(),
        use_bceloss=False,
        device=device,
        seq_length=SEQ_LENGTH,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    if IS_CLASSIFICATION:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 80., 80.]).float()).to(device)
    else:
        criterion = nn.MSELoss().to(device)

    model.train_model(
        training_dl=training_dl,
        validation_dl=validation_dl,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=EPOCHS,
        batch_size=train_dl_params["batch_size"],
        validate_batch_size=val_dl_params["batch_size"],
        validate_every_n_epoch=1
    )
