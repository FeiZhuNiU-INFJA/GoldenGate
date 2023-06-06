
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader
import tqdm
import glob
from torch.optim.lr_scheduler import StepLR
from strategy.model import MyTransformer
from data.dataset import SingleSymbolDataset
from config import BASE_FEATURES, DIR_DATA_HIST_CN, ACCELERATOR, LOGGER, DEVICE
from hubs import anno1, anno2
import numpy as np


if __name__ == '__main__':
    # device = torch.device("cuda:0")
    SEQ_LENGTH = 128
    BATCH_SIZE = 640
    LEARNING_RATE = 1e-3
    RESUME = False
    EPOCHS = 100

    TRAIN_START_DATE = "20000101"
    TRAIN_END_DATE = "20221231"
    VAL_START_DATE = "20220101"
    VAL_END_DATE = "20230101"

    train_dl_params = {'batch_size': BATCH_SIZE,
                       'shuffle': True,  # TODO
                       'drop_last': True,  # Disregard last incomplete batch
                       'num_workers': 16}

    val_dl_params = {'batch_size': 64,
                     'shuffle': False,
                     'drop_last': False,
                     'num_workers': 16}

    training_datasets = []
    validation_datasets = []

    for f_hist_csv in tqdm.tqdm(list(glob.glob(f"{DIR_DATA_HIST_CN}/*.csv"))[:],
                                desc="loading hist data", disable=not ACCELERATOR.is_main_process):
        training_datasets.append(
            SingleSymbolDataset(
                f_hist=f_hist_csv,
                anno=anno1,
                features_head=BASE_FEATURES,
                start_date=TRAIN_START_DATE,
                end_date=TRAIN_END_DATE,
                seq_len=SEQ_LENGTH,
                with_aug=True,
            )
        )
        validation_datasets.append(
            SingleSymbolDataset(
                f_hist=f_hist_csv,
                anno=anno1,
                features_head=BASE_FEATURES,
                start_date=TRAIN_START_DATE,
                end_date=TRAIN_END_DATE,
                seq_len=SEQ_LENGTH,
                with_aug=False,
            )
        )
    validation_datasets = validation_datasets[::40]
    training_dl = DataLoader(ConcatDataset(training_datasets), pin_memory=True, **train_dl_params)
    validation_dl = DataLoader(ConcatDataset(validation_datasets), pin_memory=True, **val_dl_params)
    (f"training data: {len(training_dl) * BATCH_SIZE}, validation data: {len(validation_dl)}")

    model = MyTransformer(input_dim=len(BASE_FEATURES), seq_len=SEQ_LENGTH).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 80., 80.]).float()).to(DEVICE)
    
    training_losses = []
    validation_losses = []
    min_validation_loss = np.Inf

    model, optimizer, scheduler, training_dl = ACCELERATOR.prepare(model, optimizer, scheduler, training_dl)

    for epoch in tqdm.tqdm(range(EPOCHS), desc="epoch"):
        # Set to train mode
        model.train()

        running_training_loss = 0.0

        # Begin training
        for idx, (x_batch, y_batch) in enumerate(tqdm.tqdm(training_dl, desc="training", disable=not ACCELERATOR.is_main_process)):
            # Convert to Tensors
            x_batch = x_batch.float().to(DEVICE)
            y_batch = y_batch.long().to(DEVICE)
            # Make prediction
            output = model(x_batch)
            # Calculate loss
            loss = criterion(output, y_batch)
            # LOGGER.info(f"loss: {loss.item()}", main_process_only=True)
            optimizer.zero_grad()
            # loss.backward()
            ACCELERATOR.backward(loss)
            running_training_loss += loss.item()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        scheduler.step()

        # Average loss across timesteps
        # training_losses.append(running_training_loss / len(training_dl))

        y_trues = []
        y_predicts = []
        y_confidences = []
        model.eval()
        with torch.no_grad():

            running_validation_loss = 0.0

            for idx, (x_batch, y_batch) in enumerate(tqdm.tqdm(validation_dl, desc="validation", disable=not ACCELERATOR.is_main_process)):
                # Convert to Tensors
                x_batch = x_batch.float().to(DEVICE)
                y_batch = y_batch.long().to(DEVICE)
                # validation_states = self.init_hidden_states(validate_batch_size)
                # validation_states = [state.detach() for state in validation_states]
                output = model(x_batch)
                validation_loss = criterion(output, y_batch)

                output_softmax = torch.nn.Softmax(dim=1)(output)

                y_predicts.extend(torch.argmax(output, dim=1).tolist())
                y_confidences.extend(torch.max(output_softmax, dim=1)[0].tolist())
                y_trues.extend(y_batch.tolist())
                running_validation_loss += validation_loss.item()

        cm = confusion_matrix(y_trues, y_predicts)
        LOGGER.info(f"confusion_matrix: {cm}", main_process_only=True)
        LOGGER.info(classification_report(y_trues, y_predicts), main_process_only=True)

        # TODO 不同confidence下的PR
        cur_val_loss = running_validation_loss / len(validation_dl)
        # validation_losses.append(cur_val_loss)
        LOGGER.info(f"valid loss: {cur_val_loss}", main_process_only=True)
        is_best = (cur_val_loss < min_validation_loss)

        if is_best:
            min_validation_loss = cur_val_loss
            torch.save(ACCELERATOR.unwrap_model(model).state_dict(), f"./checkpoints/mytransformer_best_{epoch}.pt")

        cur_train_loss = running_training_loss / len(training_dl)
        torch.save(ACCELERATOR.unwrap_model(model).state_dict(), f"./checkpoints/mytransformer_epoch_{epoch}.pt")
        LOGGER.info(f"train loss: {cur_train_loss}", main_process_only=True)


    
