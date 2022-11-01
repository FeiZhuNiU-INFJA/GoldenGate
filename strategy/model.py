from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.functional as F
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device('cpu')


class ModelExt(metaclass=ABCMeta):

    @abstractmethod
    def save_model(self, **kwargs):
        pass

    @abstractmethod
    def load_model(self, **kwargs):
        pass

    @abstractmethod
    def train_model(self, **kwargs):
        pass

    @abstractmethod
    def test_model(self, **kwargs):
        pass


class MyLSTM(nn.Module, ModelExt):

    def __init__(self,
                 input_size, hidden_size, num_layers, dropout_prob,
                 directions=1,
                 is_classification=False,
                 n_classes=3,
                 use_bceloss=False,
                 resume=False,
                 model_name=None):
        super(MyLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.directions = directions
        self.is_classification = is_classification
        self.use_bceloss = use_bceloss
        self.resume = resume
        self.model_name = model_name

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob,
                            bidirectional=(directions == 2))
        self.linear = nn.Linear(hidden_size * self.directions, 1 if not is_classification else n_classes)
        self.sigmoid = nn.Sigmoid()

        # state_dim = (self.num_layers * self.directions, 1, self.hidden_size)
        # self.init_h = nn.Parameter(torch.zeros(state_dim))
        # self.init_c = nn.Parameter(torch.zeros(state_dim))

    def init_hidden_states(self, batch_size):
        state_dim = (self.num_layers * self.directions, batch_size, self.hidden_size)
        return torch.zeros(state_dim).to(device), torch.zeros(state_dim).to(device)

    def forward(self, x, states=None):
        x, (h, c) = self.lstm(x, states)
        out = self.linear(x[:, -1, :])
        if self.is_classification:
            if self.use_bceloss:
                ret = self.sigmoid(out)
            else:
                ret = out
            return ret
        else:
            return out

    def save_model(self, epoch, min_val_loss, opt, path="./model_state.pt"):
        print(f"New minimum reached at epoch #{epoch + 1}, saving model state...")
        checkpoint = {
            'epoch': epoch + 1,
            'min_val_loss': min_val_loss,
            'model_state': self.state_dict(),
            'opt_state': opt.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_model(self, path, opt=None):
        # load check point
        checkpoint = torch.load(path)
        min_val_loss = checkpoint["min_val_loss"]
        self.load_state_dict(checkpoint["model_state"])
        if opt is not None:
            opt.load_state_dict(checkpoint["opt_state"])
        return opt, checkpoint["epoch"], min_val_loss

    def train_model(self, training_dl, validation_dl, optimizer, criterion,
                    epochs, batch_size, validate_batch_size,
                    is_classification, use_bceloss,
                    validate_every):
        if self.resume and self.model_name:
            self.load_model(self.model_name)
        training_losses = []
        validation_losses = []
        y_trues = []
        y_predicts = []
        y_confidences = []
        min_validation_loss = np.Inf

        # Set to train mode
        self.train()

        for epoch in tqdm(range(epochs)):

            running_training_loss = 0.0
            states = self.init_hidden_states(batch_size)
            # Begin training
            for idx, (x_batch, y_batch) in enumerate(tqdm(training_dl)):
                # Convert to Tensors
                x_batch = x_batch.float().to(device)
                if is_classification and not use_bceloss:
                    y_batch = y_batch.long().to(device)
                else:
                    y_batch = y_batch.float().to(device)

                states = [state.detach() for state in states]
                optimizer.zero_grad()

                # Make prediction
                output = self(x_batch, states)
                # Calculate loss
                loss = criterion(output, y_batch)
                # print(f"training: {F.softmax(output, dim=1) if not USE_BCELOSS else output, y_batch}")
                loss.backward()
                running_training_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 20)
                optimizer.step()

            # Average loss across timesteps
            training_losses.append(running_training_loss / len(training_dl))

            if epoch % validate_every == 0:

                # Set to eval mode
                self.eval()
                validation_states = self.init_hidden_states(validate_batch_size)
                running_validation_loss = 0.0

                for idx, (x_batch, y_batch) in enumerate(tqdm(validation_dl)):
                    # Convert to Tensors
                    x_batch = x_batch.float().to(device)
                    if is_classification and not use_bceloss:
                        y_batch = y_batch.long().to(device)
                    else:
                        y_batch = y_batch.float().to(device)
                    validation_states = [state.detach() for state in validation_states]
                    output = self(x_batch, validation_states)
                    validation_loss = criterion(output, y_batch)

                    output_softmax = torch.nn.Softmax(dim=1)(output)

                    max_idx = torch.argmax(output).item()
                    # TODO batch_size!=1
                    y_predicts.append(max_idx)
                    y_confidences.append(output_softmax[0][max_idx].item())
                    y_trues.append(y_batch.item())
                    # print(f"validation: {F.softmax(output, dim=1) if not USE_BCELOSS else output, y_batch}")
                    running_validation_loss += validation_loss.item()

                # TODO 不同confidence下的PR
                cm = confusion_matrix(y_trues, y_predicts)
                print(f"confusion_matrix: {cm}")
                print(classification_report(y_trues, y_predicts))
                cur_val_loss = running_validation_loss / len(validation_dl)
                validation_losses.append(cur_val_loss)
                print(f"valid loss: {cur_val_loss}")
                is_best = (cur_val_loss < min_validation_loss)

                if is_best:
                    min_validation_loss = cur_val_loss
                    self.save_model(epoch + 1, min_validation_loss, optimizer, f"./{self.model_name}_best.pt")
                # Reset to training mode
                self.train()

            cur_train_loss = running_training_loss / len(training_dl)
            self.save_model(epoch + 1, cur_train_loss, optimizer, f"./{self.model_name}_last.pt")
            print(f"train loss: {cur_train_loss}")

        # Visualize loss
        epoch_count = range(1, len(training_losses) + 1)
        plt.plot(epoch_count, training_losses, 'r--')
        plt.legend(['Training Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        val_epoch_count = range(1, len(validation_losses) + 1)
        plt.plot(val_epoch_count, validation_losses, 'b--')
        plt.legend(['Validation loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def test_model(self, path, data: pd.DataFrame, seq_length, threshold=0.9,
                   is_classification=True, use_bceloss=False):
        TARGET = "Y"

        self.load_model(path)
        self.eval()
        _data = data.copy()
        _data.iloc[0:seq_length][TARGET] = 0
        _data2 = _data.reset_index()
        for idx in range(len(_data) - seq_length):
            x = _data[idx:idx + seq_length].values
            y = _data.iloc[idx + seq_length - 1][TARGET]

            x = torch.tensor(x).unsqueeze(dim=0).float().to(device)
            states = self.init_hidden_states(batch_size=1)
            output = self(x, states)
            if is_classification:

                if not use_bceloss:
                    output = F.softmax(output, dim=1)
                    # output = torch.exp(output)

                max_idx = torch.argmax(output).item()  # 0, 1, 2
                conf = output[0][max_idx].item()
                if max_idx == 1 and conf >= threshold:
                    output = -1
                elif max_idx == 2 and conf >= threshold:
                    output = 1
                else:
                    output = 0
            else:
                output = output.item()
            # print(output)
            print(y, output)
            _data.loc[_data2.iloc[idx + seq_length - 1]["Date"], TARGET] = output
            # _data.iat[idx + SEQ_LENGTH - 1, TARGET] = 111
        print(_data[TARGET].value_counts())
        return _data

    def export_model_jit(self, pt_path, jit_path="model.torchscript"):
        self.load_model(pt_path)
        self.eval()
        input_ = torch.randn((1, 32, 7)).float().to(device)
        states = self.init_hidden_states(1)
        ts = torch.jit.trace(self, (input_, states))
        ts.save(jit_path)


if __name__ == '__main__':

    # model = MyLSTM(
    #     input_size=7,
    #     hidden_size=16,
    #     num_layers=2,
    #     dropout_prob=0.05,
    #     directions=1,
    #     is_classification=True,
    #     n_classes=3,
    # ).to(device)
    #
    # model.export_model_jit("None_last.pt")


    model = torch.jit.load("model.torchscript")
    model.eval()
    input_ = torch.randn((1, 32, 7)).float().to(device)
    state_dim = (2 * 1, 1, 16)
    h,c = torch.zeros(state_dim).to(device), torch.zeros(state_dim).to(device)
    output = model(input_, (h,c))
    print(output)
    output1 = torch.nn.Softmax(dim=1)(output)
    print(output1)
    print(output1[0][1].item())
    # pass
    # device = torch.device("mps")
    # EPOCHS = 10
    # DROPOUT = 0.05
    # DIRECTIONS = 1
    # NUM_LAYERS = 2
    # BATCH_SIZE = 32
    # # OUTPUT_SIZE = 1
    # SEQ_LENGTH = 256
    # HIDDEN_SIZE = 32
    # LEARNING_RATE = 0.0005
    # IS_CLASSIFICATION = True
    # N_CLASSES = 3
    # USE_BCELOSS = False
    # IS_TRAIN = False
    # USE_BERT = True
    # RESUME = True
    # RESUME_PATH = "./bert_256_last.pt"
    # TEST_MODEL = "./bert_256_last.pt"
    # VALIDATE_EVERY = 2
    # n_symbols = 50
    # symbols = [][0:n_symbols]
    # validate_symbol = "AAPL"
    # start_date = "2005-01-01"
    # end_date = "2022-09-09"
    #
    # MODEL_NAME = f"{'bert' if USE_BERT else 'lstm'}_{SEQ_LENGTH}"
