from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score
import config
from torch import Tensor
import math

class ModelExt(ABC):

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

    @abstractmethod
    def inference(self, **kwargs):
        pass

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)
    
class MyTransformer(nn.Module):
    def initialize_weights(self):
        for m in self.modules():
            # 判断是否属于Conv2d
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                # 判断是否有偏置
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data,0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1) 		 
                m.bias.data.zeros_()
    
    
    def __init__(self, seq_len, input_dim) -> None:
        super().__init__()
        embed_dim = 32
        n_clz = 3
        self.postion_enc = PositionalEncoding(d_model=embed_dim, max_len=seq_len)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=embed_dim*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.classifier = nn.Linear(embed_dim, n_clz)
        self.initialize_weights()

    def forward(self, x):
        out = self.input_proj(x)
        out = self.postion_enc(out)
        out = self.transformer_encoder(out)
        out = self.classifier(out[:,0, :])
        return out


class MyLSTM(nn.Module, ModelExt):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout_prob,
                 seq_length,
                 directions=1,
                 is_classification=False,
                 n_classes=3,
                 use_bceloss=False,
                 device=config.DEVICE,
                 weight=None):
        super(MyLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.directions = directions
        self.is_classification = is_classification
        self.use_bceloss = use_bceloss
        self.seq_length = seq_length
        self.device = device

        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            dropout=dropout_prob,
                            bidirectional=(directions == 2))
        self.linear = nn.Linear(hidden_size * self.directions, 1 if not is_classification else n_classes)
        self.sigmoid = nn.Sigmoid()

        if weight:
            self.load_model(path=weight)

    # def init_hidden_states(self, batch_size):
    #     _state_dim = (self.num_layers * self.directions, batch_size, self.hidden_size)
    #     return torch.zeros(_state_dim).to(self.device), torch.zeros(_state_dim).to(self.device)

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

    def save_model(self, epoch, min_val_loss, opt, path="./model_lstm.pt"):
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
        checkpoint = torch.load(path, map_location=self.device)
        min_val_loss = checkpoint["min_val_loss"]
        self.load_state_dict(checkpoint["model_state"])
        if opt is not None:
            opt.load_state_dict(checkpoint["opt_state"])
        return opt, checkpoint["epoch"], min_val_loss

    def train_model(
            self,
            training_dl,
            validation_dl,
            optimizer,
            criterion,
            scheduler,
            epochs,
            batch_size,
            validate_batch_size,
            validate_every_n_epoch
    ):
        training_losses = []
        validation_losses = []
        min_validation_loss = np.Inf

        for epoch in tqdm(range(epochs), desc="epoch"):
            # Set to train mode
            self.train()

            running_training_loss = 0.0

            # Begin training
            for idx, (x_batch, y_batch) in enumerate(tqdm(training_dl, desc="training")):
                # Convert to Tensors
                x_batch = x_batch.float().to(self.device)
                if self.is_classification and not self.use_bceloss:
                    y_batch = y_batch.long().to(self.device)
                else:
                    y_batch = y_batch.float().to(self.device)

                # Make prediction
                output = self(x_batch)
                # Calculate loss
                loss = criterion(output, y_batch)
                # print(f"training: {F.softmax(output, dim=1) if not USE_BCELOSS else output, y_batch}")
                optimizer.zero_grad()
                loss.backward()
                running_training_loss += loss.item()

                # torch.nn.utils.clip_grad_norm_(self.parameters(), 20)
                optimizer.step()
            scheduler.step()

            # Average loss across timesteps
            training_losses.append(running_training_loss / len(training_dl))

            if epoch % validate_every_n_epoch == 0:
                y_trues = []
                y_predicts = []
                y_confidences = []
                self.eval()
                with torch.no_grad():

                    running_validation_loss = 0.0

                    for idx, (x_batch, y_batch) in enumerate(tqdm(validation_dl, desc="validation")):
                        # Convert to Tensors
                        x_batch = x_batch.float().to(self.device)
                        if self.is_classification and not self.use_bceloss:
                            y_batch = y_batch.long().to(self.device)
                        else:
                            y_batch = y_batch.float().to(self.device)
                        # validation_states = self.init_hidden_states(validate_batch_size)
                        # validation_states = [state.detach() for state in validation_states]
                        output = self(x_batch)
                        validation_loss = criterion(output, y_batch)

                        output_softmax = torch.nn.Softmax(dim=1)(output)

                        max_idx = torch.argmax(output).item()
                        # TODO batch_size != 1
                        y_predicts.append(max_idx)
                        y_confidences.append(output_softmax[0][max_idx].item())
                        y_trues.append(y_batch.item())
                        # print(f"validation: {F.softmax(output, dim=1) if not USE_BCELOSS else output, y_batch}")
                        running_validation_loss += validation_loss.item()

                cm = confusion_matrix(y_trues, y_predicts)
                print(f"confusion_matrix: {cm}")
                print(classification_report(y_trues, y_predicts))

                # TODO 不同confidence下的PR
                cur_val_loss = running_validation_loss / len(validation_dl)
                validation_losses.append(cur_val_loss)
                print(f"valid loss: {cur_val_loss}")
                is_best = (cur_val_loss < min_validation_loss)

                if is_best:
                    min_validation_loss = cur_val_loss
                    self.save_model(epoch + 1, min_validation_loss, optimizer, f"./checkpoints/mylstm_best_{epoch}.pt")

            cur_train_loss = running_training_loss / len(training_dl)
            self.save_model(epoch + 1, cur_train_loss, optimizer, f"./checkpoints/mylstm_epoch_{epoch}.pt")
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

    def test_model(
            self,
            data: pd.DataFrame,
            threshold,
            head_label,
            head_features,
    ):
        self.eval()
        _data = data.copy()
        _data2 = _data.reset_index()
        _data[head_label] = 0
        for idx in tqdm(range(len(_data) - self.seq_length)):

            x = _data[idx:idx + self.seq_length][head_features].values
            x[:, 0:4] /= x[0][0]
            x[:, 4] /= x[0][4]
            x[:, 5] /= x[0][5]
            x[:, 0:6] -= 1
            # y = _data.iloc[idx + seq_length - 1][head_label]

            x = torch.tensor(x).unsqueeze(dim=0).float()
            y_hat, conf = self.inference(x, threshold)

            _data.loc[_data2.iloc[idx + self.seq_length - 1]["trade_date"], head_label] = y_hat * conf

        return _data

    def export_model_jit(self, pt_path, jit_path="model.torchscript"):
        self.load_model(pt_path)
        self.eval()
        # TODO
        input_ = torch.randn((1, 64, 4)).float().to(self.device)
        states = self.init_hidden_states(batch_size=1)
        ts = torch.jit.trace(self, (input_, states))
        ts.save(jit_path)

    def inference(self, input_data, threshold=None):
        """
        (1, seq_len, n_features)
        """
        input_data = input_data.to(self.device)
        # states = self.init_hidden_states(batch_size=1)
        out_data = self(input_data)
        if self.is_classification:
            if not self.use_bceloss:
                out_data = F.softmax(out_data, dim=1)
            max_idx = torch.argmax(out_data).item()  # 0, 1, 2
            conf = out_data[0][max_idx].item()
            # TODO fix below
            if threshold is None:
                return {1: 1, 2: -1, 0: 0}.get(max_idx), conf
            if max_idx == 1 and conf >= threshold:
                out_data = 1
            elif max_idx == 2 and conf >= threshold:
                out_data = -1
            else:
                out_data = 0
        else:
            # TODO
            out_data = out_data.item()
        return out_data, conf


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
    device = torch.device('cpu')
    model = torch.jit.load("model.torchscript")
    model.eval()
    input_ = torch.randn((1, 32, 7)).float().to(device)
    state_dim = (2 * 1, 1, 16)
    h, c = torch.zeros(state_dim).to(device), torch.zeros(state_dim).to(device)
    output = model(input_, (h, c))
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
