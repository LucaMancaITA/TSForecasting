
# Import modules
import os
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import ForecastingDataset
from utils.utils_train import adjust_learning_rate


class BaseModel:
    """Base model."""
    def __init__(self, data_args, train_eval_args):
        """Class initialization.

        Args:
            data_args (dict): data argument.
            train_eval_args (dict): train and eval arguments.
        """
        self.data_args = data_args
        self.train_eval_args = train_eval_args

    def build_model(self):
        """This metod must be specialized."""
        raise NotImplementedError

    def count_parameters(self, model):
        """Count model parameters.

        Args:
            model (object): pytorch model.

        Returns:
            int: num. of model parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_dataset(self, flag):
        """Set dataset,

        Args:
            flag (str): data type (train, pred, test).

        Returns:
            Dataset, DataLoader: pytorch dataset and dataloader.
        """
        Data = ForecastingDataset
        timeenc = self.data_args.timeenc

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = self.train_eval_args.batch_size
            freq=self.data_args.freq
        elif flag=='pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq=self.data_args.freq
            # Data = Dataset_Pred
        else:
            #shuffle_flag = True
            shuffle_flag = False
            drop_last = True
            batch_size = self.train_eval_args.batch_size
            freq=self.data_args.freq

        data_set = Data(
            root_path=self.data_args.root_path,
            data_path=self.data_args.data_path,
            flag=flag,
            size=[self.data_args.input_len,
                  self.data_args.label_len,
                  self.data_args.pred_len],
            features=self.data_args.features,
            target=self.data_args.target,
            timeenc=timeenc,
            freq=freq,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last)

        return data_set, data_loader

    def optimizer(self):
        """Set training optimizer.

        Returns:
            object: pytorch optimizer.
        """
        model_optim = torch.optim.Adam(
            self.model.parameters(), lr=self.train_eval_args.learning_rate)
        return model_optim

    def loss_function(self):
        """Compute loss function.

        Returns:
            float: loss function.
        """
        return nn.MSELoss()

    def process_one_batch(self):
        """This metod must be specialized."""
        raise NotImplementedError

    def training(self):
        """Training loop."""
        _, train_loader = self.get_dataset(flag = 'train')
        _, vali_loader = self.get_dataset(flag = 'val')
        _, test_loader = self.get_dataset(flag = 'test')

        optimizer = self.optimizer()
        loss_fn = self.loss_function()
        num_epochs = self.train_eval_args.epochs
        train_steps = len(train_loader)

        path = os.path.join(
            self.train_eval_args.checkpoints, self.train_eval_args.model_name)

        if not os.path.exists(path):
            os.makedirs(path)
        time_now = time.time()

        if self.train_eval_args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Tensorboard
        writer = SummaryWriter(f"runs/{self.train_eval_args.model_name}")

        for epoch in range(num_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                iter_count += 1

                optimizer.zero_grad()
                pred, true = self.process_one_batch(
                    batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = loss_fn(pred, true)
                train_loss.append(loss.item())

                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed * ((num_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.train_eval_args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)

            # Validation and testing
            vali_loss = self.validation(vali_loader, loss_fn)
            test_loss = self.validation(test_loader, loss_fn)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/vali", vali_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            # TODO
            # Early stopping

            # Adjust learning rate
            adjust_learning_rate(optimizer, epoch+1, self.train_eval_args.learning_rate)

        # Save model checkpoint
        model_path = os.path.join(path, 'checkpoint.pth')
        torch.save(self.model.state_dict(), model_path)

    def validation(self, vali_loader, loss_fn):
        """Model validation.

        Args:
            vali_loader (object): validation dataloader.
            loss_fn (object): loss function.

        Returns:
            float: validation loss.
        """
        self.model.eval()
        total_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            pred, true = self.process_one_batch(
                batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = loss_fn(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()

        return total_loss

    def testing(self):
        pass

    def predict(self):
        pass
