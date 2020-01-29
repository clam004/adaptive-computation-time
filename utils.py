import pathlib

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data

def bool_to_idx(idx):
    return idx.nonzero().squeeze(1)

def maybe_cuda_var(x, cuda):
    """Helper for converting to a Variable"""
    x = Variable(x)
    if cuda:
        x = x.cuda()
    return x

def train(config, model, data_manager, train_epoch, test_epoch):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.num_epochs + 1):
        train_data_loader = data_manager.create_dataloader(config)
        test_data_loader = data_manager.create_dataloader(config, mode="test")
        train_epoch(
            epoch=epoch, config=config, model=model,
            data_loader=train_data_loader,
            optimizer=optimizer,
        )
        test_result = test_epoch(
            config=config, model=model,
            data_loader=test_data_loader,
        )
        print(
            'Epoch: {}, Average loss: {:.4f}, '
            'Accuracy: {}/{} ({:.0f}%), PT: {}'.format(
                epoch,
                test_result["loss"], test_result["num_correct"],
                test_result["length"],
                test_result["num_correct"] * 100 / test_result["length"],
                f'{test_result["mean_ponder_time"]:.1f}'
                if test_result["mean_ponder_time"] else "N/A",
            )
        )
        if epoch % config.model_save_interval == 0:
            model_save_path = pathlib.Path(config.model_save_path)
            model_save_path.mkdir(exist_ok=True, parents=True)
            model_save_file_path = (
                model_save_path / f"epoch_{epoch}.pt"
            )
            print(f"Saving checkpoint to {model_save_file_path}")
            torch.save(model.state_dict(), model_save_file_path)
            

def test_epoch(config, model, data_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0

    #loss_func = nn.BCEWithLogitsLoss(reduction="sum")
    loss_func = nn.BCEWithLogitsLoss()

    ponder_times_ls = []
    for batch_idx, (x, y) in enumerate(data_loader):
        x_var = maybe_cuda_var(x, cuda=config.cuda)
        y_var = y
        if config.cuda:
            y_var = y_var.cuda()

        y_hat, ponder_dict = model(x_var, compute_ponder_cost=False)
        test_loss += loss_func(y_hat, y_var).item()
        y_pred = (y_hat.data > 0.5).float()
        correct += y_pred.eq(y_var.data).cpu().numpy()\
            .reshape(y.shape[0], -1).all(axis=1).sum()

        if ponder_dict:
            ponder_times_ls.append(np.array(ponder_dict["ponder_times"]).T)

    test_loss /= len(data_loader.dataset)

    if ponder_times_ls:
        mean_ponder_time = np.mean(np.vstack(ponder_times_ls))
    else:
        mean_ponder_time = None
    print(
        'Epoch: {}, Average loss: {:.4f}, '
        'Accuracy: {}/{} ({:.0f}%), PT: {}'.format(
            epoch,
            test_loss,  correct,
            len(data_loader.dataset),
            correct * 100 / len(data_loader.dataset),
            f'{mean_ponder_time:.1f}'
            if mean_ponder_time else "N/A",
        )
    )
    
    if epoch % config.model_save_interval == 0:
        model_save_path = pathlib.Path(config.model_save_path)
        model_save_path.mkdir(exist_ok=True, parents=True)
        model_save_file_path = (
            model_save_path / f"epoch_{epoch}.pt"
        )
        print(f"Saving checkpoint to {model_save_file_path}")
        torch.save(model.state_dict(), model_save_file_path)
        
    return {
        "loss": test_loss,
        "num_correct": correct,
        "length": len(data_loader.dataset),
        "mean_ponder_time": mean_ponder_time,
    }

class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, *data_list):
        assert len(data_list) > 0
        self.data_length = len(data_list[0])
        for data in data_list[1:]:
            assert len(data) == self.data_length
        self.data_list = data_list

    def __getitem__(self, index):
        return [
            data[index]
            for data in self.data_list
        ]

    def __len__(self):
        return self.data_length


class DataManager:
    @classmethod
    def create_data(cls, *args, **kwargs):
    #def create_data(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _get_length(cls, config):
        raise NotImplementedError

    @classmethod
    def _get_dataloader(cls, data, batch_size):
    #def _get_dataloader(self, data, batch_size):
        data_x, data_y = data
        return torch.utils.data.DataLoader(
            MultiDataset(data_x, data_y),
            batch_size=batch_size,
            shuffle=True,
        )

    @classmethod
    def create_dataloader(cls, config, mode="train"):
        length = cls._get_length(config)
    #@classmethod
    #def create_dataloader(self, config, mode="train"):
        length = cls._get_length(config)
        input_length = config.input_length
        if mode == "train":
            pass
        elif mode == "test":
            length = int(config.test_percentage * length)
        else:
            raise KeyError(mode)
        data = cls.create_data(length=length, input_length=input_length)
        return cls._get_dataloader(data=data, batch_size=config.batch_size)

class ParityDataManager(DataManager):
    
    @classmethod
    def create_data(cls, length, input_length):
        parity_x = np.random.randint(2, size=(length, input_length)).astype(
            np.float32) * 2 - 1
        zero_out = np.random.randint(1, input_length, size=length)
        for i in range(length):
            parity_x[i, zero_out[i]:] = 0.
        parity_y = (np.sum(parity_x == 1, axis=1) % 2).astype(np.float32)
        parity_x = np.expand_dims(parity_x, 1)
        return parity_x, parity_y

    @classmethod
    def _get_length(cls,config):
        return 16*10
        #return config.parity_data_len