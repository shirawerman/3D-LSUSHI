import torch
import glob
from Generator import create_lr_hr_pair


class Dataset(torch.utils.data.Dataset):
    def __init__(self, batch_size, conf, dataset_path, num_batches=None):
        self.conf = conf
        self.dataset_path = dataset_path
        self.inputs_list = glob.glob(f'{dataset_path}/*input*')
        self.batch_size = batch_size
        self.num_batches = num_batches if num_batches is not None else len(self.inputs_list) // batch_size

    def __len__(self):
        'Denotes the total number of samples'
        return self.batch_size * self.num_batches

    def __getitem__(self, index):
        'Generates one sample of data'
        input_name = self.inputs_list[index]
        input = torch.load(input_name).to(self.conf.device)
        target_name = input_name.replace("input", "target")
        target = torch.load(target_name).to(self.conf.device)

        return input, target
