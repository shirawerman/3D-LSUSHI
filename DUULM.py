from Models import duulm_net
import torch #noqa
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from Data_Handler import Dataset
import numpy as np
import random
import os
from PIL import Image
from Metrics import create_manual_loss
from Utils import load_data, split_tensor, norm_tensor, rebuild_tensor, log_compression
from scipy.io import savemat
from datetime import datetime


class DUULM:

    def __init__(self, conf):
        self.conf = conf
        self.model = duulm_net(scale_factor=conf.scale_factor,
                               folds=conf.num_folds,
                               kernel_size=[conf.kernel1, conf.kernel2],
                               num_channels=conf.num_channels,
                               interp_mode='nearest').to(conf.device)
        self.writer = SummaryWriter()

    def run(self):
        print('** Start training')
        self.train()
        print('** Done training.')
        self.writer.close()
   
    def prepare_test_data(self, test_path):
        test_var = load_data(test_path).to(self.conf.device)
        test_var = norm_tensor(test_var)

        return test_var.unsqueeze(0).unsqueeze(0)

    def test(self, model_path=None):
        """
        :param test_path: path to ultrasound sequence to super resolve
        :param model: path to saved model
        """
        print("start test")
        test_paths = self.conf.test_path
        for test_path in test_paths:
            if not os.path.exists(test_path):
                print(f"test path {test_path} does not exist")
                return

            test_var = self.prepare_test_data(test_path)
            resolved_var = self.test_inner(model_path, test_var).squeeze()

            save_name = test_path.split("/")[-1]

            mdic = {f'resolved_{save_name.split(".")[0]}': np.array(resolved_var.detach().cpu())}
            save_path = os.path.join(self.conf.result_path, f'resolved_{save_name}')
            savemat(save_path, mdic)

    def test_inner(self, model_path, test_var):
        if model_path is not None:
            print(f"loading model from: {model_path}")
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_sd'])
        self.model.eval()

        resolved_var = self.model(test_var)
        return resolved_var


    def train(self):
        # Ensure reproducibility
        torch.manual_seed(self.conf.seed)
        random.seed(self.conf.seed)
        np.random.seed(self.conf.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

        def seed_worker(worker_id):
            worker_seed = 2509  # torch.initial_seed()
            random.seed(worker_seed)
            np.random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.conf.seed)

        # Parameters
        params = {'batch_size': self.conf.train_batch_size,
                  'shuffle': True,
                  'num_workers': self.conf.num_workers
                  'worker_init_fn': seed_worker,
                  'generator': g}

        # Generators
        training_set = Dataset(self.conf.train_batch_size, self.conf,self.conf.train_dataset)
        training_generator = torch.utils.data.DataLoader(training_set, **params)

        params['batch_size'] = self.conf.val_batch_size
        val_dataset = os.path.join(self.conf.val_dataset)
        validation_set = Dataset(self.conf.val_batch_size, self.conf, val_dataset, self.conf.val_num_batches)
        validation_generator = torch.utils.data.DataLoader(validation_set, **params)

        # Define loss
        criterion = create_manual_loss(self.conf)
        loss_sig = self.conf.starting_loss_sig

        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.learning_rate,
                                     betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        # load saved model
        start_epoch = 0
        if len(self.conf.model_path) > 0:
            checkpoint = torch.load(self.conf.model_path)
            print(f"loading saved model at: {self.conf.model_path}")
            self.model.load_state_dict(checkpoint['model_sd'])
            optimizer.load_state_dict(checkpoint['optim_sd'])
            start_epoch = checkpoint['epoch'] + 1
            loss_sig = checkpoint['loss_sig']

        min_valid_loss = np.inf

        # Loop over epochs
        for epoch in range(start_epoch, self.conf.num_epochs):
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f"start epoch {epoch} at {current_time}")

            if epoch > 0 and epoch % self.conf.loss_sig_interval == 0:
                loss_sig = loss_sig - loss_sig/10
                # reinitialize optimizer
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf.learning_rate,
                                             betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

            # Training
            total_inputs = 0.0
            running_loss = 0.0
            for (local_inputs, local_targets) in training_generator:
                # Transfer to GPU
                local_inputs, local_targets = local_inputs.unsqueeze(1).to(self.conf.device), local_targets.unsqueeze(1).to(self.conf.device)
                # Zero grad
                optimizer.zero_grad()
                # Forward pass
                local_outputs = self.model(local_inputs)
                # Calculate loss
                loss = criterion(local_outputs, local_targets, sig=loss_sig)
                # Backward pass
                loss.backward()
                # Optimization step
                optimizer.step()
                total_inputs += local_targets.size(0)
                running_loss += loss
                print(f'data loader, epoch [{epoch}/{self.conf.num_epochs}]: Loss: {loss}')

            print(f'Epoch [{epoch}/{self.conf.num_epochs}]: Running Loss: {(running_loss / total_inputs)}')

            # Validation
            if epoch % self.conf.val_interval == 0 or epoch == self.conf.num_epochs - 1:
                with torch.set_grad_enabled(False):
                    num_inputs = 0.0
                    running_val_losses = 0.0
                    for local_inputs, local_targets in validation_generator:
                        # Transfer to GPU
                        local_inputs, local_targets = local_inputs.unsqueeze(1).to(self.conf.device), local_targets.unsqueeze(1).to(self.conf.device)
                        # Forward pass
                        local_outputs = self.model(local_inputs)
                        # Calculate loss
                        num_inputs += local_targets.size(0)
                        val_loss = criterion(local_outputs, local_targets, sig=loss_sig)
                        running_val_losses += val_loss

                self.writer.add_scalar("Loss/train", running_loss / total_inputs, epoch)
                self.writer.add_scalar("Loss/val", running_val_losses / num_inputs, epoch)


                is_loss_minimal = min_valid_loss > running_val_losses / num_inputs
                if is_loss_minimal:
                    print(f'Validation Loss Decreased({min_valid_loss}--->{running_val_losses/num_inputs}')
                    min_valid_loss = running_val_losses/num_inputs

                print('Saving The Model')
                # Saving State Dict
                state = {'epoch': epoch,
                         'model_sd': self.model.state_dict(),
                         'optim_sd': optimizer.state_dict(),
                         'loss': loss,
                         'loss_sig': loss_sig}
                torch.save(state, os.path.join('model_cp', f'model_cp_{epoch + 1}.pth'))
