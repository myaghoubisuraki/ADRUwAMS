import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
import time
from DataLoaders import get_dataloader
from Metrics import Meter
import matplotlib.pyplot as plt
import numpy as np
#########################
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

class Trainer:
    def __init__(self,
                 net: nn.Module,
                 dataset: torch.utils.data.Dataset,
                 criterion: nn.Module,
                 lr: float,
                 accumulation_steps: int,
                 batch_size: int,
                 fold: int,
                 num_epochs: int,
                 path_to_csv: str,
                 display_plot: bool = True,
                 pre_weights_dir='pretrained_weights',
                ):   
                
        self.pre_weights_dir = pre_weights_dir
        # Define the device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.display_plot = display_plot  
        
        
        # Define the model (network)
        self.net = net
        
        # Check the number of available GPUs
        num_gpus = torch.cuda.device_count()
        
        # If there's more than one GPU, use DataParallel
        if num_gpus > 1:
            device_ids = [i for i in range(num_gpus)]
            self.net = nn.DataParallel(self.net, device_ids=device_ids)
        
        self.net = self.net.to(self.device)

        # Define the loss function and move it to GPU
        self.criterion = criterion.to(self.device)
        
        # Define the optimizer
        self.optimizer = Adam(self.net.parameters(), lr=lr)

        # Define the learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=4, verbose=True)

        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["train", "val"]  # Different phases in training
        self.num_epochs = num_epochs  # Number of epochs

        # Dataloader for each phase
        self.dataloaders = {
            phase: get_dataloader(
                dataset = dataset,
                path_to_csv = path_to_csv,
                phase = phase,
                fold = fold,
                batch_size = batch_size,
                num_workers = 4,
                augmentations=augmentations,
            )
            for phase in self.phases
        }

        self.best_loss = float("inf")  # Initialize the best loss as infinity
        self.losses = {phase: [] for phase in self.phases}  # Track losses for each phase
        self.dice_scores = {phase: [] for phase in self.phases}  # Track dice scores for each phase
        

    def _compute_loss_and_outputs(self,
                                  images: torch.Tensor,
                                  targets: torch.Tensor):
        
        images = images.to(self.device)
        targets = targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch: int, phase: str):
        
        print(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}")

        self.net.train() if phase == "train" else self.net.eval()  
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()  

        for itr, data_batch in enumerate(dataloader):  
            images, targets = data_batch['image'], data_batch['mask']
            loss, _ = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()  
                if (itr + 1) % self.accumulation_steps == 0:  
                    self.optimizer.step()  
                    self.optimizer.zero_grad()  
            running_loss += loss.item()
        
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches  
        self.losses[phase].append(epoch_loss)  
        return epoch_loss
        
    def run(self):
        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val")
                self.scheduler.step(val_loss)

            # Save the best model
            if val_loss < self.best_loss:
                print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), "{self.pre_weights_dir}/_best_model_Unet_AttG_Res_multi.pth")

            # Save a temporary checkpoint every 10 epochs regardless of performance
            if epoch % 10 == 0:
                print(f"\n{'#'*20}\nSaved temp checkpoint\n{'#'*20}\n")
                torch.save(self.net.state_dict(), f"{self.pre_weights_dir}/temp_model_epoch_{epoch}_Unet_AttG_Res_multi.pth")

            print()

        self._save_train_history()

    def load_predtrain_model(self, state_path: str):
        self.net.load_state_dict(torch.load(state_path))
        print("Predtrain model loaded")
        
    def _save_train_history(self):
        torch.save(self.net.state_dict(), f"{self.pre_weights_dir}/_last_epoch_model_Unet_AttG_Res_multi.pth")

        pd.DataFrame({"train_loss": self.losses["train"], "val_loss": self.losses["val"]}).to_csv("__train_log.csv", index=False)
        
###########################################################

from scipy.ndimage import rotate
from scipy.ndimage import zoom
from scipy import ndimage

def hor_flip(img: np.ndarray, mask: np.ndarray):
    """
    flip horizontally.
    """
# Flip horizontally
    img = np.flip(img, axis=2)  # Flip along width
    mask = np.flip(mask, axis=2)
    return img, mask


augmentations = [
    hor_flip
]

