from unet import UNET
from dataset_pet import get_train_val_file_list, PetOnlySegmentationDataSet
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, CenterCrop
from torchvision.transforms.functional import crop
import wandb
from os import makedirs
from pathlib import Path
from json import dump


def main():
    DEVICE = "cuda"
    EPOCHS = 30
    ONLY_OVERFIT = False
    BATCH_SIZE = 32
    MULTICLASS_TARGT = 'animal'# , 'race'

    wandb.init(
        project="unet-pet", 
        config={
            "batch_size": BATCH_SIZE, 
            "epochs": EPOCHS,
            "multiclass_target": MULTICLASS_TARGT,
        })

    EXPERIMENT_NAME = wandb.run.name
    PATH_MODEL_DIRS = Path(f"./models/{EXPERIMENT_NAME}")
    PATH_MODEL = PATH_MODEL_DIRS / "best_model.pth"
    print(f"{EXPERIMENT_NAME=}")


if __name__ == "__main__":
    main()
