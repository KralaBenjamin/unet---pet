# Augmentation Random Horizontal Flip? Resize, Rotation, Normalize, <= nachdem Training steht
# Welche verwende ich bei Bonirop? Color Jitter?

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

    wandb.init(project="unet-pet", config={"batch_size": BATCH_SIZE, "epochs": EPOCHS})

    EXPERIMENT_NAME = wandb.run.name
    PATH_MODEL_DIRS = Path(f"./models/{EXPERIMENT_NAME}")
    PATH_MODEL = PATH_MODEL_DIRS / "best_model.pth"
    print(f"{EXPERIMENT_NAME=}")

    model = UNET(3, 1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()

    image_path = "/data/images"
    seg_path = "/data/annotations/trimaps"

    (train_image_list, train_seg_list), (
        val_image_list,
        val_seg_list,
    ) = get_train_val_file_list(image_path, seg_path)

    transform = CenterCrop((256, 256))

    train_dataset = PetOnlySegmentationDataSet(
        train_image_list, train_seg_list, transform=transform
    )
    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    val_dataset = PetOnlySegmentationDataSet(
        val_image_list, val_seg_list, transform=transform
    )
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    if ONLY_OVERFIT:
        for over_x, over_y in train_dataset:
            break

        over_x = over_x.to(DEVICE)
        over_y = over_y.unsqueeze(dim=0).to(DEVICE)
        for i in range(10):
            print(i)
            pred = model(over_x)

            loss = criterion(pred, over_y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        pred = model(over_x)
        pred_log = (pred > 0.5).float()
        summe = (pred_log == over_y).sum()
        print(summe / (pred.shape[-2] * pred.shape[-1]))
        exit()

    list_statedict = list()

    for i_epoch in range(EPOCHS):
        for batch_x, batch_y in train_dl:
            model.train()
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            pred = model(batch_x)
            pred = pred[:, 0, :, :]

            loss = criterion(pred, batch_y)
            loss.backward()

            wandb.log({"loss": loss})

            optimizer.step()
            optimizer.zero_grad()

        accuracy_values = list()
        for batch_x, batch_y in val_dl:
            model.eval()
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            batch_size = batch_x.shape[0]

            with torch.no_grad():
                pred = model(batch_x)
                pred = pred[:, 0, :, :]
                pred_log = (pred > 0.5).float()
                breakpoint()
                acc = (pred_log == batch_y).sum() / (
                    batch_size * pred.shape[-2] * pred.shape[-1]
                )
                accuracy_values.append((batch_size, acc))

        sum_batch = sum([batch_size for batch_size, _ in accuracy_values])

        mean_all = sum(
            [acc * batch_size / sum_batch for batch_size, acc in accuracy_values]
        )

        wandb.log({"accuracy_mean": mean_all})

        list_statedict.append((mean_all.item(), model.state_dict().copy()))

        # todo: welche Metriken für Segmentierung?

        print(f"{mean_all=}")

    wandb.finish()
    makedirs(PATH_MODEL_DIRS)
    best_model_sd = max(list_statedict, key=lambda x: x[0])[1]
    torch.save(best_model_sd, PATH_MODEL)
    dump(
        list(zip(train_image_list, train_seg_list)),
        open(PATH_MODEL_DIRS / "train_data_files.json", "w+"),
    )
    dump(
        list(zip(val_image_list, val_seg_list)),
        open(PATH_MODEL_DIRS / "val_data_files.json", "w+"),
    )


if __name__ == "__main__":
    main()
