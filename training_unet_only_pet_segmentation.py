# Augmentation Random Horizontal Flip? Resize, Rotation, Normalize, <= nachdem Training steht
# Welche verwende ich bei Bonirop? Color Jitter? 

from unet import UNET
from dataset_pet import get_train_val_file_list, PetOnlySegmentationDataSet
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import Resize, CenterCrop
from torchvision.transforms.functional import crop

def main():
    DEVICE = 'cpu'
    EPOCHS = 5

    model = UNET(3, 1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()

    image_path = 'data/pets/images'
    seg_path = 'data/pets/annotations/trimaps'

    (train_image_list, train_seg_list), (val_image_list, val_seg_list) = \
        get_train_val_file_list(image_path, seg_path)
    
    transform = CenterCrop((256, 256))
    
    train_dataset = PetOnlySegmentationDataSet(
        train_image_list, train_seg_list, transform=transform)
    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=16)

    val_dataset = PetOnlySegmentationDataSet(val_image_list, val_seg_list)
    val_dl = DataLoader(val_dataset)

    
    for over_x, over_y in train_dataset:
        break

    over_x = over_x
    over_y = over_y.unsqueeze(dim=0)
    for i in range(10):
        print(i)
        pred = model(over_x)

        loss = criterion(pred, over_y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    pred = model(over_x)
    pred_log = (pred > 0.5).float()
    summe  = (pred_log == over_y).sum()
    print(summe / (pred.shape[-2] * pred.shape[-1]))
    # todo: berechnung fetirg machen
    

    exit()


    n_trained_batches = 0
    for i_epoch in range(EPOCHS):
        for batch_x, batch_y in tqdm(train_dl):
            pred = model(batch_x[0])

            loss = criterion(pred[0], batch_y)
            loss.backward()

            if n_trained_batches >= 128:

                optimizer.step()
                optimizer.zero_grad()
            else:
                n_trained_batches += 1


if __name__ == '__main__':
    main()
