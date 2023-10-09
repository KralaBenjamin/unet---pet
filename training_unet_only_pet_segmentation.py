# Augmentation Random Horizontal Flip? Resize, Rotation, Normalize, <= nachdem Training steht
# Welche verwende ich bei Bonirop? Color Jitter? 

from unet import UNET
from dataset_pet import get_train_val_file_list, PetOnlySegmentationDataSet
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    
    train_dataset = PetOnlySegmentationDataSet(train_image_list, train_seg_list)
    train_dl = DataLoader(train_dataset, shuffle=True)

    val_dataset = PetOnlySegmentationDataSet(val_image_list, val_seg_list)
    val_dl = DataLoader(val_dataset)


    """    
        over_x, over_y = train_dataset[0]
        for i in range(10):
            print(i)
            pred = model(over_x)

            loss = criterion(pred[0, 0], over_y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        pred = model(over_x)
        pred_log = (pred > 0.5).float()
        summe  = (pred_log == over_y).sum()
        breakpoint()
        print(summe / (pred.shape[0] * pred.shape[1]), pred.shape, over_y)
        # todo: berechnung fetirg machen
    """

    """
    TODO: UserWarning: The default value of the antialias parameter of all the 
    resizing transforms (Resize(), RandomResizedCrop(), etc.) will change from 
    None to True in v0.17, in order to be consistent across the PIL and Tensor 
    backends. To suppress this warning, directly pass antialias=True (recommended, 
    future default), antialias=None (current default, which means False for Tensors 
    and True for PIL), or antialias=False (only works on Tensors - PIL will still use 
    antialiasing). This also applies if you are using the inference transforms from the 
    models weights: update the call to weights.transforms(antialias=True).
    """
    """
    TODO: Croppe den Scheiß für mehr Effizienz!
    
    """

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
