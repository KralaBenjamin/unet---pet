# Augmentation Random Horizontal Flip? Resize, Rotation, Normalize, <= nachdem Training steht
# Welche verwende ich bei Bonirop? Color Jitter? 

from unet import UNET
from dataset_pet import get_train_val_file_list, PetOnlySegmentationDataSet
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import Resize, CenterCrop
from torchvision.transforms.functional import crop
import mlflow

def main():
    DEVICE = 'cpu'
    EPOCHS = 5
    ONLY_OVERFIT = False

    model = UNET(3, 1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()

    print(f'{mlflow.get_tracking_url()=}')

    image_path = 'data/pets/images'
    seg_path = 'data/pets/annotations/trimaps'

    (train_image_list, train_seg_list), (val_image_list, val_seg_list) = \
        get_train_val_file_list(image_path, seg_path)
    
    transform = CenterCrop((256, 256))
    
    train_dataset = PetOnlySegmentationDataSet(
        train_image_list, train_seg_list, transform=transform)
    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=128)

    val_dataset = PetOnlySegmentationDataSet(val_image_list, val_seg_list)
    val_dl = DataLoader(val_dataset, batch_size=128)

    if ONLY_OVERFIT:
            
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
        exit()


    with mlflow.start_run():    

        for i_epoch in range(EPOCHS):
            for batch_x, batch_y in tqdm(train_dl):
                pred = model(batch_x)

                loss = criterion(pred, batch_y.unsqueeze(dim=0))
                loss.backward()

                mlflow.log_metric('loss', loss)

                optimizer.step()
                optimizer.zero_grad()

            accuracy_values = list()
            for batch_x, batch_y in tqdm(val_dl):
                breakpoint() #<= Größe?
                batch_size = batch_x.shape[0]

                pred = model(batch_x)
                pred_log = (pred > 0.5).float()
                acc  = (pred_log == over_y).sum() / (pred.shape[-2] * pred.shape[-1])
                accuracy_values.append((batch_size, acc))

            sum_batch = sum([
                batch_size for batch_size, _ in accuracy_values
            ])

            mean_all = sum([
                acc * batch_size / sum_batch
                for batch_size, acc in accuracy_values
            ])

            mlflow.log_metric(
                'accuracy_mean', mean_all
            )

if __name__ == '__main__':
    main()
