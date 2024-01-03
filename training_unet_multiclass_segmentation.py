from os import makedirs
from pathlib import Path
from json import dump
from pickle import dump as pickledump

from unet import UNET
from dataset_pet import get_train_val_file_list, PetMulticlassSegmentationSet


import torch
from torcheval.metrics.functional import multiclass_f1_score, multilabel_accuracy
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop
import wandb
from loguru import logger


def get_most_predicted_class(batch):
    # reduces the  to a number
    batch_argmax = torch.argmax(batch, dim=0)
    batch_argmax = torch.flatten(
        batch_argmax
    )

    if torch.all(batch_argmax == 0):
        return 0
    
    mask = batch_argmax > 0
    # iteration, damit der Filter beachtet wird.
    batch_class = torch.mode(
        batch_argmax[mask]
        ).values
    return batch_class

def get_segmentation_bool_matric(batch):
    return torch.argmax(batch[0], dim=0) != 0


# this metric evaluates, 
def get_homogenity_predication(batch):
    batch_argmax = torch.argmax(batch, dim=0)
    if batch_argmax[batch_argmax > 0].numel() == 0:
        return 0.0
    _, n_uniques = torch.unique(batch_argmax[batch_argmax > 0], return_counts=True)

    return torch.max(n_uniques) / torch.sum(n_uniques)


def main():
    DEVICE = "cuda"
    EPOCHS = 100
    ONLY_OVERFIT = False
    BATCH_SIZE = 32
    MULTICLASS_TARGET = 'race'# , race, animal

    wandb.init(
        project="unet-pet-multiclass", 
        config={
            "batch_size": BATCH_SIZE, 
            "epochs": EPOCHS,
            "multiclass_target": MULTICLASS_TARGET,
        })

    EXPERIMENT_NAME = wandb.run.name
    PATH_MODEL_DIRS = Path(f"./models/{EXPERIMENT_NAME}")
    PATH_MODEL = PATH_MODEL_DIRS / "best_model.pth"
    print(f"{EXPERIMENT_NAME=}")

    image_path = "/data/images"
    seg_path = "/data/annotations/trimaps"
    image_desc_file = "/data/annotations/list.txt"

    (train_image_list, train_seg_list), (
        val_image_list,
        val_seg_list,
    ) = get_train_val_file_list(image_path, seg_path)    
    transform = CenterCrop((256, 256))

    train_dataset = PetMulticlassSegmentationSet(
        image_desc_file,
        train_image_list, 
        train_seg_list, 
        transform=transform, 
        target_class=MULTICLASS_TARGET
    )
    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    class_information_train = train_dataset.get_class_information() #so labels do have the same meaning

    val_dataset = PetMulticlassSegmentationSet(
        image_desc_file,
        val_image_list, 
        val_seg_list, 
        target_class=MULTICLASS_TARGET,
        ClassInformation=class_information_train
    )
    val_dl = DataLoader(val_dataset, batch_size=1)

    n_class = train_dataset.get_n_classes() 

    model = UNET(3, n_class).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    softie = torch.nn.Softmax(dim=1)

    if ONLY_OVERFIT:
        for over_x, over_y in train_dataset:
            break

        over_x = over_x.to(DEVICE).unsqueeze(dim=0)
        over_y = over_y.unsqueeze(dim=0).to(DEVICE)

        for i in range(100):
            print(i)
            pred = model(over_x)

            over_y_arg = torch.argmax(over_y, dim=1)

            loss = criterion(pred, over_y_arg)
            loss.backward()

            print(f'True{loss=}')

            optimizer.step()
            optimizer.zero_grad()

        pred = model(over_x)
        predsof = softie(pred)
        pred_log = (predsof > 0.5).float()
        print(pred_log.shape, over_y.shape)
        summe1 = (pred_log == over_y).sum()
        print(summe1 / (pred_log.shape[-3] * pred.shape[-2] * pred.shape[-1]))
        summe2 = (torch.argmax(pred, dim=1) == torch.argmax(over_y, dim=1)).sum()
        print(summe2 / (pred.shape[-2] * pred.shape[-1]))

        segmentation_batch_y = get_segmentation_bool_matric(over_y)
        segmentation_pred = get_segmentation_bool_matric(pred)

        exit()
    list_statedict = list()
    for i_epoch in range(EPOCHS):
        for batch_x, batch_y in train_dl:

            model.train()
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            pred = model(batch_x)

            loss = criterion(pred, batch_y)
            loss.backward()

            wandb.log({"loss": loss})

            optimizer.step()
            optimizer.zero_grad()


            
        #validation

        iou_values = list()
        homogenity_values = list()

        list_y_true_most_common_class = list()
        list_pred_most_common_class = list()
        for batch_x, batch_y in val_dl:
            model.eval()
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            with torch.no_grad():
                # pred_log bauen
                pred = model(batch_x)
                pred_soft = softie(pred)
                pred_bool = pred_soft > 0.5
                pred_log = pred_bool.float()

                #multiclass prediction metrics

                batch_y_common_class = get_most_predicted_class(batch_y[0])
                pred_common_class = get_most_predicted_class(pred[0])

                list_y_true_most_common_class.append(batch_y_common_class)
                list_pred_most_common_class.append(pred_common_class)
                
                #segmentation metrics
                
                segmentation_batch_y = get_segmentation_bool_matric(batch_y)
                segmentation_pred = get_segmentation_bool_matric(pred)

                iou = torch.logical_and(
                    segmentation_pred, 
                    segmentation_batch_y).sum() / torch.logical_or(
                        segmentation_pred, segmentation_batch_y).sum()

                iou_values.append(iou)


                homo = get_homogenity_predication(pred[0])
                homogenity_values.append(homo)



        #multiclass prediction metrics
        
        tensor_y_true_most_common_class = torch.tensor(list_y_true_most_common_class)
        tensor_pred_most_common_class = torch.tensor(list_pred_most_common_class)
        
        f1 = multiclass_f1_score(
            tensor_pred_most_common_class, 
            tensor_y_true_most_common_class,
            num_classes=38, average='weighted')
        acc = (tensor_y_true_most_common_class == tensor_pred_most_common_class).sum() / tensor_pred_most_common_class.shape[0]
        
        mean_iou = sum(iou_values) / len(iou_values)
        mean_homogenity = sum(homogenity_values) / len(homogenity_values)
        print(f"{mean_iou=} \t {mean_homogenity=} \t {f1=}")
        value_dict = {
            'f1': f1,
            'acc': acc,
            'mean_iou': mean_iou,
            'homo': mean_homogenity,
            }
        state_dict = value_dict | {'state': model.state_dict().copy()}
        list_statedict.append(state_dict)

        wandb.log(
            value_dict
        )



    wandb.finish()
    makedirs(PATH_MODEL_DIRS)

    pickledump(
        train_dataset.get_class_information(),
        open(PATH_MODEL_DIRS / "class_information.pickle", "wb+")
    )

    dump(
        train_dataset.get_json_list(category=MULTICLASS_TARGET),
        open(PATH_MODEL_DIRS / "train_data_files.json", "w+"),
    )


    dump(
        val_dataset.get_json_list(category=MULTICLASS_TARGET),
        open(PATH_MODEL_DIRS / "val_data_files.json", "w+"),
    )


    for metric in value_dict.keys():
        PATH_MODEL = PATH_MODEL_DIRS / f"model_best_{metric}.pth"
        best_model_sd = max(list_statedict, key=lambda x: x[metric])['state']
        torch.save(best_model_sd, PATH_MODEL)


if __name__ == "__main__":
    main()
