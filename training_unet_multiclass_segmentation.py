from os import makedirs
from pathlib import Path
from json import dump

from unet import UNET
from dataset_pet import get_train_val_file_list, PetMulticlassSegmentationSet


import torch
from torcheval.metrics.functional import multiclass_f1_score, multilabel_accuracy
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, CenterCrop
from torchvision.transforms.functional import crop
import wandb


def main():
    DEVICE = "cuda"
    EPOCHS = 100
    ONLY_OVERFIT = False
    BATCH_SIZE = 32
    MULTICLASS_TARGET = 'race'# , race, animal

    wandb.init(
        project="unet-pet", 
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

    val_dataset = PetMulticlassSegmentationSet(
        image_desc_file,
        val_image_list, 
        val_seg_list, 
        transform=transform, 
        target_class=MULTICLASS_TARGET
    )
    val_dl = DataLoader(val_dataset, batch_size=1)

    n_class = train_dataset.get_n_classes() 

    model = UNET(3, n_class).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()

    if ONLY_OVERFIT:
        for over_x, over_y in train_dataset:
            break

        over_x = over_x.to(DEVICE).unsqueeze(dim=0)
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
        print(pred_log.shape, over_y.shape)
        summe = (pred_log == over_y).sum()
        print(summe / (pred_log.shape[-3] * pred.shape[-2] * pred.shape[-1]))
    list_statedict = list()
    one_element = False
    for i_epoch in range(EPOCHS):
        for batch_x, batch_y in train_dl:
            #if one_element:
            #    break
            one_element = True
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

        accuracy_values = list()
        iou_values = list()

        list_y_true_most_common_class = list()
        list_pred_most_common_class = list()
        for batch_x, batch_y in val_dl:
            model.eval()
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            batch_size = batch_x.shape[0]

            with torch.no_grad():
                pred = model(batch_x)
                pred_bool = (pred > 0.5)
                pred_log = pred_bool.float()


                def get_most_predicted_class(batch):
                    # reduces the  to a number
                    batch_argmax = torch.argmax(batch, dim=0)
                    

                    batch_argmax = torch.flatten(
                        batch_argmax
                    )

                    mask = batch_argmax > 0
                    if not torch.all(mask):
                        return 0
                    # iteration, damit der Filter beachtet wird.
                    batch_class = torch.mode(
                        batch_argmax[mask]
                        ).values
                    return batch_class

                #multiclass prediction metrics

                batch_y_common_class = get_most_predicted_class(batch_y[0])
                pred_common_class = get_most_predicted_class(pred[0])

                list_y_true_most_common_class.append(batch_y_common_class)
                list_pred_most_common_class.append(pred_common_class)


                # wie gut ist die Segmentierung?

                # wie homogen ist die Vorhersage?
                
                #iou = torch.logical_and(pred_bool, batch_y).sum() / torch.logical_or(pred_bool, batch_y).sum()
                #acc  = (pred_log == batch_y).sum() / (batch_size * pred.shape[-2] * pred.shape[-1])
                
                #accuracy_values.append((batch_size, acc))
                #iou_values.append([batch_size, iou])

        #multiclass prediction metrics
        
        tensor_y_true_most_common_class = torch.tensor(list_y_true_most_common_class)
        tensor_pred_most_common_class = torch.tensor(list_pred_most_common_class)
        
        f1 = multiclass_f1_score(tensor_pred_most_common_class, tensor_y_true_most_common_class)
        acc = (tensor_y_true_most_common_class == tensor_pred_most_common_class).sum() / tensor_pred_most_common_class.shape[0]
        
        """    
            sum_batch = sum([
                batch_size for batch_size, _ in accuracy_values
            ])

            mean_iou = sum([
                iou * batch_size / sum_batch
                for batch_size, iou in iou_values
            ])

            mean_acc = sum([
                acc * batch_size / sum_batch
                for batch_size, acc in accuracy_values
            ])
        """
        wandb.log(
            {
            'f1': f1,
            'acc': acc
            }
        )

        #list_statedict.append((mean_iou.item(), model.state_dict().copy()))  

        #print(f"{mean_acc=}")
        #print(f"{mean_iou=}")

        continue

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
