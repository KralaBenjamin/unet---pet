import torch
import click
from pathlib import Path
from unet import UNET
from matplotlib import pyplot as plt


from PIL import Image
import numpy as np
from einops import rearrange
from datetime import datetime
from json import load as jsonload
from random import randint
import pickle
from loguru import logger

def get_image_file2category(ClassDescriptionFile):
    image_file2category = dict()
    with open(ClassDescriptionFile, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            image, class_image, species_image, breed_id = line.split(" ")
            image_file2category[image] = {
                "race": (int(species_image), int(breed_id)),
                "animal": int(species_image),
            }
    
    return image_file2category


@click.command()
@click.argument("experiment_name")
@click.option("--path_input_image", default=None, help="Pfad zum Input Bild.")
@click.option("--path_seg_image", default=None, help="Pfad zum Segmentation Bild.")
@click.option("--path_output_image", default=None, help="Pfad zum Output Bild.")
@click.option("--use-model-which-metric", default="f1", help="Bestes Modell aus welcher Metrik")
@click.option("--multi-class-target", default='race', help="Welches Ziel das Modell hat")
def main(
    experiment_name, 
    path_input_image, 
    path_seg_image, 
    path_output_image, 
    use_model_which_metric,
    multi_class_target):

    if multi_class_target == 'race':
        num_classes = 38
    elif multi_class_target == "animal":
        num_classes = 2
    else:
        logger.debug("kein valides Ziel für Multiclass")
        import sys; sys.exit()

    model_file_name = f"model_best_{use_model_which_metric}.pth"
    model_dir = Path(f"./models/{experiment_name}") 

    model = UNET(3, num_classes) # wie groß?
    model.load_state_dict(torch.load(model_dir / model_file_name))

    label_binariser, class_to_i = pickle.load(
        open(model_dir / "class_information.pickle", "br")
    )

    # lade bild
    list_path_val = jsonload(open(model_dir / "val_data_files.json", "r"))
    path_2_class = {
        image: image_class
        for image, _, image_class in list_path_val
    }

    if path_input_image is None:
        i_image = randint(0, len(list_path_val))
        path_input_image = list_path_val[i_image][0]
        path_seg_image = list_path_val[i_image][1]
        
    true_class_information = tuple(path_2_class[path_input_image])
    true_class = class_to_i[true_class_information]

    if path_output_image is None:
        dt = datetime.now()
        date_time_str = dt.strftime("%Y-%m-%d-%H-%M-%S")
        path_output_image = f"results/{date_time_str}.png"

    print(
        f"{experiment_name=} \t {path_input_image=} \t {path_seg_image=} \t {path_output_image=}"
    )

    image = np.array(Image.open(path_input_image).convert("RGB"))
    segmentation = np.array(Image.open(path_seg_image))

    #todo: Daten richtig laden

    # siehe Pet Readme, beides Hintergrund
    segmentation[segmentation == 2] = 0
    segmentation[segmentation == 3] = 0

    model.eval()

    image = torch.tensor(image)
    image_channel = rearrange(image, "H W C -> C H W")
    image_channel = image_channel.float().unsqueeze(dim=0)

    #todo: Daten in das richtige Format bringen

    with torch.no_grad():
        pred = model(image_channel)
        pred_arg_max = torch.argmax(pred, dim=1)

    pred_true_class = pred_arg_max == true_class

    # berechnet die top 5 Werte
    pred_values, pred_n_values = torch.unique(pred_arg_max, return_counts=True)
    indices_highest_values = torch.sort(pred_n_values).indices.flip(dims=[0])
    pred_values = pred_values[indices_highest_values]
    pred_n_values = pred_n_values[indices_highest_values]

    pred_arg_max_channel = rearrange(pred_arg_max, "C H W -> H W C")
    pred_arg_max_image = plt.cm.tab10(pred_arg_max_channel[:, :, 0])
    pred_arg_max_image = torch.tensor(pred_arg_max_image[:, :, :3])
    pred_arg_max_image = (pred_arg_max_image * 255).int()

    segmentation = torch.tensor(segmentation)
    segmentation_image = bool_tensor_into_rgb_tensor(segmentation)
    segmentation_image = rearrange(segmentation_image, "C H W -> H W C")
    
    comparision_image = bool_tensor_into_rgb_tensor(pred_true_class == segmentation)
    comparision_image = rearrange(comparision_image, "C H W -> H W C")
    
    list_outputs = [image, pred_arg_max_image, segmentation_image, comparision_image]
    
    combined_tensor = torch.cat(list_outputs, dim=1).int().numpy()
    combined_tensor = np.uint8(combined_tensor)
    plt.imsave(path_output_image, combined_tensor)




def bool_tensor_into_rgb_tensor(bool_img):
    rgb_img = bool_img.float() * 255
    if len(rgb_img.shape) == 2:
        rgb_img = rgb_img.unsqueeze(dim=0)
        rgb_img = rgb_img.repeat(3, 1, 1)
    elif len(rgb_img.shape) == 3:
        rgb_img = rgb_img.repeat(3, 1, 1)
    elif len(rgb_img.shape) == 4:
        rgb_img = rgb_img[0].repeat(3, 1, 1)

    return rgb_img


if __name__ == "__main__":
    main()
