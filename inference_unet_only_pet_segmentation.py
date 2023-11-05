import torch
import click
from pathlib import Path
from unet import UNET
from PIL import Image
import numpy as np
from einops import rearrange
from datetime import datetime
from json import load as jsonload


@click.command()
@click.argument('experiment_name')
@click.option('--path_input_image', default=None, help='Pfad zum Input Bild.')
@click.option('--path_seg_image', default=None, help='Pfad zum Segmentation Bild.')
@click.option('--path_output_image', default=None, help='Pfad zum Output Bild.')
def main(experiment_name, path_input_image, path_seg_image, path_output_image):
    model_dir = Path(f"./models/{experiment_name}")

    model = UNET(3, 1)
    model.load_state_dict(torch.load(model_dir / 'best_model.pth'))    

    #lade bild

    if path_input_image is None:
        list_path_val = jsonload(open(
            model_dir / 'val_data_files.json', 'r'))
        path_input_image = list_path_val[0][0]
        path_seg_image = list_path_val[0][1]

    if path_output_image is None:
        dt = datetime.now()
        date_time_str = dt.strftime("%Y-%m-%d-%H-%M-%S")
        path_output_image = f"results/{date_time_str}.png"

    print(f"{experiment_name=} \t {path_input_image=} \t {path_seg_image=} \t {path_output_image=}")

    image = np.array(Image.open(path_input_image).convert('RGB'))
    segmentation = np.array(Image.open(path_seg_image))

    # siehe Pet Readme, beides Hintergrund
    segmentation[segmentation == 2] = 0
    segmentation[segmentation == 3] = 0

    model.eval()

    image = torch.tensor(image)
    image = rearrange(image, "H W C -> C H W")
    image = image.float().unsqueeze(dim=0)

    with torch.no_grad():
        pred = model(image)
        pred_log = (pred > 0.5).float()

    list_output = [
        image[0], 
        bool_tensor_into_rgb_tensor(pred)
        ]
    if not path_seg_image is None:
        segmentation =  torch.tensor(segmentation).float()
        comparison = pred == segmentation

        list_output.append(
            bool_tensor_into_rgb_tensor(comparison)
        )
    combined_tensor = torch.cat(list_output, dim=2)
    combined_tensor = rearrange(combined_tensor, "C H W -> H W C")
    np_image = np.uint8(combined_tensor.numpy() * 255)
    combined_image = Image.fromarray(np_image, 'RGB')
    combined_image.save(path_output_image)

def bool_tensor_into_rgb_tensor(bool_img):
    rgb_img = bool_img.float() * 255
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img.repeat(3, 1, 1)
    elif len(rgb_img.shape) == 4:
        rgb_img = rgb_img[0].repeat(3, 1, 1)

    return rgb_img

if __name__ == '__main__':
    main()



