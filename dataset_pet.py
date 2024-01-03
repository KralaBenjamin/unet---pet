# lade Bilder und wandel sie in NP Arrays um

# https://academictorrents.com/details/b18bbd9ba03d50b0f7f479acc9f4228a408cecc1


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from os import listdir, PathLike
from os.path import join
from random import random
import torch
from einops import rearrange
from pathlib import Path
from typing import List
from loguru import logger
from typing import Literal
from copy import copy
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

class PetOnlySegmentationDataSet(Dataset):
    def __init__(self, ImageFileList, SegmentationFileList, transform=None):
        self.imagefilelist = ImageFileList
        self.segmentationfilelist = SegmentationFileList
        self.transform = transform

    def __getitem__(self, index):
        image_file = self.imagefilelist[index]
        seg_file = self.segmentationfilelist[index]

        image = np.array(Image.open(image_file).convert("RGB"))
        segmentation = np.array(Image.open(seg_file))

        # siehe Pet Readme, beides Hintergrund
        segmentation[segmentation == 2] = 0
        segmentation[segmentation == 3] = 0

        image = torch.tensor(image)
        image = rearrange(image, "H W C -> C H W")
        image = image.float()

        segmentation = torch.tensor(segmentation).float()

        if self.transform:
            image, segmentation = self.transform(image), self.transform(segmentation)

        return image, segmentation

    def __len__(self):
        return len(self.imagefilelist)


def get_train_val_file_list(image_path: Path, seg_path: Path, split=0.7):
    image_filelist = sorted(
        [
            Path(filename)
            for filename in listdir(image_path)
            if not filename.endswith(".mat")
        ]
    )
    seg_file_list = [
        Path(filename)
        for filename in listdir(seg_path)
        if not filename.startswith("._")
    ]
    seg_file_list = sorted(seg_file_list)

    train_image_list = list()
    train_seg_list = list()

    val_image_list = list()
    val_seg_list = list()

    for image_file, seg_file in zip(image_filelist, seg_file_list):
        long_image_file = image_path / image_file
        long_seg_file = seg_path / seg_file

        if random() <= split:
            train_image_list.append(long_image_file)
            train_seg_list.append(long_seg_file)
        else:
            val_image_list.append(long_image_file)
            val_seg_list.append(long_seg_file)

    return (train_image_list, train_seg_list), (val_image_list, val_seg_list)



class PetMulticlassSegmentationSet(Dataset):
    def __init__(
        self,
        ClassDescriptionFile: Path,
        ImageFileList: List[Path],
        SegmentationFileList: List[Path],
        target_class: Literal['animal', 'race'] ="animal",
        ClassInformation=None, 
        transform=None,
        debug=False
    ):
        self.class_description_file = ClassDescriptionFile
        self.imagefilelist = ImageFileList
        self.segmentationfilelist = SegmentationFileList
        self.transform = transform
        self.target_class = target_class

        self.list_data = list()
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



        if ClassInformation is None:
            list_classes = list({animal[target_class] for animal in image_file2category.values()})
            class_to_i = {
                class_element: i + 1 for i, class_element in enumerate(list_classes)
            }
            class_to_i |= {'no_content': 0}
            self.label_binariser = LabelBinarizer()
            self.label_binariser.fit(list(class_to_i.values()))
            self.class_to_i = class_to_i
        else:
            label_binariser, class_to_i = ClassInformation
            self.label_binariser = label_binariser
            self.class_to_i = class_to_i

        for image_file_path, segmentation_file_path in zip(
            ImageFileList, SegmentationFileList
        ):
            image_file = image_file_path.name
            image_name = image_file.split(".")[:-1][0]
            if not image_name in image_file2category:
                logger.debug(f"not found {image_name} in image_desc_file")
                continue
            self.list_data.append(
                {
                    "image_path": image_file_path,
                    "seg_path": segmentation_file_path,
                }
                | image_file2category[image_name]
            )
        self.debug = debug

    def __getitem__(self, index):
        if self.debug:
            breakpoint()
        item = self.list_data[index]
        image_file = item["image_path"]
        seg_file = item["seg_path"]

        current_class_i = self.class_to_i[item[self.target_class]]
        he_target = self.label_binariser.transform([current_class_i])[0]
        zero_target = self.label_binariser.transform([0])[0]

        image = np.array(Image.open(image_file).convert("RGB"))
        segmentation = np.array(Image.open(seg_file))
        
        segmentation_shape = (
            segmentation.shape[0],
            segmentation.shape[1],
            he_target.shape[0])
        
        he_segmentation = np.zeros(segmentation_shape)
        he_segmentation[segmentation == 1] = he_target
        he_segmentation[segmentation == 2] = zero_target
        he_segmentation[segmentation == 3] = zero_target

        image = torch.tensor(image)
        image = rearrange(image, "H W C -> C H W")
        image = image.float()

        he_segmentation = torch.tensor(he_segmentation)
        he_segmentation = rearrange(he_segmentation, "H W C -> C H W")
        he_segmentation = he_segmentation.float()

        if self.transform:
            image, he_segmentation = self.transform(image), self.transform(he_segmentation)

    
        return image, he_segmentation

    def __len__(self):
        return len(self.list_data)
    
    def get_n_classes(self):
        _, y_element = self.__getitem__(0)
        return y_element.shape[0]
    
    def get_class_information(self):
        return self.label_binariser, self.class_to_i
    
    def get_json_list(self, category="race"):
        json_list = [
            (
                str(element['image_path']),
                str(element['seg_path']),
                element[category],
            )
            for element in self.list_data
        ]
        return json_list


if __name__ == "__main__":
    image_list = "data/pets/images"
    seg_list = "data/pets/annotations/trimaps"
    (train_image_list, train_seg_list), (
        val_image_list,
        val_seg_list,
    ) = get_train_val_file_list(image_list, seg_list)

    image_desc_file = "data/pets/annotations/list.txt"
    from torchvision.transforms import CenterCrop

    transform = CenterCrop((256, 256))
    target_class = 'animal'

    train_dataset = PetMulticlassSegmentationSet(
        image_desc_file, 
        train_image_list, 
        train_seg_list, 
        transform=transform,
        target_class=target_class
    )
    train_dl = DataLoader(train_dataset, batch_size=16)
    print("n_classes train", train_dataset.get_n_classes())

    val_dataset = PetMulticlassSegmentationSet(
        image_desc_file, 
        val_image_list, 
        val_seg_list,
        transform=transform,
        target_class=target_class
    )
    val_dl = DataLoader(val_dataset, batch_size=16)
    print("n_classes val", val_dataset.get_n_classes())


    print("trainign..")
    for x, y in train_dl:
        print(x.shape, y.shape)
    # pass

    print("validation...")
    # for x, y in val_dl:
    #    print(x.shape, y.shape)
