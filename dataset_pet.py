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


def extract_hot_encodings(all_classes):
    print(all_classes)
    class2id = dict(zip(list(all_classes), range(len(all_classes))))

    # creates hot encoding
    eye_matrice = torch.eye(len(all_classes))

    hot_encode = lambda x: eye_matrice[x]
    class2hot_code = {classe: hot_encode(ide) for classe, ide in class2id.items()}

    return class2hot_code


def create_hot_encoders(ClassDescriptionFile: Path):
    list_data = list()
    with open(ClassDescriptionFile, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            image, class_image, species_image, breed_id = line.split(" ")
            list_data.append((int(species_image), int(breed_id)))

    breedclass2he = extract_hot_encodings(list_data)
    animal2he = extract_hot_encodings([1, 2])

    return breedclass2he, animal2he


class PetMulticlassSegmentationSet(Dataset):
    def __init__(
        self,
        ClassDescriptionFile: Path,
        ImageFileList: List[Path],
        SegmentationFileList: List[Path],
        transform=None,
    ):
        self.class_description_file = ClassDescriptionFile
        self.imagefilelist = ImageFileList
        self.segmentationfilelist = SegmentationFileList
        self.transform = transform

        self.list_data = list()
        image_file2category = dict()
        breedclass2he, animal2he = create_hot_encoders(ClassDescriptionFile)
        with open(ClassDescriptionFile, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                image, class_image, species_image, breed_id = line.split(" ")
                image_file2category[image] = {
                    "race": breedclass2he[(int(species_image), int(breed_id))],
                    "animal": animal2he[int(species_image)],
                }

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

    def __getitem__(self, index):
        image_file = self.list_data[index]["image_path"]
        seg_file = self.list_data[index]["seg_path"]

        image = np.array(Image.open(image_file).convert("RGB"))
        segmentation = np.array(Image.open(seg_file))

        # siehe Pet Readme, beides Hintergrund
        # todo: ändern
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
        return len(self.list_data)


if __name__ == "__main__":
    image_list = "data/pets/images"
    seg_list = "data/pets/annotations/trimaps"
    (train_image_list, train_seg_list), (
        val_image_list,
        val_seg_list,
    ) = get_train_val_file_list(image_list, seg_list)

    image_desc_file = "data/pets/annotations/list.txt"
    from torchvision.transforms import Resize, CenterCrop

    transform = CenterCrop((256, 256))

    train_dataset = PetMulticlassSegmentationSet(
        image_desc_file, train_image_list, train_seg_list, transform=transform
    )
    train_dl = DataLoader(train_dataset, batch_size=16)

    val_dataset = PetMulticlassSegmentationSet(
        image_desc_file, val_image_list, val_seg_list
    )
    val_dl = DataLoader(val_dataset, batch_size=16)

    print("trainign..")
    for x, y in train_dl:
        print(x.shape, y.shape)
    # pass

    print("validation...")
    # for x, y in val_dl:
    #    print(x.shape, y.shape)
