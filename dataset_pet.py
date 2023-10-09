# lade Bilder und wandel sie in NP Arrays um

# https://academictorrents.com/details/b18bbd9ba03d50b0f7f479acc9f4228a408cecc1


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from os import listdir
from os.path import join
from random import random
import torch
from einops import rearrange


class PetOnlySegmentationDataSet(Dataset):
    def __init__(self, ImageFileList, SegmentationFileList, transform=None):
        self.imagefilelist = ImageFileList
        self.segmentationfilelist = SegmentationFileList
        self.transform = transform  

    def __getitem__(self, index):
        image_file = self.imagefilelist[index]
        seg_file = self.segmentationfilelist[index]

        image = np.array(Image.open(image_file).convert('RGB'))
        segmentation = np.array(Image.open(seg_file))

        # siehe Pet Readme, beides Hintergrund
        segmentation[segmentation == 2] = 0
        segmentation[segmentation == 3] = 0

        if self.transform is not None:
            pass

        image = torch.tensor(image)
        image = rearrange(image, "H W C -> C H W")
        image = image.unsqueeze(dim=0).float()

        segmentation =  torch.tensor(segmentation).float()

        return image, segmentation
    
    def __len__(self):
        return len(self.imagefilelist)

def get_train_val_file_list(image_path, seg_path, split=0.7):

    image_filelist = sorted([filename
        for filename in listdir(image_path)
        if not filename.endswith('.mat')
    ])
    seg_file_list = [filename
        for filename in listdir(seg_path)
        if not filename.startswith('._')
    ]
    seg_file_list = sorted(seg_file_list)

    train_image_list = list()
    train_seg_list = list()

    val_image_list = list()
    val_seg_list = list()


    for image_file, seg_file in zip(image_filelist, seg_file_list):
        long_image_file = join(image_path, image_file)
        long_seg_file = join(seg_path, seg_file)

        if random() <= split:
            train_image_list.append(long_image_file)
            train_seg_list.append(long_seg_file)
        else:
            val_image_list.append(long_image_file)
            val_seg_list.append(long_seg_file)

    return (train_image_list, train_seg_list), (val_image_list, val_seg_list)
    

if __name__ == "__main__":
    (train_image_list, train_seg_list), (val_image_list, val_seg_list) = get_train_val_file_list()
    breakpoint()

    train_dataset = PetOnlySegmentationDataSet(train_image_list, train_seg_list)
    train_dl = DataLoader(train_dataset)


    val_dataset = PetOnlySegmentationDataSet(val_image_list, val_seg_list)
    val_dl = DataLoader(val_dataset)

    print("trainign..")
    #for x, y in train_dl:
        #print(x.shape, y.shape)
        #pass

    print('validation...')
    for x, y in val_dl:
        print(x.shape, y.shape)
