import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import glob
from scipy.io import loadmat


class OrganSegmentationDataset(Dataset):
    def __init__(
            self,
            images_dir,
            subset="train",
    ):
        self.subset = subset
        self.data_paths = []
        self.patient_ids = []
        self.required_test = False

        if subset == 'train':
            self.images_dir = os.path.join(images_dir, "demo_train")
        else:
            self.images_dir = os.path.join(images_dir, "test_selected")


        print("reading {} images...".format(subset))
        self.data_paths = sorted(glob.glob(self.images_dir + os.sep + "*.mat"), key=lambda x: x.split(os.sep)[-1])
        print("Count ", len(self.data_paths))


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, id):
        file_path = self.data_paths[id]
        file_name = file_path.split(os.sep)[-1]
        patient_id = file_name.split("_")[-3]
        slice_id = file_name.split("_")[-1].split(".")[0]
        mat = loadmat(file_path)
        affine = 0
        mask = mat['mask']
        image = mat['img']
        
        # expand dimention
        image = image[None, :, :]

        if int(slice_id) == 1:
            affine = mat['affine']
        
        # normalization of image
        min_value = np.amin(image)
        if min_value < 0:
            image = image - min_value
        max_value = np.amax(image)
        image = image / max_value

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(int))
        mask_tensor = mask_tensor.long()

        return image_tensor, mask_tensor, affine, patient_id, slice_id


def data_loaders(data_dir):
    dataset_train, dataset_valid = datasets(data_dir)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(data_dir):
    train = OrganSegmentationDataset(images_dir=data_dir,
                                     subset="train"
                                     )
    valid = OrganSegmentationDataset(images_dir=data_dir,
                                     subset="test"
                                     )
    return train, valid
