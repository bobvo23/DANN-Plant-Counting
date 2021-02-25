"""Dataset setting and data loader for MNIST-M.
Modified from
https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
CREDIT: https://github.com/eriklindernoren

"""

import errno
import os
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import numpy as np


def data_from_pkl(pkl_path):
    with open(pkl_path, 'rb') as fp:
        d = pickle.load(fp)

    return d


class CVPPP(data.Dataset):
    """CVPPP Leaf counting Dataset."""
    CVPPP_pickle_file = "./data/CVPPP/CVPPP_img_lbl_dict.pkl"

    def __init__(self, root, train=True, transform=None):
        """Init CVPPP dataset."""
        # toclarify: do we need this parent initialization? tut7 doesn't have it? Verify later.
        super(CVPPP, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set

        if not self._check_exists():
            print(f"Dataset CVPPP not found at {CVPPP_pickle_file}")
            raise RuntimeError("Dataset not found." +
                               " You can use download CVPPP to data folder")

        # TODO: split data to train and test based on the train param input
        # Now all data are used for training/valid

        #----------Loading Data from Pickle--------------#
        self.train_data_dict = data_from_pkl(self.CVPPP_pickle_file)
        self.train_data_keys = list(self.train_data_dict.keys())

        self.train_data = []
        self.train_labels = []

        for key in self.train_data_keys:
            self.train_data.append(self.train_data_dict[key][0])
            self.train_labels.append(self.train_data_dict[key][1])

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.train_data[index], self.train_labels[index]

        if self.transform != None:
            img = self.transform(Image.fromarray((img*255).astype(np.uint8)))

        return img, torch.Tensor(target)

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        #path = os.path.join(self.root, self.CVPPP_pickle_file)
        path = self.CVPPP_pickle_file
        #print(f"dataset file path {path}")
        return os.path.exists(path)


class KOMATSUNA(data.Dataset):
    """Komatsuna Leaf counting Dataset."""
    KOMATSUNA_images_file = "./data/KOMATSUNA/komatsuna_ds.pkl"
    KOMATSUNA_label_file = "./data/KOMATSUNA/komatsuna_lbl.pkl"

    def __init__(self, root, train=True, transform=None):
        """Init KOMATSUNA dataset."""
        # toclarify: do we need this parent initialization? Verify later.
        super(KOMATSUNA, self).__init__()
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download CVPPP to data folder")

        # TODO: split data to train and test based on the train param input
        # Now all data are used for training/valid

        #----------Loading Data from Pickle--------------#
        # This dataset is slightly different from CVPPP (without density labels)
        self.images_dict = data_from_pkl(self.KOMATSUNA_images_file)
        self.images_keys = list(self.images_dict.keys())
        self.labels_dict = data_from_pkl(self.KOMATSUNA_label_file)

        self.train_data = []
        self.train_labels = []

        for key in self.images_keys:
            self.train_data.append(self.images_dict[key])
            # key in images dict: 'rgb_04_009_00'
            # key in label dict: 'label_04_009_00'
            key = "label" + key[3:]
            self.train_labels.append(self.labels_dict[key])

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.train_data[index], self.train_labels[index]

        if self.transform != None:
            img = self.transform(Image.fromarray((img*255).astype(np.uint8)))

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        path = self.KOMATSUNA_images_file
        print(f"dataset file path {path}")
        return os.path.exists(path)


class MNISTM(data.Dataset):
    """`MNIST-M Dataset."""

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    def __init__(self, root, mnist_root="data", train=True, transform=None, target_transform=None, download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder,
                             self.training_file)
            )
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file)
            )

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_file)
        )

    def download(self):
        """Download the MNIST data."""
        # import essential packages
        from six.moves import urllib
        import gzip
        import pickle
        from torchvision import datasets

        # check if dataset already exists
        if self._check_exists():
            return

        # make data dirs
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        print("Downloading " + self.url)
        filename = self.url.rpartition("/")[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path.replace(".gz", "")):
            data = urllib.request.urlopen(self.url)
            with open(file_path, "wb") as f:
                f.write(data.read())
            with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print("Processing...")

        # load MNIST-M images from pkl file
        with open(file_path.replace(".gz", ""), "rb") as f:
            mnist_m_data = pickle.load(f, encoding="bytes")
        mnist_m_train_data = torch.ByteTensor(mnist_m_data[b"train"])
        mnist_m_test_data = torch.ByteTensor(mnist_m_data[b"test"])

        # get MNIST labels
        mnist_train_labels = datasets.MNIST(
            root=self.mnist_root, train=True, download=True).train_labels
        mnist_test_labels = datasets.MNIST(
            root=self.mnist_root, train=False, download=True).test_labels

        # save MNIST-M dataset
        training_set = (mnist_m_train_data, mnist_train_labels)
        test_set = (mnist_m_test_data, mnist_test_labels)
        with open(os.path.join(self.root, self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        print("Done!")
