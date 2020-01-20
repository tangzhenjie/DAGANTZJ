from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import torchvision.transforms as transforms
from PIL import Image as m
import torch
import numpy as np
import os.path as osp
import json
import torchvision.transforms.functional as TF
#(1024, 512)
image_sizes = {'cityscapes': (1024, 512), 'gta5': (1280, 720), 'synthia': (1280, 760)}

class SingleDataset(BaseDataset):
    """load train and val for segmentation network
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        :param parser: -- original option parser
        :param is_train: -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        :return: the modified parser.
        """
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        return parser
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_B_images = opt.dataroot + "/" + 'cityscapes/val_images/'
        self.dir_B_labels = opt.dataroot + "/" + 'cityscapes/val_labels/'

        self.B_images_paths_txt = opt.dataroot + "/" + 'cityscapes_list/val.txt'
        self.B_labels_paths_txt = opt.dataroot + "/" + 'cityscapes_list/label.txt'

        self.B_images_paths = sorted([image_path.strip() for image_path in open(self.B_images_paths_txt)])
        self.B_labels_paths = sorted([label_path.strip() for label_path in open(self.B_labels_paths_txt)])
        self.B_size = len(self.B_images_paths)
        self.infor_json = opt.dataroot + "/" + 'cityscapes_list/info.json'
        with open(self.infor_json, 'r') as fp:
            self.info = json.load(fp)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # read the image and corresponding to the label
        name = self.B_labels_paths[index % self.B_size]
        B_image_path = self.dir_B_images + self.B_images_paths[index % self.B_size]
        B_label_path = self.dir_B_labels + self.B_labels_paths[index % self.B_size]

        B_image = m.open(B_image_path).convert('RGB')
        B_image = B_image.resize(image_sizes['cityscapes'], m.BICUBIC)
        #B_image = np.asarray(B_image, np.float32)

        B_label = m.open(B_label_path).convert('L')
        #B_label = B_label.resize(image_sizes['cityscapes'], m.NEAREST) # 因为我们最后结果会上采样到label一样大小
        #B_label = np.asarray(B_label, np.long)
        B_label = np.array(B_label).astype(np.long)

        # re-assign labels to match the format of trainid
        mapping = np.array(self.info['label2train'], dtype=np.int)
        B_label = self.label_mapping(B_label, mapping)

        # to tensor
        nomal_fun_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        B_image = TF.to_tensor(B_image)
        B_image = nomal_fun_image(B_image)

        # label is np
        return {'B_image': B_image, 'B_label': B_label, "name": name}

    def label_mapping(self, input, mapping):
        output = np.copy(input)
        for ind in range(len(mapping)):
            output[input == mapping[ind][0]] = mapping[ind][1]
        return np.array(output, dtype=np.int64)


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.B_size