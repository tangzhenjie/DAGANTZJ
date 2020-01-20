from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import random
from PIL import Image as m
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

image_sizes = {'cityscapes': (1024, 512), 'gta5': (1280, 720), 'synthia': (1280, 760)}

class SrdaDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets

    It requires two directories to host training images from domain A '/path/A/train/images
    and from domain B '/path/B/train/images
    You can train the model with flag '--dataroot /path/'
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        :param parser: -- original option parser
        :param is_train: -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        :return: the modified parser.
        """
        parser.add_argument('--A_crop_size', type=int, default=560, help='A crop to this size')
        parser.add_argument('--B_crop_size', type=int, default=560, help='B crop to this size')
        parser.add_argument('--no_label', type=bool, default=False, help='A is not with label')
        parser.add_argument('--A_dataset', type=str, default='synthia', help='what is the A: synthia')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A_images = opt.dataroot + "/" + opt.A_dataset + '/RGB/'
        self.dir_A_labels = opt.dataroot + "/" + opt.A_dataset + '/labels/'
        self.A_images_paths_txt = opt.dataroot + "/" + opt.A_dataset + "_list/train.txt"

        self.dir_B_images = opt.dataroot + "/" + 'cityscapes/train_images/'
        self.B_images_paths_txt = opt.dataroot + "/" + 'cityscapes_list/train.txt'


        self.A_images_paths = [i_id.strip()[4:] for i_id in open(self.A_images_paths_txt)]
        self.B_images_paths = [B_path.strip() for B_path in open(self.B_images_paths_txt)]
        self.A_size = len(self.A_images_paths)  # get the size of dataset A
        self.B_size = len(self.B_images_paths)  # get the size of dataset B

        self.id_to_trainid = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                              15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                              8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}


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
        # 提前声明
        A_label_copy = []
        A_label_path = ''

        # get the image datas
        A_image_path = self.dir_A_images + self.A_images_paths[index % self.A_size]
        if not self.opt.no_label:
            A_label_path = self.dir_A_labels + self.A_images_paths[index % self.A_size]

        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_image_path = self.dir_B_images + self.B_images_paths[index_B]

        # preprocess
        A_image = m.open(A_image_path).convert('RGB')
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            A_image, output_size=((self.opt.A_crop_size, self.opt.A_crop_size))) # (480, 480)
        A_image = TF.crop(A_image, i, j, h, w)
        # resize
        A_image = A_image.resize((int(self.opt.A_crop_size * 0.5), int(self.opt.A_crop_size * 0.5)), resample=m.BICUBIC) # (240, 240)

        if not self.opt.no_label:
            A_label = m.open(A_label_path).convert('L')
            # crop
            A_label = TF.crop(A_label, i, j, h, w)
            # resize
            A_label = A_label.resize((int(self.opt.A_crop_size * 0.5), int(self.opt.A_crop_size * 0.5)), m.NEAREST) # (240, 240)
            A_label = np.asarray(A_label, np.long)
            # re-assign labels to match the format of Cityscapes
            A_label_copy = 255 * np.ones(A_label.shape, dtype=np.long)
            for k, v in self.id_to_trainid.items():
                A_label_copy[A_label == k] = v

        B_image = m.open(B_image_path).convert('RGB')
        # Random crop
        i_1, j_1, h_1, w_1 = transforms.RandomCrop.get_params(
            B_image, output_size=((self.opt.B_crop_size, self.opt.B_crop_size)))  # (480, 480)
        B_image = TF.crop(B_image, i_1, j_1, h_1, w_1)

        # resize
        B_image = B_image.resize((int(self.opt.A_crop_size * 0.5), int(self.opt.A_crop_size * 0.5)),
                                 resample=m.BICUBIC)  # (240, 240)

        # change to the tensor
        nomal_fun_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        A_image = TF.to_tensor(A_image)
        A_image = nomal_fun_image(A_image)

        if not self.opt.no_label:
            A_label_copy = TF.to_tensor(A_label_copy)

        B_image = TF.to_tensor(B_image)
        B_image = nomal_fun_image(B_image)

        if not self.opt.no_label:
            return {"A_image": A_image, "A_label": A_label_copy, "B_image": B_image}
        else:
            return {"A_image": A_image, "B_image": B_image}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)