# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


from typing import Callable
import torch
from torch.utils.data import Dataset
from .parse_data import load_nerf_standard_data

from torchvision.transforms import Resize
import numpy as np
import cv2


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # ? why need transpose here
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose  # ! return cam2world matrix here


class MultiviewDataset(Dataset):
    """This is a static multiview image dataset class.

    This class should be used for training tasks where the task is to fit a static 3D volume from
    multiview images.
    """

    def __init__(self, 
        dataset_path             : str,
        mip                      : int      = None,
        bg_color                 : str      = None,
        sample_rays              : bool     = False,
        n_rays                   : int      = 1024,
        split                    : str      = 'train',
        **kwargs
    ):
        """Initializes the dataset class.

        Note that the `init` function to actually load images is separate right now, because we don't want 
        to load the images unless we have to. This might change later.

        Args: 
            dataset_path (str): Path to the dataset.
            multiview_dataset_format (str): The dataset format. Currently supports standard (the same format
                used for instant-ngp) and the RTMV dataset.
            mip (int): The factor at which the images will be downsampled by to save memory and such.
                       Will downscale by 2**mip.
            bg_color (str): The background color to use for images with 0 alpha.
            dataset_num_workers (int): The number of workers to use if the dataset format uses multiprocessing.
        """
        self.root = dataset_path
        self.mip = mip
        self.bg_color = bg_color
        self.sample_rays = sample_rays
        self.n_rays = n_rays
        self.split = split
        self.init()

    def init(self):
        """Initializes the dataset.
        """

        # Get image tensors 
        
        self.coords = None

        self.resizer = Resize(256)
        self.down_factor = 4

        self.data = self.get_images(self.split, self.mip)

        self.img_shape = self.data["imgs"].shape[1:3]
        self.num_imgs = self.data["imgs"].shape[0]

        self.data["imgs"] = self.data["imgs"].reshape(self.num_imgs, -1, 3)
        self.data["rays"] = self.data["rays"].reshape(self.num_imgs, -1, 6)

        if "masks" in self.data:
            self.data["masks"] = self.data["masks"].reshape(self.num_imgs, -1, 1)

        self.downsample_imgs()
        self.compute_camera_params()

    def get_images(self, split='train', mip=None):
        """Will return the dictionary of image tensors.

        Args:
            split (str): The split to use from train, val, test
            mip (int): If specified, will rescale the image by 2**mip.

        Returns:
            (dict of torch.FloatTensor): Dictionary of tensors that come with the dataset.
        """
        
        data = load_nerf_standard_data(self.root, split,
                        bg_color=self.bg_color, num_workers=-1, mip=self.mip)

        return data
    
    def downsample_imgs(self):
        down_imgs = self.resizer(
            torch.permute(
                self.data["imgs"].reshape(self.num_imgs, *self.img_shape, 3).float(), (0, 3, 1, 2)
            )
        )
        self.data['down_imgs'] = down_imgs
    
    def compute_camera_params(self):
        self.data['ref_poses'] = dict()
        all_extr = torch.zeros((self.data['down_imgs'].shape[0], 3, 4))
        all_intr = torch.zeros((self.data['down_imgs'].shape[0], 3, 3))
        all_near_far = torch.zeros((self.data['down_imgs'].shape[0], 2))
        w2cs = torch.zeros((self.data['down_imgs'].shape[0], 4, 4))
        c2ws = torch.zeros((self.data['down_imgs'].shape[0], 4, 4))
        affine_mats = torch.zeros((self.data['down_imgs'].shape[0], 4, 4))
        for i, (k, v) in enumerate(self.data['cameras'].items()):
            print(v)
            extr, intr = v.parameters()

            print(extr)
            print(intr)

            if i == 0:
                w2c_ref = extr.reshape(4, 4)
                w2c_ref_inv = np.linalg.inv(w2c_ref)

            extr = extr.reshape(4, 4)[:3, :]
            intr = torch.tensor([[intr[0][2], 0, intr[0][0]],
                                 [0, intr[0][3], intr[0][1]],
                                 [0, 0, 1]])
            intr[:2] *= self.down_factor
            all_extr[i] = extr
            all_intr[i] = intr
            # Default values from load_nerf_standard_data()
            all_near_far[i] = torch.tensor([0., 6.])


            ###########################################################
            # PART RELEVANT FOR SPARSENEUS
            # w2c = extr @ w2c_ref_inv

            # # NOTE: Removed calculate scale mat!
            # P = intr.numpy() @ w2c.numpy()
            # P = P[:3, :4]

            # c2w = load_K_Rt_from_P(None, P)[1]
            # w2c = np.linalg.inv(c2w)

            # w2cs[i] = torch.tensor(w2c)
            # c2ws[i] = torch.tensor(c2w)
            w2c = extr
            affine_mat = np.eye(4)
            affine_mat[:3, :4] = intr[:3, :3] @ w2c[:3, :4]
            affine_mats[i] = v.parameters()[0].reshape(4, 4)
            ###########################################################
            
        self.data['ref_poses']['extrinsics'] = all_extr
        self.data['ref_poses']['intrinsics'] = all_intr
        self.data['ref_poses']['near_fars'] = all_near_far

        self.data['ref_poses']['w2cs'] = w2cs
        self.data['ref_poses']['c2ws'] = c2ws
        self.data['ref_poses']['proj_mats'] = affine_mats

        # NOTE: Not sure if this is the right value! Based on SparseNeuS
        self.data['ref_poses']['partial_vol_origin'] = torch.Tensor([0., 0., 0.])


    def sample(self, inputs, num_samples):
        """ Samples a subset of rays from a single image.
            50% of the rays are sampled randomly from the image.
            50% of the rays are sampled randomly within the valid mask.
        """
        valid_idx = torch.nonzero(inputs['masks'].squeeze()).squeeze()

    
        ray_idx = torch.randperm(
            inputs['imgs'].shape[0],
            device=inputs['imgs'].device)[:num_samples]
        
        select_idx = torch.randperm( valid_idx.shape[0], device=inputs['imgs'].device) [:num_samples // 2] 

        ray_idx [:num_samples // 2] = valid_idx [select_idx]    
        out = {}
        out['rays'] = inputs['rays'][ray_idx].contiguous()
        out['imgs'] = inputs['imgs'][ray_idx].contiguous()
        out['masks'] = inputs['masks'][ray_idx].contiguous()
        return out


    def __len__(self):
        """Length of the dataset in number of rays.
        """
        return self.data["imgs"].shape[0]

    def __getitem__(self, idx : int):
        """Returns rays, gt ray colors, and binary masks. 
        """
        out = {}
        out['rays'] = self.data["rays"][idx].float()
        out['imgs'] = self.data["imgs"][idx].float()
        out['masks'] = self.data["masks"][idx].bool()

        if self.sample_rays and self.split == 'train':
            out = self.sample(out, self.n_rays)

        out['full_imgs'] = self.data['down_imgs'].float()

        out['ref_poses'] = self.data['ref_poses']

        return out
    
    def get_img_samples(self, idx, batch_size):
        """Returns a batch of samples from an image, indexed by idx.
        """

        ray_idx = torch.randperm(self.data["imgs"].shape[1])[:batch_size]

        out = {}
        out['rays'] = self.data["rays"][idx, ray_idx]
        out['imgs'] = self.data["imgs"][idx, ray_idx]
        
        return out
