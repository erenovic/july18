import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging as log
from tqdm import tqdm
from PIL import Image
import trimesh
import mcubes
import wandb

from .ray import exponential_integration
from ..utils.metrics import psnr

# Warning: you MUST NOT change the resolution of marching cube
RES = 256

class Trainer(nn.Module):

    def __init__(self, config, renderer, log_dir, device):

        super().__init__()

        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_dict = {}
        self.log_dir = log_dir
        self.valid_mesh_dir = os.path.join(self.log_dir, "mesh")

        self.renderer = renderer.to(self.device)

        self.n_samples = config.renderer["n_samples"]
        self.perturb = config.renderer["perturb"]
        self.mask_weight = config.mask_weight
        self.igr_weight = config.igr_weight
        
        self.init_optimizer()
        self.init_log_dict()

    def init_optimizer(self):
        
        trainable_parameters = list(self.renderer.parameters())
        self.optimizer = torch.optim.Adam(trainable_parameters, lr=self.cfg.lr, 
                                    betas=(self.cfg.beta1, self.cfg.beta2),
                                    weight_decay=self.cfg.weight_decay)

    def init_log_dict(self):
        """Custom log dict.
        """
        self.log_dict['total_loss'] = 0.0
        self.log_dict['rgb_loss'] = 0.0
        self.log_dict['total_iter_count'] = 0
        self.log_dict['image_count'] = 0


    def sample_points(self, ray_orig, ray_dir, near=1.0, far=3.0, perturb_overwrite=-1):
        """Sample points along rays. Retruns 3D coordinates of the points.
        TODO: One and extend this function to the hirachical sampling technique 
             used in NeRF or design a more efficient sampling technique for 
             better surface reconstruction.

        Args:
            ray_orig (torch.FloatTensor): Origin of the rays of shape [B, Nr, 3].
            ray_dir (torch.FloatTensor): Direction of the rays of shape [B, Nr, 3].
            near (float): Near plane of the camera.
            far (float): Far plane of the camera.
            num_points (int): Number of points (Np) to sample along the rays.

         Returns:
            points (torch.FloatTensor): 3D coordinates of the points of shape [B, Nr, Np, 3].
            z_vals (torch.FloatTensor): Depth values of the points of shape [B, Nr, Np, 1].
            deltas (torch.FloatTensor): Distance between the points of shape [B, Nr, Np, 1].

        """
        B = len(ray_orig)

        # Distances to sample at for each ray. First uniform sampling, then conducts importance sampling later.
        sample_dist = 2.0 / self.n_samples                      # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(self.device)
        z_vals = near + (far - near) * z_vals[None, :]
        
        # Add noise to sampling distances if perturb > 0. If overwritten, use the overwriting value
        perturb = self.perturb
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([B, 1]) - 0.5).to(self.device)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        bg_alpha, bg_sampled_color= None, None

        points = ray_orig[:, None, :] + ray_dir[:, None, :] * z_vals[..., :, None]

        # print("Points shape:", points.shape)  torch.Size([512, 64, 3])
        # print("Z vals shape:", z_vals.shape)  torch.Size([512, 64])

        return points, z_vals, sample_dist

    def forward(self, ray_orig, ray_dir, imgs, bg_rgb=None, ref_poses=None):
        """Forward pass of the network. 
        TODO: Adjust the neural rendering pipeline according to your needs.

        Returns:
            rgb (torch.FloatTensor): Ray codors of shape [B, Nr, 3].

        """
        ray_orig = ray_orig.squeeze(0)
        ray_dir = ray_dir.squeeze(0)

        # print(ray_orig.shape)     torch.Size([512, 3])
        # print(ray_dir.shape)      torch.Size([512, 3])
        # print(img_gts.shape)      torch.Size([512, 3])

        # Step 1 : Sample points along the rays
        points, z_vals, sample_dist = self.sample_points(
            ray_orig, ray_dir, near=self.cfg.near, far=self.cfg.far
        )

        # Step 2 : Compute conditional volume and mask
        feature_maps = self.renderer.compute_features(imgs)
        conditional_volume, conditional_volume_mask = self.renderer.get_conditional_volume(imgs, ref_poses, feature_maps)

        # Step 3 : Predict radiance and volume density at the sampled points
        img_size = imgs.shape[-2:]
        sizeH, sizeW = img_size[0], img_size[1]

        breakpoint()
        output = self.renderer(
            points, ray_orig, ray_dir, z_vals, sample_dist, 
            bg_rgb, conditional_volume, conditional_volume_mask, ref_poses=ref_poses, img_wh=[sizeW, sizeH],
            feature_maps=feature_maps, color_maps=imgs[0]
        )

        return output

    def backward(self, output, img_gts, mask):
        """Backward pass of the network.
        TODO: You can also desgin your own loss function.
        """
        img_gts = img_gts.squeeze(0)
        mask = mask.squeeze(0)

        if self.mask_weight > 0.0:
            mask = (mask > 0.5).float()
        else:
            mask = torch.ones_like(mask)

        mask_sum = mask.sum() + 1e-5

        # Loss
        color_error = (output["color_fine"] - img_gts) * mask
        color_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
        psnr = 20.0 * torch.log10(1.0 / (((output["color_fine"] - img_gts)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

        eikonal_loss = output["gradient_error"]

        mask_loss = F.binary_cross_entropy(output["weight_sum"].clip(1e-3, 1.0 - 1e-3), mask)

        loss = color_loss + \
            eikonal_loss * self.igr_weight + \
        mask_loss * self.mask_weight

        self.log_dict['rgb_loss'] += color_loss.item()
        self.log_dict['total_loss'] += loss.item()

        loss.backward()

    def step(self, data):
        """
        A signle training step.
        """

        # Get rays, and put them on the device
        ray_orig = data['rays'][..., :3].to(self.device)
        ray_dir = data['rays'][..., 3:].to(self.device)
        img_gts = data['imgs'].to(self.device)
        mask = data['masks'].to(self.device)

        full_imgs = data['full_imgs'].to(self.device)
        ref_poses = data['ref_poses']

        # breakpoint()

        bg_rgb = torch.zeros([1, 3]).type(torch.float32).to(self.device)

        self.optimizer.zero_grad()

        output = self.forward(ray_orig, ray_dir, full_imgs, bg_rgb, ref_poses)
        self.backward(output, img_gts, mask)
        
        self.optimizer.step()
        self.log_dict['total_iter_count'] += 1
        self.log_dict['image_count'] += ray_orig.shape[0]

        # Returning conditional volume and mask for the validation and test
        return output['conditional_volume'].detach(), output['conditional_volume_mask'].detach()

    def render(self, ray_orig, ray_dir, bg_rgb=None):
        """Render a full image for evaluation.
        """
        ray_orig = ray_orig.squeeze(0)
        ray_dir = ray_dir.squeeze(0)

        # print(ray_orig.shape)     torch.Size([1, 512, 3])
        # print(ray_dir.shape)      torch.Size([1, 512, 3])
        # print(img_gts.shape)      torch.Size([1, 512, 3])

        # Step 1 : Sample points along the rays
        points, z_vals, sample_dist = self.sample_points(
            ray_orig, ray_dir, near=self.cfg.near, far=self.cfg.far
        )

        # Step 2 : Predict radiance and volume density at the sampled points
        # Conditional volume and mask are already computed for rendering, less computation overhead
        output = self.renderer(
            points, ray_orig, ray_dir, z_vals, sample_dist, 
            bg_rgb, self.renderer.conditional_volume, self.renderer.conditional_volume_mask
        )
        return output["color_fine"]

    def reconstruct_3D(self, save_dir, epoch=0, sdf_threshold=0., chunk_size=8192):
        """
        Reconstruct the 3D shape from the volume density.
        """

        conditional_volume, conditional_volume_mask = \
            self.renderer.conditional_volume, self.renderer.conditional_volume_mask

        # Mesh evaluation
        window_x = torch.linspace(-1., 1., steps=RES, device=self.device)
        window_y = torch.linspace(-1., 1., steps=RES, device=self.device)
        window_z = torch.linspace(-1., 1., steps=RES, device=self.device)
        
        coord = torch.stack(torch.meshgrid(window_x, window_y, window_z)).permute(1, 2, 3, 0).reshape(-1, 3).contiguous()

        _points = torch.split(coord, int(chunk_size), dim=0)
        sdf_vals = []
        
        for _p in _points:
            sdf_output = self.renderer.sdf_net.sdf(_p, conditional_volume)
            sdf_vals.append(sdf_output['sdf_pts'].detach().cpu())
        sdf_vals = torch.cat(sdf_vals, dim=0)

        np_sdf_vals = sdf_vals.reshape(RES, RES, RES).cpu().numpy()

        vertices, faces = mcubes.marching_cubes(np_sdf_vals, sdf_threshold)
        #vertices = ((vertices - 0.5) / (res/2)) - 1.0
        vertices = (vertices / (RES-1)) * 2.0 - 1.0
        h = trimesh.Trimesh(vertices=vertices, faces=faces)
        h.export(os.path.join(save_dir, '%04d.obj' % (epoch)))

    def log(self, step, epoch):
        """
        Log the training information.
        """
        log_text = 'STEP {} - EPOCH {}/{}'.format(step, epoch, self.cfg.epochs)
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])
        self.log_dict['rgb_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['rgb_loss'])

        log.info(log_text)

        for key, value in self.log_dict.items():
            if 'loss' in key:
                wandb.log({key: value}, step=step)
        self.init_log_dict()

    def validate(self, loader, img_shape, step=0, epoch=0, sdf_threshold=0., chunk_size=8192, save_img=False):
        """
        Validation function for generating final results.
        """
        if torch.cuda.is_available(): torch.cuda.empty_cache()  # To avoid CUDA out of memory
        self.eval()

        log.info("Beginning validation...")
        log.info(f"Loaded validation dataset with {len(loader)} images at resolution {img_shape[0]}x{img_shape[1]}")

        log.info(f"Saving reconstruction result to {self.valid_mesh_dir}")
        if not os.path.exists(self.valid_mesh_dir):
            os.makedirs(self.valid_mesh_dir)

        if save_img:
            self.valid_img_dir = os.path.join(self.log_dir, "img")
            log.info(f"Saving rendering result to {self.valid_img_dir}")
            if not os.path.exists(self.valid_img_dir):
                os.makedirs(self.valid_img_dir)

        psnr_total = 0.0

        wandb_img = []
        wandb_img_gt = []

        # Using white background again
        bg_rgb = torch.ones([1, 3]).to(self.device)

        with torch.no_grad():
            # Evaluate 3D reconstruction
            self.reconstruct_3D(
                self.valid_mesh_dir, epoch=epoch,
                sdf_threshold=sdf_threshold, chunk_size=chunk_size
            )

        # Evaluate 2D novel view rendering
        for i, data in enumerate(tqdm(loader)):
            rays = data['rays'].to(self.device)          # [1, Nr, 6]
            img_gt = data['imgs'].to(self.device)        # [1, Nr, 3]
            # mask = data['masks'].repeat(1, 1, 3).to(self.device)
            #TODO: add data['ref_poses']['query_c2w'] to be used for projector in validation
            _rays = torch.split(rays, int(chunk_size), dim=1)
            pixels = []
            for _r in _rays:
                ray_orig = _r[..., :3]          # [1, chunk, 3]
                ray_dir = _r[..., 3:]           # [1, chunk, 3]

                # One forward pass to calculate the color per pixel
                ray_rgb = self.render(ray_orig, ray_dir, bg_rgb=bg_rgb).unsqueeze(0)
                pixels.append(ray_rgb.detach())

            pixels = torch.cat(pixels, dim=1)

            psnr_total += psnr(pixels, img_gt)

            img = (pixels).reshape(*img_shape, 3).cpu().numpy() * 255
            gt = (img_gt).reshape(*img_shape, 3).cpu().numpy() * 255
            wandb_img.append(wandb.Image(img))
            wandb_img_gt.append(wandb.Image(gt))

            if save_img:
                Image.fromarray(gt.astype(np.uint8)).save(
                    os.path.join(self.valid_img_dir, "gt-{:04d}-{:03d}.png".format(epoch, i)) )
                Image.fromarray(img.astype(np.uint8)).save(
                    os.path.join(self.valid_img_dir, "img-{:04d}-{:03d}.png".format(epoch, i)) )

        wandb.log({"Rendered Images": wandb_img}, step=step)
        wandb.log({"Ground-truth Images": wandb_img_gt}, step=step)
                
        psnr_total /= len(loader)

        log_text = 'EPOCH {}/{}'.format(epoch, self.cfg.epochs)
        log_text += ' {} | {:.2f}'.format(f"PSNR", psnr_total)

        wandb.log({'PSNR': psnr_total, 'Epoch': epoch}, step=step)
        log.info(log_text)
        self.train()

    def save_model(self, epoch, conditional_volume, conditional_volume_mask):
        """
        Save the model checkpoint.
        TODO: Save the optimizer and lr scheduler as well!
        """
        
        # Saving the conditional volume and mask to prevent computation overhead!
        self.renderer.conditional_volume = conditional_volume
        self.renderer.conditional_volume_mask = conditional_volume_mask

        fname = os.path.join(self.log_dir, f'model-{epoch}.pth')
        log.info(f'Saving model checkpoint to: {fname}')
        torch.save(self.renderer, fname)
    