
import torch
from torch.nn.functional import grid_sample


def generate_grid(n_vox, interval):
    """
    generate grid
    if 3D volume, grid[:,:,x,y,z]  = (x,y,z)
    :param n_vox:
    :param interval:
    :return:
    """
    with torch.no_grad():
        # Create voxel grid
        grid_range = [torch.arange(0, n_vox[axis], interval) for axis in range(3)]
        grid = torch.stack(torch.meshgrid(grid_range[0], grid_range[1], grid_range[2]))  # 3 dx dy dz
        # ! don't create tensor on gpu; imbalanced gpu memory in ddp mode
        grid = grid.unsqueeze(0).type(torch.float32)  # 1 3 dx dy dz

    return grid


def back_project_sparse_type(coords, origin, voxel_size, feats, KRcam, 
                             sizeH=None, sizeW=None, only_mask=False,
                             with_proj_z=False):
    # - modified version from NeuRecon
    '''
    Unproject the image fetures to form a 3D (sparse) feature volume

    :param coords: coordinates of voxels,
    dim: (num of voxels, 4) (4 : batch ind, x, y, z)
    :param origin: origin of the partial voxel volume (xyz position of voxel (0, 0, 0))
    dim: (batch size, 3) (3: x, y, z)
    :param voxel_size: floats specifying the size of a voxel
    :param feats: image features
    dim: (num of views, batch size, C, H, W)
    :param KRcam: projection matrix
    dim: (num of views, batch size, 4, 4)
    :return: feature_volume_all: 3D feature volumes
    dim: (num of voxels, num_of_views, c)
    :return: mask_volume_all: indicate the voxel of sampled feature volume is valid or not
    dim: (num of voxels, num_of_views)
    '''
    
    n_views, bs, c, h, w = feats.shape
    device = feats.device

    if sizeH is None:
        sizeH, sizeW = h, w  # - if the KRcam is not suitable for the current feats

    feature_volume_all = torch.zeros(coords.shape[0], n_views, c).to(device)
    mask_volume_all = torch.zeros([coords.shape[0], n_views], dtype=torch.int32).to(device)

    for batch in range(bs):
        batch_ind = torch.nonzero(coords[:, 0] == batch).squeeze(1).to(device)
        coords_batch = coords[batch_ind][:, 1:]

        coords_batch = coords_batch.view(-1, 3)
        origin_batch = origin[batch].unsqueeze(0)
        feats_batch = feats[:, batch]
        proj_batch = KRcam[:, batch]

        grid_batch = coords_batch * voxel_size + origin_batch.float()
        rs_grid = grid_batch.unsqueeze(0).expand(n_views, -1, -1)
        rs_grid = rs_grid.permute(0, 2, 1).contiguous()
        nV = rs_grid.shape[-1]
        rs_grid = torch.cat([rs_grid, torch.ones([n_views, 1, nV]).to(device)], dim=1)

        # Project grid
        im_p = proj_batch @ rs_grid  # - transform world pts to image UV space
        im_x, im_y, im_z = im_p[:, 0], im_p[:, 1], im_p[:, 2]
        im_x = im_x / im_z
        im_y = im_y / im_z

        im_grid = torch.stack([2 * im_x / (sizeW - 1) - 1, 2 * im_y / (sizeH - 1) - 1], dim=-1)
        mask = im_grid.abs() <= 1
        mask = (mask.sum(dim=-1) == 2) & (im_z > 0)

        mask = mask.view(n_views, -1)
        mask = mask.permute(1, 0).contiguous()  # [num_pts, nviews]

        mask_volume_all[batch_ind] = mask.to(torch.int32)

        if only_mask:
            return mask_volume_all

        feats_batch = feats_batch.view(n_views, c, h, w)
        im_grid = im_grid.view(n_views, 1, -1, 2)
        features = grid_sample(feats_batch, im_grid, padding_mode='zeros', align_corners=True)

        features = features.view(n_views, c, -1)
        features = features.permute(2, 0, 1).contiguous()  # [num_pts, nviews, c]

        feature_volume_all[batch_ind] = features

        if with_proj_z:
            im_z = im_z.view(n_views, 1, -1).permute(2, 0, 1).contiguous()  # [num_pts, nviews, 1]
            return feature_volume_all, mask_volume_all, im_z

    return feature_volume_all, mask_volume_all


def sparse_to_dense_channel(locs, values, dim, c, default_val, device):
    locs = locs.to(torch.int64)
    dense = torch.full([dim[0], dim[1], dim[2], c], float(default_val), device=device)
    if locs.shape[0] > 0:
        dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
    return dense