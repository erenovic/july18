
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


def grid_sample_3d(volume, optical):
    """
    bilinear sampling cannot guarantee continuous first-order gradient
    mimic pytorch grid_sample function
    The 8 corner points of a volume noted as: 4 points (front view); 4 points (back view)
    fnw (front north west) point
    bse (back south east) point
    :param volume: [B, C, X, Y, Z]
    :param optical: [B, x, y, z, 3]
    :return:
    """
    N, C, ID, IH, IW = volume.shape
    _, D, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]
    iz = optical[..., 2]

    ix = ((ix + 1) / 2) * (IW - 1)
    iy = ((iy + 1) / 2) * (IH - 1)
    iz = ((iz + 1) / 2) * (ID - 1)

    mask_x = (ix > 0) & (ix < IW)
    mask_y = (iy > 0) & (iy < IH)
    mask_z = (iz > 0) & (iz < ID)

    mask = mask_x & mask_y & mask_z  # [B, x, y, z]
    mask = mask[:, None, :, :, :].repeat(1, C, 1, 1, 1)  # [B, C, x, y, z]

    with torch.no_grad():
        # back north west
        ix_bnw = torch.floor(ix)
        iy_bnw = torch.floor(iy)
        iz_bnw = torch.floor(iz)

        ix_bne = ix_bnw + 1
        iy_bne = iy_bnw
        iz_bne = iz_bnw

        ix_bsw = ix_bnw
        iy_bsw = iy_bnw + 1
        iz_bsw = iz_bnw

        ix_bse = ix_bnw + 1
        iy_bse = iy_bnw + 1
        iz_bse = iz_bnw

        # front view
        ix_fnw = ix_bnw
        iy_fnw = iy_bnw
        iz_fnw = iz_bnw + 1

        ix_fne = ix_bnw + 1
        iy_fne = iy_bnw
        iz_fne = iz_bnw + 1

        ix_fsw = ix_bnw
        iy_fsw = iy_bnw + 1
        iz_fsw = iz_bnw + 1

        ix_fse = ix_bnw + 1
        iy_fse = iy_bnw + 1
        iz_fse = iz_bnw + 1

    # back view
    bnw = (ix_fse - ix) * (iy_fse - iy) * (iz_fse - iz)  # smaller volume, larger weight
    bne = (ix - ix_fsw) * (iy_fsw - iy) * (iz_fsw - iz)
    bsw = (ix_fne - ix) * (iy - iy_fne) * (iz_fne - iz)
    bse = (ix - ix_fnw) * (iy - iy_fnw) * (iz_fnw - iz)

    # front view
    fnw = (ix_bse - ix) * (iy_bse - iy) * (iz - iz_bse)  # smaller volume, larger weight
    fne = (ix - ix_bsw) * (iy_bsw - iy) * (iz - iz_bsw)
    fsw = (ix_bne - ix) * (iy - iy_bne) * (iz - iz_bne)
    fse = (ix - ix_bnw) * (iy - iy_bnw) * (iz - iz_bnw)

    with torch.no_grad():
        # back view
        torch.clamp(ix_bnw, 0, IW - 1, out=ix_bnw)
        torch.clamp(iy_bnw, 0, IH - 1, out=iy_bnw)
        torch.clamp(iz_bnw, 0, ID - 1, out=iz_bnw)

        torch.clamp(ix_bne, 0, IW - 1, out=ix_bne)
        torch.clamp(iy_bne, 0, IH - 1, out=iy_bne)
        torch.clamp(iz_bne, 0, ID - 1, out=iz_bne)

        torch.clamp(ix_bsw, 0, IW - 1, out=ix_bsw)
        torch.clamp(iy_bsw, 0, IH - 1, out=iy_bsw)
        torch.clamp(iz_bsw, 0, ID - 1, out=iz_bsw)

        torch.clamp(ix_bse, 0, IW - 1, out=ix_bse)
        torch.clamp(iy_bse, 0, IH - 1, out=iy_bse)
        torch.clamp(iz_bse, 0, ID - 1, out=iz_bse)

        # front view
        torch.clamp(ix_fnw, 0, IW - 1, out=ix_fnw)
        torch.clamp(iy_fnw, 0, IH - 1, out=iy_fnw)
        torch.clamp(iz_fnw, 0, ID - 1, out=iz_fnw)

        torch.clamp(ix_fne, 0, IW - 1, out=ix_fne)
        torch.clamp(iy_fne, 0, IH - 1, out=iy_fne)
        torch.clamp(iz_fne, 0, ID - 1, out=iz_fne)

        torch.clamp(ix_fsw, 0, IW - 1, out=ix_fsw)
        torch.clamp(iy_fsw, 0, IH - 1, out=iy_fsw)
        torch.clamp(iz_fsw, 0, ID - 1, out=iz_fsw)

        torch.clamp(ix_fse, 0, IW - 1, out=ix_fse)
        torch.clamp(iy_fse, 0, IH - 1, out=iy_fse)
        torch.clamp(iz_fse, 0, ID - 1, out=iz_fse)

    # xxx = volume[:, :, iz_bnw.long(), iy_bnw.long(), ix_bnw.long()]
    volume = volume.view(N, C, ID * IH * IW)
    # yyy = volume[:, :, (iz_bnw * ID + iy_bnw * IW + ix_bnw).long()]

    # back view
    bnw_val = torch.gather(volume, 2,
                           (iz_bnw * ID ** 2 + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(volume, 2,
                           (iz_bne * ID ** 2 + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(volume, 2,
                           (iz_bsw * ID ** 2 + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(volume, 2,
                           (iz_bse * ID ** 2 + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    # front view
    fnw_val = torch.gather(volume, 2,
                           (iz_fnw * ID ** 2 + iy_fnw * IW + ix_fnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    fne_val = torch.gather(volume, 2,
                           (iz_fne * ID ** 2 + iy_fne * IW + ix_fne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    fsw_val = torch.gather(volume, 2,
                           (iz_fsw * ID ** 2 + iy_fsw * IW + ix_fsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    fse_val = torch.gather(volume, 2,
                           (iz_fse * ID ** 2 + iy_fse * IW + ix_fse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    out_val = (
        # back
            bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
            bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
            bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
            bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W) +
            # front
            fnw_val.view(N, C, D, H, W) * fnw.view(N, 1, D, H, W) +
            fne_val.view(N, C, D, H, W) * fne.view(N, 1, D, H, W) +
            fsw_val.view(N, C, D, H, W) * fsw.view(N, 1, D, H, W) +
            fse_val.view(N, C, D, H, W) * fse.view(N, 1, D, H, W)

    )

    # * zero padding
    out_val = torch.where(mask, out_val, torch.zeros_like(out_val).float().to(out_val.device))

    return out_val