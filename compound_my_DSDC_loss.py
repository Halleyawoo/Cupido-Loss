# 代码说明
# Dice 损失：
# 衡量分割区域的重叠程度，反映整体拓扑完整性。

# 骨架损失：
# 使用 skeletonize_3d 提取目标的中心线，并将其与预测结果进行对比。

# 方向性损失：
# 利用 3D Sobel 算子提取方向向量，约束预测与目标方向场的一致性。

# 连通性损失：
# 惩罚小的伪连通区域，增强结构的连贯性。

import torch
import torch.nn.functional as F
from torch import nn
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.cldice_loss import SoftclDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
import torch
from monai.transforms import distance_transform_edt
from nnunetv2.training.loss.skeletonize import Skeletonize
from nnunetv2.training.loss.soft_skeleton import SoftSkeletonize


def get_weights(mask_input, skel_input, dim, prob_flag=True):
    if prob_flag:
        mask_prob = mask_input
        skel_prob = skel_input

        mask = (mask_prob > 0.5).int()
        skel = (skel_prob > 0.5).int()
    else:
        mask = mask_input
        skel = skel_input
    
    distances = distance_transform_edt(mask).float()

    smooth = 1e-7
    distances[mask == 0] = 0
    
    skel_radius = torch.zeros_like(distances, dtype=torch.float32)
    skel_radius[skel == 1] = distances[skel == 1]
    
    
    dist_map_norm = torch.zeros_like(distances, dtype=torch.float32) 
    skel_R_norm = torch.zeros_like(skel_radius, dtype=torch.float32) 
    I_norm = torch.zeros_like(mask, dtype=torch.float32)

    
    for i in range(skel_radius.shape[0]):
        distances_i = distances[i]
        skel_i = skel_radius[i]
        skel_radius_max = max(skel_i.max(), 1) 
        skel_radius_min = max(skel_i.min(), 1)

        distances_i[distances_i > skel_radius_max] = skel_radius_max
        dist_map_norm[i] = distances_i / skel_radius_max
        skel_R_norm[i] = skel_i / skel_radius_max

        if dim == 2:
            I_norm[i] = (skel_radius_max - skel_i + skel_radius_min) / skel_radius_max
        else:
            I_norm[i] = ((skel_radius_max - skel_i + skel_radius_min) / skel_radius_max) ** 2
    
    I_norm[skel == 0] = 0 # 0 for non-skeleton pixels

    if prob_flag:
        return dist_map_norm * mask_prob, skel_R_norm * mask_prob, I_norm * skel_prob
    else:
        return dist_map_norm * mask, skel_R_norm * mask, I_norm * skel


def combine_tensors(A, B, C):
    """
    Combine tensors A, B, C for dynamic weight adjustment.
    """
    A_C = A * C
    B_C = B * C
    D = B_C.clone()
    mask_AC = (A != 0) & (B == 0)
    D[mask_AC] = A_C[mask_AC]
    return D


class CombinedLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, cldc_kwargs,
                 weight_dice=1, weight_ce=1, weight_cldice=1,
                 weight_direction=1, weight_connectivity=1, weight_union=1,
                 ignore_label=None, dice_class=SoftDiceLoss):
        super(CombinedLoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_cldice = weight_cldice
        self.weight_direction = weight_direction
        self.weight_connectivity = weight_connectivity
        self.weight_union = weight_union

        self.ignore_label = ignore_label

        # Topology-preserving skeletonization
        self.t_skeletonize = Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')

        # Morphological skeletonization
        self.m_skeletonize = SoftSkeletonize(num_iter=10)  # 默认迭代次数

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        self.cldice = SoftclDiceLoss(**cldc_kwargs)

    def general_union_loss(self, pred, target, weight):
        """
        General Union Loss using a provided weight.
        """
        smooth = 1.0
        alpha = 0.1
        beta = 1 - alpha
        small_smooth = 0.0001

        intersection = (weight * ((pred + small_smooth) ** 0.7) * target).sum()
        all_union = (weight * (alpha * pred + beta * target)).sum()

        return 1 - (intersection + smooth) / (all_union + smooth)

    def union_loss(self, y_pred, y_true, t_skeletonize_flage=False):
        """
        Compute the union loss based on general_union_loss.
        """
        if len(y_true.shape) == 4:
            dim = 2
        elif len(y_true.shape) == 5:
            dim = 3
        else:
            raise ValueError("y_true should be 4D or 5D tensor.")

        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0]  # C foreground channels -> 1 channel
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1]  # predicted probability map of foreground

        with torch.no_grad():
            y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float()  # ground truth of foreground
            y_pred_hard = (y_pred_prob > 0.5).float()

            if t_skeletonize_flage:
                skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.t_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            else:
                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.m_skeletonize(y_true.unsqueeze(1)).squeeze(1)

        skel_pred_prob = skel_pred_hard * y_pred_prob
        # dist_map_norm * mask, skel_R_norm * mask, I_norm * skel
        q_vl, q_slvl, q_sl = get_weights(y_true, skel_true, dim, prob_flag=False)
        q_vp, q_spvp, q_sp = get_weights(y_pred_prob, skel_pred_prob, dim, prob_flag=True)

        # def general_union_loss(self, pred, target, weight):
        w_tprec = self.general_union_loss(q_sp, q_vl, q_sp)
        w_tsens = self.general_union_loss(q_vp, q_sl, q_sl)

        return w_tprec + w_tsens


    # 对整张 3D 图像（所有体素）都计算了梯度并求方向。
    def directional_loss(self, pred, target):
        """
        Directional loss constrains the consistency between predicted and ground truth gradient directions.

        Args:
            pred (torch.Tensor): The prediction tensor of shape (B, C_pred, D, H, W).
            target (torch.Tensor): The ground truth tensor of shape (B, C_target, D, H, W).

        Returns:
            torch.Tensor: The computed directional loss value.
        """
        # Define Sobel filters for gradient calculation
        # Sobel filter for prediction (2-channel input)
        # 计算 X 方向的梯度
        sobel_x_pred = torch.tensor(
            [[[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
            [[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]
            ]).to(pred.device).float()

        # Sobel filter for target (1-channel input)
        sobel_x_target = torch.tensor(
            [[[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]]
            ).to(target.device).float()

        # Sobel filters for y and z directions (for pred and target)
        # 交换索引 2 和 3，使过滤器变成 Y 方向
        sobel_y_pred = sobel_x_pred.permute(0, 1, 3, 2, 4)
        # 交换索引 2 和 4，变成 Z 方向
        sobel_z_pred = sobel_x_pred.permute(0, 1, 4, 2, 3)

        sobel_y_target = sobel_x_target.permute(0, 1, 3, 2, 4)
        sobel_z_target = sobel_x_target.permute(0, 1, 4, 2, 3)

        # 计算 pred 和 target 的梯度
        # Compute gradients for prediction
        grad_pred_x = F.conv3d(pred, sobel_x_pred, padding=1)
        grad_pred_y = F.conv3d(pred, sobel_y_pred, padding=1)
        grad_pred_z = F.conv3d(pred, sobel_z_pred, padding=1)

        # Compute gradients for target
        grad_target_x = F.conv3d(target, sobel_x_target, padding=1)
        grad_target_y = F.conv3d(target, sobel_y_target, padding=1)
        grad_target_z = F.conv3d(target, sobel_z_target, padding=1)

        # 归一化梯度，计算方向向量
        # Normalize gradients to compute direction vectors
        # 组合 X, Y, Z 方向梯度
        grad_pred = torch.stack([grad_pred_x, grad_pred_y, grad_pred_z], dim=-1)
        grad_target = torch.stack([grad_target_x, grad_target_y, grad_target_z], dim=-1)

        # 归一化梯度方向
        # 进行单位化 梯度仅表示方向 不受大小的影响
        grad_pred_norm = F.normalize(grad_pred, dim=-1)
        grad_target_norm = F.normalize(grad_target, dim=-1)

        # Compute cosine similarity
        # 计算方向一致性（余弦相似度）
        # 结果范围 [0, 1]：
        # 1 代表方向 完全一致
        # 0 代表 完全相反
        cos_sim = F.cosine_similarity(grad_pred_norm, grad_target_norm, dim=-1)

        # Directional loss
        # 计算最终方向损失
        direction_loss = 1 - cos_sim.mean()

        return direction_loss


    def connectivity_loss(self, pred, target):
        """
        Connectivity loss penalizes small pseudo-connected regions to improve structure coherence.
        """
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()

        # Compute differences between connected regions
        diff = torch.abs(pred_binary - target_binary)
        connectivity_loss = diff.mean()
        return connectivity_loss
    
    def forward(self, net_output: torch.Tensor, target: torch.Tensor, t_skeletonize_flage=False):
        """
        Compute the combined loss.

        :param net_output: Model output
        :param target: Ground truth labels
        :param t_skeletonize_flage: Flag for skeletonization method
        :return: Combined loss value
        """
        mask = None  # 默认值为 None

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = (target != self.ignore_label).bool()
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target

        # Dice loss
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dice != 0 else 0
        # Cross-entropy loss
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0
        # Skeleton loss
        cldice_loss = self.cldice(net_output, target, t_skeletonize_flage=t_skeletonize_flage) if self.weight_cldice != 0 else 0
        # Directional loss
        dir_loss = self.directional_loss(net_output, target) if self.weight_direction != 0 else 0
        # Connectivity loss
        conn_loss = self.connectivity_loss(net_output, target) if self.weight_connectivity != 0 else 0
        # Union loss
        union_loss = self.union_loss(net_output, target, t_skeletonize_flage=t_skeletonize_flage) if self.weight_union != 0 else 0

        # Combine all losses
        result = (self.weight_ce * ce_loss +
                self.weight_dice * dc_loss +
                self.weight_cldice * cldice_loss +
                self.weight_direction * dir_loss +
                self.weight_connectivity * conn_loss +
                self.weight_union * union_loss)

        return result


    
