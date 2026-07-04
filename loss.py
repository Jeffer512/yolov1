import torch
import torch.nn as nn
from config import S, LAMBDA_COORD, LAMBDA_NOOBJ
from utils import iou_vectorized


class YOLOv1Loss(nn.Module):
    """Vectorized YOLOv1 Loss Function.
    
    Calculates multi-part loss including coordinate regression, confidence 
    scores, and class probabilities using vectorized operations.
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        """Computes the total loss between model predictions and targets.

        Args:
            predictions (Tensor): Raw model outputs of shape (B, B*5 + C, S, S).
            targets (Tensor): Ground truth target tensors of shape (B, S, S, B*5 + C).

        Returns:
            Tensor: Scalar normalized total loss.
        """
        # Permute predictions to channel-last format (B, S, S, depth) to match target layout
        pred = predictions.permute(0, 2, 3, 1).contiguous()
        target = targets.contiguous()

        # Identify cells containing objects (p_c > 0)
        obj_mask = target[..., 0] > 0
        noobj_mask = ~obj_mask
        obj_mask_f = obj_mask.float()

        # Apply bounding activations to raw model outputs
        pred_conf = torch.sigmoid(pred[..., 0])
        pred_xy = torch.sigmoid(pred[..., 1:3])
        pred_wh = torch.exp(pred[..., 3:5])
        pred_classes = torch.softmax(pred[..., 5:], dim=-1)

        target_xy = target[..., 1:3]
        target_wh = target[..., 3:5]
        target_classes = target[..., 5:]

        # --- 1. Box coordinate loss ---

        # Center coordinates (x, y) MSE loss
        xy_loss = (pred_xy - target_xy).pow(2).sum(-1)
        xy_loss = (xy_loss * obj_mask_f).sum()

        # Width and height MSE loss utilizing square roots
        pred_wh_sqrt = torch.sqrt(pred_wh + 1e-6)
        target_wh_sqrt = torch.sqrt(target_wh + 1e-6)
        wh_loss = (pred_wh_sqrt - target_wh_sqrt).pow(2).sum(-1)
        wh_loss = (wh_loss * obj_mask_f).sum()

        box_loss = LAMBDA_COORD * (xy_loss + wh_loss)

        # --- 2. Confidence loss (target = IoU(pred, truth) for obj cells, 0 for noobj) ---

        # Generate spatial coordinate grids to map cell-relative centers to absolute image-relative values

        device = pred.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(S, device=device),
            torch.arange(S, device=device),
            indexing='ij',
        )

        # Expand spatial index matrices to match batch size
        grid_x = grid_x.unsqueeze(0).expand_as(pred_xy[..., 0])
        grid_y = grid_y.unsqueeze(0).expand_as(pred_xy[..., 1])
        
        # Stack indices along a new 4th dimension to match pred_xy shape: (batch_size, S, S, 2)
        grid_xy = torch.stack([grid_x, grid_y], dim=-1)

        # Convert center coordinates from cell-relative to absolute image-relative
        pred_abs_xy = (pred_xy + grid_xy.float()) / S
        target_abs_xy = (target_xy + grid_xy.float()) / S

        # Concatenate absolute coordinates (batch_size, S, S, 4)
        pred_box_abs = torch.cat([pred_abs_xy, pred_wh], dim=-1)
        target_box_abs = torch.cat([target_abs_xy, target_wh], dim=-1)

        # Calculate vectorized IoUs (squeezed to match mask dimensions)
        iou = iou_vectorized(pred_box_abs, target_box_abs).squeeze(-1)

        # Confidence loss for occupied cells (target is real-time IoU)
        conf_target = iou * obj_mask_f

        obj_conf_loss = ((pred_conf - conf_target).pow(2) * obj_mask_f).sum()

        # Confidence loss for unoccupied cells (target is 0.0)
        noobj_conf_loss = ((pred_conf - 0).pow(2) * noobj_mask.float()).sum()
        noobj_conf_loss = LAMBDA_NOOBJ * noobj_conf_loss

        # --- 3. Classification loss ---
        
        class_loss = ((pred_classes - target_classes).pow(2).sum(-1) * obj_mask_f).sum()

        total_loss = box_loss + obj_conf_loss + noobj_conf_loss + class_loss
        
        # Normalize the loss by the batch size to keep gradient updates stable
        return total_loss / predictions.size(0)       