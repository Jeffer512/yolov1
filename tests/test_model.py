import torch
from yolov1.model import YOLOv1
from config import IMAGE_HEIGHT, IMAGE_WIDTH, S_H, S_W, B, C


def test_model_output_shape():
    """Verify the model output tensor has the correct grid dimensions and channel depth."""
    model = YOLOv1(pretrained=False)
    
    #  FIX: Simulate a batch using your actual configured image dimensions!
    dummy_input = torch.randn(2, 3, IMAGE_HEIGHT, IMAGE_WIDTH) # (2, 3, 288, 448)
    output = model(dummy_input)
    
    # Expected channels: B * 5 + C = 16
    expected_channels = B * 5 + C
    
    #  FIX: Assert shape matches the asymmetric grid layout!
    assert output.shape == (2, expected_channels, S_H, S_W) # (2, 16, 9, 14)


def test_freeze_backbone():
    """Verify that freeze_backbone correctly toggles parameter gradient tracking."""
    model = YOLOv1(pretrained=False)
    
    # Stage 1: Freeze entire backbone (only head is active)
    model.freeze_backbone(True)
    for param in model.backbone.parameters():
        assert not param.requires_grad
    for param in model.head.parameters():
        assert param.requires_grad

    # Stage 2: Unfreeze layer4 for fine-tuning
    model.freeze_backbone(False)
    
    # Final residual block (layer4) should be trainable
    for param in model.backbone[-1].parameters():
        assert param.requires_grad
        
    # Earlier residual blocks (layers 1-3) must remain frozen
    for param in model.backbone[:-1].parameters():
        assert not param.requires_grad