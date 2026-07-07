# tests/test_model.py
import torch
from yolov1.model import YOLOv1
from config import S, B, C


def test_model_output_shape():
    """Verify the model output tensor has the correct grid dimensions and channel depth."""
    # Instantiate without pre-trained weights to keep test execution fast
    model = YOLOv1(pretrained=False)
    
    # Simulate a batch of 2 images
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    # Expected channels: B * 5 + C (for B=1, C=11, this is 16 channels)
    expected_channels = B * 5 + C
    assert output.shape == (2, expected_channels, S, S)


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