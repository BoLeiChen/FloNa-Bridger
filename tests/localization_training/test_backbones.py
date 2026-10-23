"""Actual frozen encoders; run explicitly with CF2LOC_REAL_WEIGHTS=1."""
import os
import pytest
import torch
from cf2loc_mm.encoders import FrozenFeatures,CLIP_MEAN,CLIP_STD

@pytest.mark.skipif(os.getenv('CF2LOC_REAL_WEIGHTS')!='1',reason='requires installed local pretrained files')
def test_real_clip_geometry_features_and_freezing():
    cfg=dict(setting='joint_geometry',clip_checkpoint='/mnt/data/cf2loc_multimodal/pretrained/ViT-B-16.pt',depth_checkpoint='/mnt/data/cf2loc_multimodal/pretrained/depth_anything_v2_vits.pth')
    f=FrozenFeatures(cfg);f.train();assert not f.training and not f.clip.training and not f.geometry.training
    text=f.text(['chair','bed']);assert text.shape==(2,512)
    rgb=torch.rand(2,3,224,224);image=f.image(rgb)
    assert image['image_features'].shape==(2,50,512)
    assert image['geometry_features'].shape==(2,49,384)
    assert not any(x.requires_grad for x in f.parameters())
    assert all(torch.isfinite(x).all() for x in image.values())
    normalized=(rgb-rgb.new_tensor(CLIP_MEAN)[None,:,None,None])/rgb.new_tensor(CLIP_STD)[None,:,None,None]
    with torch.no_grad():official=f.clip.encode_image(normalized).float()
    torch.testing.assert_close(image['image_features'][:,0],official)
