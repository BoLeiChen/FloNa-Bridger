import torch
from localization.disco import DisCoMatcher


def test_new_map_encoder_updates_statistics_and_trains_without_weights():
    torch.set_num_threads(2)
    model=DisCoMatcher({'feature_dim':128,'map_encoder_initialization':'random'}).train()
    batch_norms=[m for m in model.local_map_encoder.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    assert batch_norms and all(m.training for m in batch_norms)
    before=[m.num_batches_tracked.clone() for m in batch_norms]
    scores=model(torch.randn(2,256,128),torch.rand(2,2,1,128,128))
    scores.square().mean().backward()
    assert all(int(m.num_batches_tracked)==int(n)+1 for m,n in zip(batch_norms,before))
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.local_map_encoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    assert all(not m.training for m in batch_norms)


def test_legacy_checkpoint_configuration_preserves_frozen_statistics():
    model=DisCoMatcher({'feature_dim':128}).train()
    norms=[m for m in model.local_map_encoder.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    assert norms and all(not m.training for m in norms)
