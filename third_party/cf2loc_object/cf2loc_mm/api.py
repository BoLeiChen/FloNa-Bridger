"""Standalone navigation-facing API. No GT target/episode data is accepted."""
import torch
import numpy as np
from PIL import Image
from .encoders import FrozenFeatures,image_tensor
from .model import MultimodalLocalizer,FORMAT
from .prepare import category_name
from .metrics import rank_candidates

class Localizer:
    def __init__(self,checkpoint,device='cpu',backbone_paths=None):
        package=torch.load(checkpoint,map_location='cpu')
        if package.get('format')!=FORMAT:raise ValueError('Not a multimodal CF2Loc checkpoint; old single-image weights are incompatible')
        self.config=dict(package['config'])
        if backbone_paths:
            if set(backbone_paths)-{'clip_checkpoint','depth_checkpoint'}:raise ValueError('Only frozen checkpoint paths may be overridden')
            self.config.update(backbone_paths)
        self.device=torch.device(device);self.features=FrozenFeatures(self.config,self.device)
        self.model=MultimodalLocalizer(self.config).to(self.device).eval();self.model.load_state_dict(package['model'],strict=True)

    @torch.no_grad()
    def predict(self,floorplan,*,origin_xy,extent_xy,image=None,category=None,num_candidates=64,top_k=16,seed=0):
        if (image is None)==(category is None):raise ValueError('Supply exactly one image or object category')
        if image is not None and self.config['setting']=='object_only':raise ValueError('object_only checkpoint does not support images')
        origin=np.asarray(origin_xy,dtype=float);extent=np.asarray(extent_xy,dtype=float)
        if origin.shape!=(2,) or extent.shape!=(2,) or not np.isfinite(origin).all() or not np.isfinite(extent).all() or (extent<=0).any():raise ValueError('Invalid world map calibration')
        if not 1<=top_k<=num_candidates:raise ValueError('top_k must be in [1,num_candidates]')
        batch=dict(task='image' if image is not None else 'object',floorplan=image_tensor(floorplan,256)[None].to(self.device),origin=torch.tensor(origin,dtype=torch.float32,device=self.device)[None],extent=torch.tensor(extent,dtype=torch.float32,device=self.device)[None])
        if image is None:batch['text_features']=self.features.text([category_name(category)])
        else:batch.update(self.features.image(image_tensor(image,224)[None].to(self.device)))
        generator=torch.Generator(device=self.device).manual_seed(seed)
        poses=self.model.sample(batch,num_candidates,int(self.config.get('sample_steps',20)),generator)[0].cpu().numpy()
        ranking_mode=self.config.get('candidate_ranking','pose')
        indices,weights=rank_candidates(poses,batch['task'],ranking_mode=ranking_mode);indices=indices[:top_k];weights=weights[:top_k]
        weights=weights/weights.sum()
        return dict(task=batch['task'],world_xy=poses[indices,:2].tolist(),world_yaw=poses[indices,2].tolist() if image is not None else None,weights=weights.tolist(),confidence_kind='uncalibrated_sample_density',raw_world_xy=poses[:,:2].tolist(),requested_top_k=top_k,returned_top_k=len(indices),ranking='KDE_then_NMS',candidate_ranking=ranking_mode,coordinate_convention='world_xy_meters_same_axes_as_floorplan',requires_reachability_check=True)
