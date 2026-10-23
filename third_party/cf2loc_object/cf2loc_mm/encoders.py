"""Frozen CLIP semantics and optional frozen Depth Anything V2 geometry."""
from pathlib import Path
import hashlib
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

CLIP_MEAN=(.48145466,.4578275,.40821073)
CLIP_STD=(.26862954,.26130258,.27577711)

def image_tensor(image, size=224):
    if not isinstance(image,Image.Image):image=Image.fromarray(np.asarray(image,dtype=np.uint8))
    return torch.from_numpy(np.asarray(image.convert('RGB').resize((size,size),Image.Resampling.BICUBIC)).copy()).permute(2,0,1).float()/255

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

class FrozenFeatures(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        import clip
        path=Path(config['clip_checkpoint'])
        if not path.is_file():raise FileNotFoundError('Local CLIP checkpoint required: '+str(path))
        if config.get('clip_sha256') and file_hash(path)!=config['clip_sha256']:raise ValueError('CLIP checkpoint hash mismatch')
        self.clip,_=clip.load(str(path),device='cpu',jit=False)
        self.clip=self.clip.float().eval().requires_grad_(False)
        if not hasattr(self.clip.visual,'class_embedding'):raise ValueError('A CLIP ViT checkpoint is required')
        self.geometry=None
        if config['setting']=='joint_geometry':
            from .vendor.depth_anything_v2.dinov2 import DINOv2
            path=Path(config['depth_checkpoint'])
            if config.get('depth_sha256') and file_hash(path)!=config['depth_sha256']:raise ValueError('Depth checkpoint hash mismatch')
            self.geometry=DINOv2('vits')
            weights=torch.load(path,map_location='cpu')
            self.geometry.load_state_dict({k[len('pretrained.'):]:v for k,v in weights.items() if k.startswith('pretrained.')},strict=True)
            self.geometry.eval().requires_grad_(False)
        self.dim=int(self.clip.text_projection.shape[1]);self.grid=int(config.get('feature_grid',7))
        self.text_cache={};self.to(device)

    def train(self,mode=True):return super().train(False)

    @torch.no_grad()
    def text(self,categories):
        import clip
        missing=sorted(set(categories)-set(self.text_cache))
        if missing:
            # Short class names are normalized at both dataset and inference boundary.
            tokens=clip.tokenize(['a photo of a '+x+'.' for x in missing]).to(next(self.parameters()).device)
            encoded=F.normalize(self.clip.encode_text(tokens).float(),dim=-1)
            for key,value in zip(missing,encoded):self.text_cache[key]=value
        return torch.stack([self.text_cache[x] for x in categories])

    @torch.no_grad()
    def image(self, rgb):
        v=self.clip.visual
        rgb=F.interpolate(rgb,size=(int(v.input_resolution),)*2,mode='bicubic',align_corners=False)
        mean=rgb.new_tensor(CLIP_MEAN)[None,:,None,None];std=rgb.new_tensor(CLIP_STD)[None,:,None,None]
        x=v.conv1((rgb-mean)/std);side=x.shape[-1];x=x.flatten(2).transpose(1,2)
        cls=v.class_embedding.to(x)[None,None].expand(len(x),1,-1)
        x=v.ln_pre(torch.cat([cls,x],1)+v.positional_embedding.to(x))
        x=v.transformer(x.permute(1,0,2)).permute(1,0,2)
        x=v.ln_post(x)@v.proj
        pooled=x[:,:1]
        patch=x[:,1:].transpose(1,2).reshape(len(x),self.dim,side,side)
        patch=F.adaptive_avg_pool2d(patch,(self.grid,self.grid)).flatten(2).transpose(1,2)
        result={'image_features':torch.cat([pooled,patch],1).float()}
        if self.geometry is not None:
            z=F.interpolate(rgb,size=(224,224),mode='bilinear',align_corners=False)
            z=(z-z.new_tensor([.485,.456,.406])[None,:,None,None])/z.new_tensor([.229,.224,.225])[None,:,None,None]
            z=self.geometry.get_intermediate_layers(z,n=[11],return_class_token=False,norm=True)[0]
            z=z.transpose(1,2).reshape(len(z),384,16,16)
            result['geometry_features']=F.adaptive_avg_pool2d(z,(self.grid,self.grid)).flatten(2).transpose(1,2).float()
        return result
