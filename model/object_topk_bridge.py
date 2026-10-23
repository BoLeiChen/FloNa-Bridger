"""Object modality adapter; identical learned Top-K route/rejection and bridge policy."""
import torch
from torch import nn
from model.topk_route_bridge import TopKRouteBridge
CHECKPOINT_FORMAT='floverse-objectgoal-topk-route-bridge-v1'

class ObjectTopKRouteBridge(TopKRouteBridge):
    def __init__(self,config,load_pretrained=True):
        if [float(config[k]) for k in ('point_goal_train_prob','img_goal_train_prob','object_goal_train_prob')]!=[0.,0.,1.]:raise ValueError('ObjectGoal-only')
        super().__init__(dict(config,point_goal_train_prob=0.,img_goal_train_prob=1.,object_goal_train_prob=0.),load_pretrained)
        self.config=config
        # The only target-modality adapter: frozen CLIP category -> shared route dimension.
        self.scorer.category_adapter=nn.Sequential(nn.Linear(512,384),nn.LayerNorm(384))
    def visual_features(self,batch,include_alternate=False):
        if include_alternate:raise ValueError('No image-goal alternate pairs for object policy')
        observed,_,rgb=super().visual_features(batch,include_goal=False)
        category=self.scorer.category_adapter(batch['goal_text'])
        return observed,category[:,None,None,:],rgb
    def score_paths(self,batch):
        if not (batch['task_type']==2).all():raise ValueError('ObjectGoal-only')
        # Reuse exactly the ImageGoal scorer, rejection mixture and selection contract.
        return super().score_paths(dict(batch,task_type=torch.ones_like(batch['task_type'])))
