"""RGB metric-depth guidance for the VE A*-conditioned diffusion bridge.

Guidance is a bounded proximal edit of the clean denoised prediction D, so
bridge scores use (D_guided - x)/sigma**2. The xT endpoint, bridge drift,
churn noise and Heun coefficients stay intact. This is approximate energy
steering, not an exact conditional score or a hard collision guarantee.
"""
from pathlib import Path
import importlib
import sys
import types
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F


class MetricDepthEstimator:
    """Official Hypersim ViT-S metric model; separate namespace from policy DINO."""
    def __init__(self, config, device):
        self.device = torch.device(device)
        source = Path(config['source_dir']).resolve() / 'metric_depth/depth_anything_v2'
        if not (source/'dpt.py').is_file():
            raise FileNotFoundError(str(source/'dpt.py'))
        name = '_vgcnav_metric_depth'
        if name in sys.modules:
            if list(sys.modules[name].__path__) != [str(source)]:
                raise ValueError('another metric-depth source is already loaded')
        else:
            package = types.ModuleType(name)
            package.__path__ = [str(source)]
            sys.modules[name] = package
        dpt = importlib.import_module(name+'.dpt')
        # Model initialization must not shift the bridge's random noise stream.
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            self.net = dpt.DepthAnythingV2(encoder='vits', features=64,
                out_channels=[48,96,192,384], max_depth=20.0)
            state = torch.load(config['checkpoint'],map_location='cpu',weights_only=True)
            self.net.load_state_dict(state,strict=True)
            self.net.to(self.device).eval().requires_grad_(False)
        size = int(config.get('input_size',518))
        self.transforms = [dpt.Resize(width=size,height=size,resize_target=False,
            keep_aspect_ratio=True,ensure_multiple_of=14,resize_method='lower_bound',
            image_interpolation_method=2),
            dpt.NormalizeImage(mean=[.485,.456,.406],std=[.229,.224,.225]),dpt.PrepareForNet()]

    @torch.no_grad()
    def predict(self, rgb):
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('expected native-resolution uint8 RGB')
        # Official infer_image expects BGR and converts to RGB. This interface
        # already receives RGB, so no second channel swap is performed.
        data = {'image': rgb.astype(np.float32)/255.}
        for transform in self.transforms:
            data = transform(data)
        x = torch.from_numpy(data['image']).unsqueeze(0).to(self.device)
        z = self.net(x)
        z = F.interpolate(z[:,None],rgb.shape[:2],mode='bilinear',align_corners=True)[0,0]
        return z.cpu().numpy()


class DepthMap:
    def __init__(self, occupied, known, resolution, x_min, y_max, device):
        self.resolution, self.x_min, self.y_max = resolution, x_min, y_max
        # No synthetic obstacle at a local-map boundary. Outside space is
        # separately accounted for as unknown exposure.
        if occupied.any():
            sdf = (distance_transform_edt(~occupied)-distance_transform_edt(occupied))*resolution
        else:
            sdf = np.full(occupied.shape,100.,dtype=np.float32)
        free = distance_transform_edt(np.pad(known & ~occupied,1))[1:-1,1:-1]*resolution
        self.sdf = torch.as_tensor(sdf,dtype=torch.float32,device=device)[None]
        self.clearance = torch.as_tensor(free,dtype=torch.float32,device=device)[None]
        self.occupied_cells = int(occupied.sum())
        self.known_fraction = float(known.mean())
        self.occupied = torch.as_tensor(occupied, device=device)[None]
        self.known = torch.as_tensor(known | occupied, device=device)[None]

    def query(self, field, points):
        h,w = field.shape[-2:]
        row = (points[...,0]-self.x_min)/self.resolution
        col = (self.y_max-points[...,1])/self.resolution
        grid = torch.stack((2*col/(w-1)-1,2*row/(h-1)-1),-1)
        values = F.grid_sample(field[:,None],grid[:,:,None],align_corners=True,padding_mode='border')[:,0,:,0]
        outside = (F.relu(-row)+F.relu(row-h+1)+F.relu(-col)+F.relu(col-w+1))*self.resolution
        return values,outside


def depth_to_map(depth, intrinsic, config, device):
    """Optical Z (not radial range) -> x-forward/y-left/z-up, level camera.

    Only predicted floor returns mark known free cells; no ray carving through
    unseen ground, no traversability map, and no simulator depth enters here.
    """
    depth=np.asarray(depth,dtype=np.float32); k=np.asarray(intrinsic,dtype=np.float64)
    if depth.ndim!=2 or k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:
        raise ValueError('invalid depth shape or camera intrinsics')
    calibration=dict(accepted=True,scale=1.,reason='disabled')
    raw_valid_fraction=float((np.isfinite(depth)&(depth>.1)&(depth<19.5)).mean())
    if config.get('floor_scale_calibration', False):
        from .depth_bridge_route import calibrate_floor_scale
        depth,calibration=calibrate_floor_scale(depth,k,float(config.get('camera_height',.85)))
        if not calibration['accepted']:
            # Unreliable geometry must not contaminate memory or steer policy.
            depth=np.zeros_like(depth)
    res=float(config.get('resolution',.025)); xmin,xmax=-1.,4.; ymin,ymax=-2.5,2.5
    if not math.isfinite(res) or res<=0:raise ValueError('invalid map resolution')
    h,w=depth.shape;v,u=np.mgrid[:h,:w]
    valid=np.isfinite(depth)&(depth>.1)&(depth<19.5)
    z=np.where(valid,depth,0.)
    forward=z;left=-(u-k[0,2])*z/k[0,0]
    height=float(config.get('camera_height',.85))-(v-k[1,2])*z/k[1,1]
    floor=valid&(np.abs(height)<=.08)
    obstacle=valid&(height>.12)&(height<float(config.get('obstacle_height',1.2)))
    # Occlusion: a floor return behind a body-height hit is not free evidence.
    azimuth=np.arctan2(left,forward);bins=np.floor((azimuth+np.pi)*1024/(2*np.pi)).astype(int)%1024
    radial=np.hypot(forward,left);nearest=np.full(1024,np.inf)
    np.minimum.at(nearest,bins[obstacle],radial[obstacle])
    nearest=np.minimum(nearest,np.minimum(np.roll(nearest,1),np.roll(nearest,-1)))
    floor &= radial<nearest[bins]-res
    shape=(round((xmax-xmin)/res)+1,round((ymax-ymin)/res)+1)
    occ=np.zeros(shape,bool);known=np.zeros(shape,bool)
    row=np.rint((forward-xmin)/res).astype(int);col=np.rint((ymax-left)/res).astype(int)
    inside=(row>=0)&(row<shape[0])&(col>=0)&(col<shape[1])
    for mask, dest in [(obstacle,occ),(floor,known)]:
        use=mask&inside;dest[row[use],col[use]]=True
    if config.get('floor_surface', False):
        from .depth_bridge_route import rasterize_floor_surface
        known |= rasterize_floor_surface(forward, left, floor & inside, shape,
            res, xmin, ymax, float(config.get('floor_surface_max_edge_m', .1)))
    known |= occ
    local=DepthMap(occ,known,res,xmin,ymax,device)
    local.valid_fraction=float(valid.mean())
    local.raw_valid_fraction=raw_valid_fraction
    local.depth_calibration=calibration
    return local


def decode_metric(x, metric_scale):
    lo=x.new_tensor([-2.5,-4.]);hi=x.new_tensor([5.,4.])
    points=((x+1)*.5*(hi-lo)+lo).cumsum(1)*metric_scale
    return torch.cat((points.new_zeros(len(x),1,2),points),1)


class BridgeDepthGuidance:
    """Optimize the mean candidate path actually executed by VGCNav.

    Candidate diversity is retained via a shared delta correction, followed by
    the native [-1,1] clipping. Acceptance checks the resulting clipped mean.
    """
    def __init__(self, local_map, goal_local, config, metric_scale=.045):
        self.map,self.cfg,self.metric_scale=local_map,config,float(metric_scale)
        self.scale=float(config.get('scale',1.))
        self.limit=float(config.get('max_update_m',.01))
        self.sigma_start=float(config.get('sigma_start',.5))
        self.radius=float(config.get('robot_radius',.15))+math.sqrt(2)*local_map.resolution
        self.margin=float(config.get('safety_margin',.1))
        vals=[self.metric_scale,self.limit,self.sigma_start,self.radius,self.margin]
        if not all(math.isfinite(v) and v>0 for v in vals) or not math.isfinite(self.scale) or self.scale<0:
            raise ValueError('invalid guidance scales')
        self.weights=dict(collision=1.,unknown=1.,boundary=1.,shape=.1,goal=.1)
        self.weights.update(config.get('weights',{}))
        if any(not math.isfinite(float(v)) or float(v)<0 for v in self.weights.values()):
            raise ValueError('invalid objective weight')
        self.goal=torch.as_tensor(goal_local,device=local_map.sdf.device,dtype=torch.float32).reshape(1,1,2)
        if not torch.isfinite(self.goal).all():raise ValueError('nonfinite point goal')
        self.records=[]
        self.visible=self._goal_visible()

    @torch.no_grad()
    def _goal_visible(self):
        t=torch.linspace(0,1,600,device=self.goal.device)
        line=self.goal*t[None,:,None]
        sdf,outside=self.map.query(self.map.sdf,line)
        free,_=self.map.query(self.map.clearance,line)
        return bool((outside.max()==0)&(sdf.min()>=self.radius+self.margin)&(free.min()>=self.radius))

    def geometry(self,path):
        t=torch.linspace(0,1,9,device=path.device,dtype=path.dtype)
        dense=(path[:,:-1,None]*(1-t[None,None,:,None])+path[:,1:,None]*t[None,None,:,None]).flatten(1,2)
        sdf,outside=self.map.query(self.map.sdf,dense)
        free,_=self.map.query(self.map.clearance,dense)
        z=(sdf-self.radius)/self.margin
        risk=torch.where(z<0,-z+.5,.5*F.relu(1-z).square())
        collision=.5*risk.mean(-1)+.5*risk.amax(-1)
        return collision,F.relu(1-free/self.radius).square(),(outside/self.margin).square()

    def terms(self,path,reference,ref_geometry):
        collision,unknown,boundary=self.geometry(path)
        delta=path[:,1:]-path[:,:-1];ref_delta=reference[:,1:]-reference[:,:-1]
        return dict(collision=collision,unknown=F.relu(unknown-ref_geometry[1]).mean(-1),
            boundary=F.relu(boundary-ref_geometry[2]).mean(-1),
            shape=((delta-ref_delta)/.1).square().sum(-1).mean(-1),
            goal=(torch.sqrt(1+(path[:,-1:]-self.goal).square().sum(-1)).squeeze(-1)-1)*float(self.visible))

    def total(self,terms):
        return sum(float(self.weights[k])*v for k,v in terms.items())

    def bind_reference(self, reference):
        """Optional local-goal selection at each clean prediction."""

    @torch.no_grad()
    def __call__(self, denoised, sigma, stage):
        if self.scale==0 or float(sigma.max())>self.sigma_start or self.map.valid_fraction<.05:
            return denoised
        # Evaluation is a single observation with N candidate paths.
        mean_encoded=denoised.mean(0,keepdim=True).detach()
        reference=decode_metric(mean_encoded,self.metric_scale)
        self.bind_reference(reference)
        ref_geometry=self.geometry(reference)
        with torch.enable_grad():
            mean=mean_encoded.clone().requires_grad_(True)
            path=decode_metric(mean,self.metric_scale)
            before=self.terms(path,reference,ref_geometry)
            energy=self.total(before)
            gradient=torch.autograd.grad(energy.sum(),mean)[0]
        if not torch.isfinite(gradient).all():raise ValueError('nonfinite bridge energy gradient')
        # VE clean-prediction correction: induced score change is -scale*grad,
        # before clipping/trust bounds. NOT a DDPM posterior variance update.
        change=-self.scale*sigma[0].square()*gradient
        shift=(change*change.new_tensor([3.75,4.])*self.metric_scale).cumsum(1).norm(dim=-1).amax()
        change*=min(1.,self.limit/max(float(shift),1e-12))
        original=self.total(before).detach();accepted=0.;after=before
        result=denoised
        for trial in range(6):
            factor=.5**trial
            proposal=(denoised+factor*change).clamp(-1,1)
            candidate_path=decode_metric(proposal,self.metric_scale)
            actual_shift=(candidate_path-decode_metric(denoised,self.metric_scale)).norm(dim=-1).amax()
            terms=self.terms(candidate_path.mean(0,keepdim=True),reference,ref_geometry)
            if (torch.isfinite(proposal).all() and actual_shift<=self.limit+1e-7
                and self.total(terms)<=original+1e-7
                and terms['collision']<=before['collision'].detach()+1e-7):
                result,after,accepted=proposal,terms,factor
                break
        self.records.append(dict(sigma=float(sigma[0]),stage=stage,accepted_scale=accepted,
            gradient_norm=float(gradient.norm()),visible_goal=self.visible,
            max_update_m=float((decode_metric(result,self.metric_scale)-decode_metric(denoised,self.metric_scale)).norm(dim=-1).amax()),
            before={k:float(v.detach()) for k,v in before.items()},after={k:float(v.detach()) for k,v in after.items()}))
        return result
