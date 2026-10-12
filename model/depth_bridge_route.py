"""Server V3 route-energy method adapted to VGCNav's mean-path VE bridge.

Perception additions interpolate only adjacent observed floor surfaces and
optionally align historical depth evidence. Neither grants free blind space.
"""
from collections import deque
import math
import numpy as np
import torch
from .depth_bridge_guidance import DepthMap, BridgeDepthGuidance


def calibrate_floor_scale(depth,intrinsic,camera_height):
    """Scale monocular Z using a broadly supported horizontal floor plane.

    q=(v-cy)*Z/fy is constant on a horizontal plane for a level camera.
    Require support across lower-image rows AND columns. This assumes that
    broadly visible lower plane is floor, not tabletop. No simulator depth,
    navigation outcomes, or ground-truth trajectory enters calibration.
    """
    h,w=depth.shape;fy,cy=float(intrinsic[1,1]),float(intrinsic[1,2])
    if not math.isfinite(camera_height) or camera_height<=0:
        raise ValueError('Positive calibrated camera height required')
    q=(np.arange(h)[:,None]-cy)*depth/fy
    valid=np.isfinite(depth)&(depth>.1)&(depth<19.5)&np.isfinite(q)&(q>0)
    roi=np.zeros_like(valid);roi[int(.6*h):int(.98*h),int(.1*w):int(.9*w)]=True
    valid &= roi;values=q[valid]
    info=dict(accepted=False,scale=1.,reason='insufficient_floor_support',support_fraction=0.)
    if values.size<128:return depth,info
    edges=np.arange(math.log(camera_height/2),math.log(camera_height/.25)+.04,.04)
    counts,_=np.histogram(np.log(values),edges)
    best=None
    for index in np.argsort(counts)[-8:]:
        center=math.exp((edges[index]+edges[index+1])/2)
        nearby=values[np.abs(values-center)<.08*center]
        if len(nearby)<128:continue
        level=float(np.median(nearby))
        inliers=valid&(np.abs(q-level)<=.04*level)
        support=float(inliers.sum()/roi.sum())
        if best is None or support>best[0]:best=(support,level,inliers)
    if best is None:return depth,info
    support,level,inliers=best;info['support_fraction']=support
    band=inliers[int(.6*h):int(.98*h),int(.1*w):int(.9*w)]
    covered=all(part.mean()>=.35 for part in np.array_split(band,3,axis=0)) and all(
        part.mean()>=.35 for part in np.array_split(band,4,axis=1))
    scale=camera_height/level
    mad=float(np.median(np.abs(q[inliers]-level))/level) if inliers.any() else float('inf')
    info.update(estimated_camera_height_m=level,relative_mad=mad)
    if support<.55 or not covered or mad>.025 or not .25<=scale<=2.:
        return depth,info
    info.update(accepted=True,scale=scale,reason='supported_horizontal_floor')
    return depth*scale,info


class DepthObservationBuffer:
    """Keep actual native RGB observations/poses from executed motion.

    Subsample very close views before metric-depth inference; never synthesize
    camera poses or step/render the simulator to obtain memory observations.
    """
    def __init__(self, capacity=16, spacing=.1):
        if capacity<1 or not math.isfinite(spacing) or spacing<=0:
            raise ValueError('Invalid observation buffer')
        self.frames=deque(maxlen=capacity);self.last_pose=None;self.spacing=spacing

    def observe(self,rgb,position,heading):
        direction=np.asarray(heading)[:2]-np.asarray(position)[:2]
        if np.linalg.norm(direction)<1e-8:return
        pose=np.array([position[0],position[1],np.arctan2(direction[1],direction[0])])
        if not np.isfinite(pose).all():raise ValueError('Invalid observed camera pose')
        if self.last_pose is not None:
            angle=math.atan2(math.sin(pose[2]-self.last_pose[2]),math.cos(pose[2]-self.last_pose[2]))
            if np.linalg.norm(pose[:2]-self.last_pose[:2])<self.spacing and abs(angle)<.1:return
        self.frames.append((np.asarray(rgb).copy(),pose.copy()));self.last_pose=pose.copy()

    def drain(self):
        frames=list(self.frames);self.frames.clear();return frames


def rasterize_floor_surface(forward, left, floor, shape, resolution, xmin, ymax, max_edge):
    """Rasterize small triangles between three adjacent observed floor pixels.

    Reject depth discontinuities by metric edge length. No convex hull, hole
    closing, ray carving, or interpolation across invalid/occluded pixels.
    """
    import cv2
    if not math.isfinite(max_edge) or not 0 < max_edge <= .1:
        raise ValueError('Floor interpolation edge must be in (0, .1] metres')
    xy = np.stack((forward, left), -1)
    result = np.zeros(shape, np.uint8)
    # Two independent triangles, each requiring all three observed vertices.
    for offsets in [((0,0),(1,0),(0,1)), ((1,1),(0,1),(1,0))]:
        h,w = floor.shape
        masks=[floor[a:a+h-1,b:b+w-1] for a,b in offsets]
        vertices=[xy[a:a+h-1,b:b+w-1] for a,b in offsets]
        valid=masks[0]&masks[1]&masks[2]
        for i,j in [(0,1),(1,2),(2,0)]:
            valid &= np.linalg.norm(vertices[i]-vertices[j],axis=-1)<=max_edge
        triangles=np.stack([v[valid] for v in vertices],1)
        pixels=np.stack(((ymax-triangles[...,1])/resolution,
                         (triangles[...,0]-xmin)/resolution),-1)
        for triangle in np.rint(pixels).astype(np.int32):
            cv2.fillConvexPoly(result, triangle, 1)
    return result.astype(bool)


class DepthMapMemory:
    """Same occupied-wins pose fusion as server ObservationMapMemory.

    Sources are retained without repeated resampling, and reset per episode.
    A failed current depth estimate contributes no evidence; retained supported
    observations may still guide inside their observed area in static scenes.
    """
    def __init__(self, capacity=16):
        if int(capacity)!=capacity or capacity<1:
            raise ValueError('Positive integer memory capacity required')
        self.frames=deque(maxlen=int(capacity))

    def add(self, local, world_pose):
        pose=np.asarray(world_pose,dtype=float)
        if pose.shape!=(3,) or not np.isfinite(pose).all():
            raise ValueError('Expected finite (x,y,yaw) pose')
        if self.frames:
            old=self.frames[-1][1]
            angle=math.atan2(math.sin(pose[2]-old[2]),math.cos(pose[2]-old[2]))
            if np.linalg.norm(pose[:2]-old[:2])<.025 and abs(angle)<.02:
                self.frames.pop()
        self.frames.append((local,pose.copy()))

    @torch.no_grad()
    def current(self):
        if not self.frames:
            raise ValueError('No observed maps')
        target,origin=self.frames[-1]
        h,w=target.sdf.shape[-2:];device=target.sdf.device
        row,col=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
        xy=torch.stack((target.x_min+row*target.resolution,target.y_max-col*target.resolution),-1).float().reshape(1,-1,2)
        known=target.known.clone();occupied=target.occupied.clone()
        for source,pose in list(self.frames)[:-1]:
            if (source.resolution,source.x_min,source.y_max)!=(target.resolution,target.x_min,target.y_max):
                raise ValueError('Memory map geometry mismatch')
            shift=pose[:2]-origin[:2];c,s=math.cos(origin[2]),math.sin(origin[2])
            relative=xy.new_tensor([c*shift[0]+s*shift[1],-s*shift[0]+c*shift[1]])
            delta=xy-relative;angle=pose[2]-origin[2];c,s=math.cos(angle),math.sin(angle)
            query=torch.stack((c*delta[...,0]+s*delta[...,1],-s*delta[...,0]+c*delta[...,1]),-1)
            seen,outside=source.query(source.known.float(),query)
            hits,_=source.query(source.occupied.float(),query)
            known |= ((seen>.999)&(outside==0)).reshape_as(known)
            occupied |= ((hits>.001)&(outside==0)).reshape_as(occupied)
        fused=DepthMap(occupied[0].cpu().numpy(),(known|occupied)[0].cpu().numpy(),
                       target.resolution,target.x_min,target.y_max,device)
        fused.valid_fraction=max(local.valid_fraction for local,_ in self.frames)
        fused.raw_valid_fraction=getattr(target,'raw_valid_fraction',target.valid_fraction)
        fused.depth_calibration=getattr(target,'depth_calibration',{})
        return fused


class BridgeRouteGuidance(BridgeDepthGuidance):
    """Same local route energy and correction-corridor gate as server V3.

    Input prior is the existing unnormalized A* condition in metres, with its
    valid mask. The executed endpoint is path[:,execute_horizon], because the
    decoded path prepends origin but VGCNav executes future points only.
    """
    def __init__(self,local_map,goal_local,config,prior,valid,execute_horizon,metric_scale=.045):
        super().__init__(local_map,goal_local,config,metric_scale)
        self.prior=torch.as_tensor(prior,device=self.goal.device,dtype=torch.float32).reshape(1,-1,2).detach()
        self.valid=torch.as_tensor(valid,device=self.goal.device,dtype=torch.float32).reshape(1,-1).detach()
        self.execute_horizon=int(execute_horizon)
        if (self.valid.shape!=self.prior.shape[:2] or self.prior.shape[1]<2
            or not torch.isfinite(self.prior).all() or not torch.isfinite(self.valid).all()
            or not 1<=self.execute_horizon<=self.prior.shape[1]):
            raise ValueError('Invalid metric route or execution horizon')
        self.lookahead=float(config.get('route_lookahead_m',.25))
        self.max_cross_track=float(config.get('route_max_cross_track_m',.5))
        if not all(math.isfinite(v) and v>0 for v in [self.lookahead,self.max_cross_track]):
            raise ValueError('Invalid route geometry')
        self.route_target=None;self.gate_reason='unbound'

    def _goal_visible(self):
        # The final goal's origin corridor is deliberately not used in V3.
        return False

    @torch.no_grad()
    def bind_reference(self,path):
        if self.execute_horizon>=path.shape[1]:
            raise ValueError('Execution endpoint outside decoded path')
        endpoint=path[:,self.execute_horizon]
        start=self.prior[:,:-1];segment=self.prior[:,1:]-start
        length=segment.norm(dim=-1)
        valid=self.valid>.5
        # Only the initial contiguous valid prior prefix may guide. No jump
        # through an invalid segment, including A* fallback trajectories.
        prefix=valid.to(torch.int64).cumprod(-1).bool()
        usable=prefix[:,:-1]&prefix[:,1:]&(length>1e-6)
        self.visible=False;self.route_target=endpoint.detach();self.gate_reason='invalid_prior'
        if not bool(usable.any()):return
        fraction=(((endpoint[:,None]-start)*segment).sum(-1)/length.square().clamp_min(1e-12)).clamp(0,1)
        nearest=start+fraction[...,None]*segment
        distance=(nearest-endpoint[:,None]).norm(dim=-1).masked_fill(~usable,float('inf'))
        index=int(distance.argmin(-1).item())
        arc=torch.cat((torch.zeros_like(length[:,:1]),length.cumsum(-1)),-1)
        current=arc[0,index]+fraction[0,index]*length[0,index]
        last=int(torch.where(usable[0])[0][-1])
        target_s=torch.minimum(current+self.lookahead,arc[0,last+1])
        candidates=torch.where(usable[0]&(arc[0,1:]>=target_s))[0]
        ti=int(candidates[0]) if len(candidates) else last
        t=((target_s-arc[0,ti])/length[0,ti].clamp_min(1e-6)).clamp(0,1)
        target=start[:,ti]+t*segment[:,ti];self.route_target=target.detach()
        if distance[0,index]>self.max_cross_track:self.gate_reason='cross_track';return
        if target_s<=current+1e-4:self.gate_reason='no_progress';return
        n=max(2,int(2*(self.max_cross_track+self.lookahead)/self.map.resolution)+2)
        u=torch.linspace(0,1,n,device=path.device,dtype=path.dtype)
        points=endpoint[:,None]*(1-u[None,:,None])+target[:,None]*u[None,:,None]
        sdf,outside=self.map.query(self.map.sdf,points)
        known,_=self.map.query(self.map.clearance,points)
        if outside.max()>0:self.gate_reason='outside';return
        if sdf.min()<self.radius+self.margin:self.gate_reason='obstacle';return
        if known.min()<self.radius:self.gate_reason='unknown';return
        self.visible=True;self.gate_reason='active'

    def terms(self,path,reference,ref_geometry):
        terms=super().terms(path,reference,ref_geometry)
        d=(path[:,self.execute_horizon]-self.route_target).square().sum(-1)
        terms['goal']=(torch.sqrt(1+d)-1)*float(self.visible)
        return terms

    def __call__(self,denoised,sigma,stage):
        count=len(self.records)
        result=super().__call__(denoised,sigma,stage)
        if len(self.records)>count:
            self.records[-1].update(route_active=self.visible,gate_reason=self.gate_reason,
                route_target_m=self.route_target.cpu().tolist(),execute_horizon=self.execute_horizon)
        return result
