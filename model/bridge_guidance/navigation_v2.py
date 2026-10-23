"""Scale-normalized, prior-relative local navigation guidance.

Unknown space is not certified safe. We penalize added exposure relative to
this reverse-step proposal, without pulling the whole path into a blind spot.
This is a local soft objective, not a collision-free or semantic guarantee.
"""
import math
import torch
import torch.nn.functional as F
def grid_safety_margin(local_map):
    return math.sqrt(2) * local_map.resolution + local_map.resolution / 4
from .geometry import interpolate_segments


class NavigationObjectiveV2:
    def __init__(self, local_map, config, point_goal=None):
        self.map, self.cfg = local_map, config
        self.reference = None
        self.goal = None if point_goal is None else point_goal.detach().reshape(-1, 2)
        self.goal_energy = self.goal is not None
        self.radius = float(config.get("robot_radius", 0.25)) + grid_safety_margin(local_map)
        self.margin = float(config.get("safety_margin", 0.1))
        self.goal_scale = float(config.get("goal_scale_m", 1.0))
        self.shape_scale = float(config.get("shape_scale_m", 0.1))
        self.step_limit = float(config.get("max_step_m", 0.5))
        self.subdivisions = int(config.get("subdivisions", 8))
        scales = [self.radius, self.margin, self.goal_scale, self.shape_scale, self.step_limit]
        if not all(math.isfinite(x) and x > 0 for x in scales) or self.subdivisions < 1:
            raise ValueError("positive finite objective scales required")
        weights = config.get("weights", {})
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in weights.values()):
            raise ValueError("objective weights must be finite and nonnegative")
        self.visible_goal = self._visible_goal()

    @torch.no_grad()
    def _visible_goal(self):
        if self.goal is None:
            return None
        # Only local, fully observed straight corridors receive analytic goal
        # attraction. Occluded/out-of-map goals retain the learned prior alone.
        h, w = self.map.sdf.shape[-2:]
        extent = self.map.resolution * math.hypot(h, w)
        n = max(2, math.ceil(extent / (self.map.resolution / 2)))
        t = torch.linspace(0, 1, n, device=self.goal.device, dtype=self.goal.dtype)
        points = self.goal[:, None] * t[None, :, None]
        sdf, outside = self.map.query(self.map.sdf, points)
        known, _ = self.map.query(self.map.free_clearance, points)
        return ((outside.max(-1).values == 0) &
                (sdf.min(-1).values >= self.radius + self.margin) &
                (known.min(-1).values >= self.radius)).to(self.goal)

    def _geometry(self, path):
        dense = interpolate_segments(path, self.subdivisions)
        sdf, outside = self.map.query(self.map.sdf, dense)
        clearance, _ = self.map.query(self.map.free_clearance, dense)
        z = (sdf - self.radius) / self.margin
        penalty = torch.where(z < 0, -z + 0.5, 0.5 * F.relu(1-z).square())
        # Peak risk must not disappear in a long safe trajectory.
        collision = 0.5 * penalty.mean(-1) + 0.5 * penalty.amax(-1)
        exposure = F.relu(1 - clearance / self.radius).square()
        boundary = (outside / self.margin).square()
        return collision, exposure, boundary

    @torch.no_grad()
    def bind_reference(self, path):
        self.reference = path.detach().clone()
        _, self.reference_exposure, self.reference_boundary = self._geometry(self.reference)

    def terms(self, path):
        if self.reference is None or path.shape != self.reference.shape:
            raise ValueError("bind a matching reverse-step reference first")
        collision, exposure, boundary = self._geometry(path)
        unknown = F.relu(exposure - self.reference_exposure).mean(-1)
        boundary = F.relu(boundary - self.reference_boundary).mean(-1)
        delta = path[:, 1:] - path[:, :-1]
        ref_delta = self.reference[:, 1:] - self.reference[:, :-1]
        # Penalize distortion of the learned path, not movement itself.
        shape = ((delta-ref_delta) / self.shape_scale).square().sum(-1).mean(-1)
        excess = F.relu(delta.norm(dim=-1) / self.step_limit - 1)
        ref_excess = F.relu(ref_delta.norm(dim=-1) / self.step_limit - 1)
        motion = F.relu(excess.square() - ref_excess.square()).mean(-1)
        goal = path.new_zeros(path.shape[0])
        if self.goal is not None:
            distance2 = ((path[:, -1]-self.goal.to(path)) / self.goal_scale).square().sum(-1)
            goal = (torch.sqrt(1+distance2)-1) * self.visible_goal.to(path)
        return dict(collision=collision, unknown=unknown, boundary=boundary,
                    shape=shape, motion=motion, goal=goal)

    def __call__(self, path):
        return sum(float(self.cfg.get("weights", {}).get(k, 0)) * v
                   for k, v in self.terms(path).items())

    @torch.no_grad()
    def accept_update(self, before, after):
        a, b = self.terms(before), self.terms(after)
        # Both checks are per trajectory. A task gain cannot pay for increased
        # modeled collision risk. No claim is made about the subsequent noise.
        return ((self(after) <= self(before) + 1e-7) &
                (b["collision"] <= a["collision"] + 1e-7) &
                torch.isfinite(after).flatten(1).all(-1))
