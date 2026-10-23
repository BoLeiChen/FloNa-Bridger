"""Sequential predicted-goal search. No reference trajectory or GT goal input."""
import math
import numpy as np


class CandidateNavigator:
    def __init__(self, goals, max_steps=500, collision_limit=15, reach_radius=.5,
                 stall_steps=20, min_progress=.2, unreachable_steps=3, dedup_radius=.5):
        goals = np.asarray(goals, dtype=np.float64)
        if goals.ndim != 2 or goals.shape[1] != 2 or not len(goals) or not np.isfinite(goals).all():
            raise ValueError('Expected nonempty finite predicted world XY goals')
        if min(max_steps, collision_limit, reach_radius, stall_steps, min_progress, unreachable_steps) <= 0 or dedup_radius < 0:
            raise ValueError('Candidate budgets and thresholds must be positive')
        self.goals, self.ranks, self.duplicates = [], [], []
        for rank, goal in enumerate(goals):
            if any(np.linalg.norm(goal - other) < dedup_radius for other in self.goals):
                self.duplicates.append(rank+1)
            else:
                self.goals.append(goal.copy()); self.ranks.append(rank+1)
        self.max_steps, self.collision_limit = max_steps, collision_limit
        self.reach_radius, self.stall_steps = reach_radius, stall_steps
        self.min_progress, self.unreachable_steps = min_progress, unreachable_steps
        self.index = 0
        self.events = []
        self._enter(0, 0)

    @property
    def exhausted(self):
        return self.index >= len(self.goals)

    @property
    def goal(self):
        if self.exhausted:
            raise RuntimeError('No remaining candidate')
        return self.goals[self.index].copy()

    def _enter(self, steps, collisions):
        remaining = len(self.goals)-self.index
        self.entry_step, self.entry_collisions = steps, collisions
        self.step_quota = math.ceil((self.max_steps-steps)/remaining)
        self.collision_quota = math.ceil((self.collision_limit-collisions)/remaining)
        self.best_distance = float('inf')
        self.last_progress = steps
        self.invalid_count = 0

    def observe(self, position, path_distance, steps, collisions):
        """Return switch reason, or None; distance is from predicted map goal tree.

        Called once before each replan (and once for each newly activated goal).
        Counters are absolute episode counters, never reset by this controller.
        """
        if self.exhausted:
            return None
        euclidean = float(np.linalg.norm(np.asarray(position)-self.goal))
        if np.isfinite(path_distance):
            self.invalid_count = 0
            if path_distance <= self.best_distance-self.min_progress:
                self.best_distance = float(path_distance)
                self.last_progress = steps
        else:
            self.invalid_count += 1
        reason = None
        if euclidean < self.reach_radius:
            reason = 'reached_prediction'
        elif steps-self.entry_step >= self.step_quota:
            reason = 'planning_quota'
        elif collisions-self.entry_collisions >= self.collision_quota:
            reason = 'collision_quota'
        elif self.invalid_count >= self.unreachable_steps:
            reason = 'unreachable'
        elif steps-self.last_progress >= self.stall_steps:
            reason = 'stalled'
        if reason is not None:
            self.events.append(dict(rank=self.ranks[self.index], goal=self.goal.tolist(),
                reason=reason, global_replans=int(steps), global_collisions=int(collisions),
                candidate_replans=int(steps-self.entry_step),
                candidate_collisions=int(collisions-self.entry_collisions),
                distance_to_prediction_m=euclidean,
                path_distance_m=float(path_distance) if np.isfinite(path_distance) else None))
            self.index += 1
            if not self.exhausted:
                self._enter(steps, collisions)
        return reason

    def summary(self):
        return dict(candidate_ranks=self.ranks, dropped_duplicate_ranks=self.duplicates,
                    exhausted=self.exhausted, active_rank=None if self.exhausted else self.ranks[self.index],
                    switches=self.events, reach_radius_m=self.reach_radius,
                    stall_steps=self.stall_steps, min_progress_m=self.min_progress,
                    unreachable_steps=self.unreachable_steps)
