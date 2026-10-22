# NaviBridger A* bridge mode

`diffusion_mode: navibridger` replaces the Gaussian-start DDPM trajectory
process with a VE denoising diffusion bridge.  It deliberately leaves VGCNav's
dataset tuple, visual/floorplan encoder, pose inputs, action normalization and
trajectory output unchanged.

This release uses the six-epoch PointGoal configuration in
`training/pointgoal_v3.yaml`. Heading realignment is disabled.

For every sample, the existing A* local waypoints are differenced and
normalized with the same action statistics as the target trajectory.  This is
the bridge endpoint `xT`; the demonstration trajectory is `x0`.  Training
converts the dataset's metric A* path to waypoint units before this operation,
matching simulator inference and the target-action scale. Training
draws an intermediate bridge state between them and predicts `x0`. Inference
starts from `xT` and refines it with the existing ConditionalUnet1D conditioned
on the existing visual/floorplan encoding and the A* path as `local_cond` at
every denoising step. A* therefore supplies both the bridge prior and a
persistent path reference for the denoiser.  The global encoder does not ingest
that full path again: it receives only a robust turn summary computed from the
clean, valid A* prefix.

The global visual condition keeps the original final DINO layer as its semantic
navigation backbone.  Shallow DINO patches from the current frame retain their
full 7x7 layout and are attention-pooled into a residual correction.  A learned
gate uses both visual summaries and the clean-path turn risk, but is not forced
to be monotonic in curvature.  Its residual contribution is capped at 0.7 and
zero-initialized, so optimization starts exactly at the deep-feature baseline.

Turn risk uses a mask-aware smoothed path and headings spanning three waypoint
intervals.  It contains no maximum-turn term and is computed before training
perturbations.  Training path uncertainty is spatially correlated, limited to
a nominal 2 cm deformation amplitude on a 40-epoch sinusoidal schedule
(about 4.67 mm by epoch six), plus a 1 cm Gaussian global translation
standard deviation, and copies the last
valid perturbation into terminal padding.  ImageNet normalization is applied by
the dataset transform exactly once; the DINO encoder does not normalize that
tensor a second time.

Sampling follows NaviBridger's VE Heun solver, including its stochastic
`churn_step_ratio: 0.33` transition before each deterministic Heun update.

Simulator evaluation defaults to `collision_accounting_mode: legacy`, which
counts collision-recovery turns in the saved trajectory like the earlier
evaluation code. Set it to `movement_only` to use the newer behavior that
excludes recovery turns from the collision metric.

Start from the A*-enabled DPT configuration and add the fields in
`training/navibridger_astar_bridge.yaml`.  Do not use this mode with
`--no_map_no_astar`, because that intentionally disables the required A* prior.

For simulator evaluation, start from `test_vgc_navibridger.yaml`, set
`checkpoint_path` to the bridge-trained EMA checkpoint, and retain the A* path
(`--no_map_no_astar` is incompatible with this mode).
