"""Adapter around CF2Loc-HM3D Stage-1 diffusion localization."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .filter import SequentialPoseFilter
from .geometry import cf2loc_to_world_pose


@dataclass(frozen=True)
class ImageGoalLocalizationEstimate:
    """One-shot global target pose inferred from an ImageGoal observation."""

    pose: np.ndarray
    stage1_pose: np.ndarray
    refiner_used: bool
    refiner_score: float = float("nan")
    refiner_candidate_count: int = 0
    disco_used: bool = False
    disco_pose: object = None
    disco_candidate_count: int = 0
    disco_match_logit: float = float("nan")


class CF2LocEngine:
    """Feed Stage-1 diffusion candidates directly to an SE(2) filter."""

    def __init__(self, config, device):
        self.config = config
        self.device = torch.device(device)
        self.filter = SequentialPoseFilter(config, self.device)
        self._raw_map = None
        self._floorplan_tensor = None
        self._map_wh = None
        self._scale = None
        self._offset_xy = None
        self.refiner = None
        self.disco = None
        self._disco_image_tokens = None
        self.refiner_config = None
        self._crop_to_refiner_tensor = None
        self._apply_refiner_delta = None
        self._load_models()

    @staticmethod
    def _require_file(path, description, expected_sha256=None):
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = Path(__file__).resolve().parents[1] / resolved
        resolved = resolved.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"{description} does not exist: {resolved}")
        if expected_sha256:
            digest = hashlib.sha256()
            with resolved.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"{description} SHA-256 mismatch for {resolved}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
        return resolved

    def _load_models(self):
        from third_party.cf2loc_hm3d.DisCo_model import (
            pose_query_diffusion as diffusion_module,
        )

        diffusion_checkpoint = self._require_file(
            self.config["diffusion_checkpoint"],
            "CF2Loc diffusion checkpoint",
            self.config["diffusion_checkpoint_sha256"],
        )
        checkpoint_config = dict(torch.load(diffusion_checkpoint, map_location='cpu')['hyper_parameters'])
        checkpoint_config['dptv2_ckpt_path'] = self.config.get('dinov2_checkpoint', 'checkpoints/depth_anything_v2_vits.pth')
        self.diffusion = diffusion_module.PoseQueryDiffusionLocalizer.load_from_checkpoint(
            str(diffusion_checkpoint), map_location="cpu", config=checkpoint_config
        ).to(
            self.device
        )
        self.diffusion.eval()
        if self.config.get("cf2loc_disco_enabled", False):
            from .disco import DisCoRanker
            disco_checkpoint = self._require_file(
                self.config["cf2loc_disco_checkpoint"], "DisCo checkpoint",
                self.config["cf2loc_disco_checkpoint_sha256"],
            )
            self.disco = DisCoRanker(
                disco_checkpoint, self.config["diffusion_checkpoint_sha256"],
                self.device, self.config["map_resolution_m"],
            )
            if (self.config["num_particles"] != self.disco.config["particles"]
                    or self.config["sample_steps"] != self.disco.config["sample_steps"]):
                raise ValueError("DisCo candidate sampling differs from training")
            def capture_tokens(module, inputs, output):
                self._disco_image_tokens = output.detach()
            self._disco_hook = self.diffusion.condition_encoder.image_mixer.register_forward_hook(capture_tokens)
        if not self.config["cf2loc_refiner_enabled"]:
            return

        from third_party.cf2loc_hm3d.DisCo_model.pose_local_refiner import (
            PoseLocalRefinerLightning,
            apply_local_delta_to_pose,
            crop_to_refiner_tensor,
            load_refiner_config,
        )

        refiner_config_path = self._require_file(
            self.config["cf2loc_refiner_config"], "CF2Loc refiner config"
        )
        refiner_config = load_refiner_config(str(refiner_config_path))
        refiner_config["baseline_checkpoint_path"] = str(diffusion_checkpoint)
        refiner_config.setdefault("datasets", {})["map_res"] = float(
            self.config["map_resolution_m"]
        )
        refiner_checkpoint = self._require_file(
            self.config["cf2loc_refiner_checkpoint"],
            "CF2Loc refiner checkpoint",
            self.config["cf2loc_refiner_checkpoint_sha256"] or None,
        )
        self.refiner = PoseLocalRefinerLightning.load_from_checkpoint(
            str(refiner_checkpoint), config=refiner_config, map_location="cpu"
        ).to(self.device)
        self.refiner.eval()
        self.refiner_config = refiner_config
        self._crop_to_refiner_tensor = crop_to_refiner_tensor
        self._apply_refiner_delta = apply_local_delta_to_pose

    @staticmethod
    def _pil_to_tensor(image):
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        return torch.from_numpy(array.transpose(2, 0, 1).copy())

    def reset_scene(self, floorplan_path, scale, offset_xy):
        """Build the exact isotropic map geometry used by HM3D preprocessing."""
        floorplan_path = self._require_file(floorplan_path, "evaluation floorplan")
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("floorplan scale must be finite and positive")
        offset_xy = np.asarray(offset_xy, dtype=np.float64)
        if offset_xy.shape != (2,) or not np.isfinite(offset_xy).all():
            raise ValueError("floorplan offset must contain two finite values")

        map_resolution = self.config["map_resolution_m"]
        factor = 1.0 / (map_resolution * scale)
        with Image.open(floorplan_path) as image:
            gray = image.convert("L")
            size = tuple(int(math.ceil(value * factor)) for value in gray.size)
            metric_map = gray.transform(
                size,
                Image.Transform.AFFINE,
                (1.0 / factor, 0.0, 0.0, 0.0, 1.0 / factor, 0.0),
                resample=Image.Resampling.BILINEAR,
                fillcolor=255,
            )

        self._raw_map = np.asarray(metric_map, dtype=np.uint8)
        network_map = metric_map.convert("RGB").resize(
            self.config["floorplan_input_size"], Image.Resampling.BICUBIC
        )
        self._floorplan_tensor = (
            self._pil_to_tensor(network_map).unsqueeze(0).to(self.device)
        )
        self._map_wh = torch.tensor(
            [[metric_map.width, metric_map.height]],
            dtype=torch.float32,
            device=self.device,
        )
        self._scale = scale
        self._offset_xy = offset_xy
        world_origin = self._map_pose_to_world(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        )[:2]
        map_size_m = (
            np.asarray([metric_map.width, metric_map.height], dtype=np.float64)
            * map_resolution
        )
        free_mask = self._raw_map >= int(self.config["belief_free_space_threshold"])
        self.filter.reset(map_size_m, world_origin, free_mask=free_mask)

    def begin_tracking(self):
        self.filter.begin_tracking()

    def preprocess_observation(self, observation):
        if isinstance(observation, Image.Image):
            image = observation.convert("RGB")
        else:
            array = np.asarray(observation)
            if array.dtype != np.uint8:
                if array.size and float(np.nanmax(array)) <= 1.0:
                    array = array * 255.0
                array = np.clip(array, 0, 255).astype(np.uint8)
            image = Image.fromarray(array[:, :, :3]).convert("RGB")
        image = image.resize(self.config["observation_size"], Image.Resampling.BICUBIC)
        tensor = self._pil_to_tensor(image)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return ((tensor - mean) / std).unsqueeze(0).to(self.device)

    def _map_pose_to_world(self, pose):
        return cf2loc_to_world_pose(
            pose, self._scale, self._offset_xy, self.config["map_resolution_m"],
        )

    def _sample_pose_candidates(self, observation):
        if self._floorplan_tensor is None:
            raise RuntimeError("reset_scene must be called before localization")
        obs_img = self.preprocess_observation(observation)
        map_tokens, map_coordinates, image_global = self.diffusion.condition_encoder(
            obs_img, self._floorplan_tensor
        )
        pose_samples = self.diffusion.sample_from_context(
            map_tokens,
            map_coordinates,
            image_global,
            self._map_wh,
            num_particles=self.config["num_particles"],
            num_steps=self.config["sample_steps"],
        )
        stage1_pose, density = self.diffusion.select_pose_mode(pose_samples)
        return obs_img, pose_samples[0], stage1_pose[0], density[0]

    @staticmethod
    def _select_diverse_candidates(
        samples, density, top_k, map_resolution_m, min_position_m, min_yaw_deg,
    ):
        if samples.ndim != 2 or samples.shape[-1] != 3:
            raise ValueError("samples must have shape [N, 3]")
        ordered = torch.argsort(density, descending=True)
        selected = []
        min_yaw = math.radians(float(min_yaw_deg))
        for index in ordered.detach().cpu().tolist():
            candidate = samples[index]
            keep = True
            for chosen_index in selected:
                chosen = samples[chosen_index]
                position_distance = torch.linalg.norm(candidate[:2] - chosen[:2])
                yaw_distance = torch.abs(
                    torch.remainder(candidate[2] - chosen[2] + math.pi, 2 * math.pi)
                    - math.pi
                )
                position_is_close = (
                    float(position_distance) * map_resolution_m < min_position_m
                )
                yaw_is_close = float(yaw_distance) < min_yaw
                if position_is_close and yaw_is_close:
                    keep = False
                    break
            if keep:
                selected.append(index)
                if len(selected) == int(top_k):
                    break
        return torch.as_tensor(selected, dtype=torch.long, device=samples.device)

    def _refine_candidates(self, obs_img, samples, density, selected_only=False):
        dataset_config = self.refiner_config["datasets"]
        if selected_only:
            indices = torch.zeros(1, dtype=torch.long, device=samples.device)
        elif self.config["cf2loc_refiner_selection"] == "cf2loc_topk":
            # Match CF2Loc-HM3D eval/eval_pose_local_refiner.py exactly:
            # KDE-density top-k, no diversity/NMS filtering, then refiner score.
            indices = torch.topk(
                density, k=self.config["cf2loc_refiner_top_k"]
            ).indices
        else:
            indices = self._select_diverse_candidates(
                samples,
                density,
                self.config["cf2loc_refiner_top_k"],
                self.config["map_resolution_m"],
                self.config["cf2loc_refiner_nms_position_m"],
                self.config["cf2loc_refiner_nms_yaw_deg"],
            )
        candidate_poses = samples[indices]
        representation = dataset_config.get(
            "refiner_floorplan_representation",
            dataset_config.get("floorplan_representation", "gray"),
        )
        local_maps = [
            self._crop_to_refiner_tensor(
                self._raw_map,
                pose.detach().cpu().numpy(),
                crop_size_meters=float(
                    self.refiner_config.get("refiner_crop_size_meters", 5.0)
                ),
                map_res=float(self.config["map_resolution_m"]),
                output_size=int(
                    self.refiner_config.get("refiner_crop_output_size", 256)
                ),
                representation=representation,
                oriented=bool(dataset_config.get("refiner_oriented_crop", True)),
            )
            for pose in candidate_poses
        ]
        local_maps = torch.stack(local_maps).to(self.device)
        wh_batch = self._map_wh.expand(len(candidate_poses), -1)
        with torch.inference_mode():
            image_tokens = self.refiner.refiner.encode_image(obs_img)
            outputs = self.refiner(
                obs_img=obs_img.expand(len(candidate_poses), -1, -1, -1),
                local_map=local_maps,
                candidate_pose=candidate_poses,
                wh=wh_batch,
                image_tokens=image_tokens.expand(len(candidate_poses), -1, -1),
            )
            refined = self._apply_refiner_delta(
                candidate_poses,
                outputs["delta_xy_m"],
                outputs["delta_theta"],
                map_res=float(self.config["map_resolution_m"]),
                wh=wh_batch,
            )
        best = int(torch.argmax(outputs["score_logit"]).item())
        return (
            refined[best],
            float(torch.sigmoid(outputs["score_logit"][best]).item()),
            int(len(candidate_poses)),
        )

    @torch.inference_mode()
    def localize_image(self, observation, seed=None):
        """Optional per-goal RNG isolation makes localization independent of navigation."""
        if seed is None:
            return self._localize_image(observation)
        devices = []
        if self.device.type == "cuda":
            devices = [
                self.device.index if self.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(int(seed))
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(int(seed))
            return self._localize_image(observation)

    @torch.inference_mode()
    def _localize_image(self, observation):
        """Localize a static ImageGoal without temporal filtering."""
        obs_img, samples, stage1_pose, density = self._sample_pose_candidates(
            observation
        )
        final_pose = stage1_pose
        disco_pose = None
        disco_info = {}
        if self.disco is not None:
            if self._disco_image_tokens is None:
                raise RuntimeError("Diffusion image features were not captured")
            final_pose, disco_info = self.disco.select(
                self._disco_image_tokens, samples, density, self._raw_map,
            )
            disco_pose = self._map_pose_to_world(final_pose.detach().cpu().numpy()).astype(np.float32)
        refiner_score = float("nan")
        refiner_candidate_count = 0
        if self.refiner is not None:
            # A locally trained score cannot reliably compare distant rooms.
            # Keep the globally selected mode unless explicitly reproducing reranking.
            candidates, candidate_density = samples, density
            if self.disco is not None or self.config["cf2loc_refiner_selection"] == "stage1_mode":
                candidates = final_pose[None]
                candidate_density = torch.ones(1, device=samples.device)
            (
                final_pose,
                refiner_score,
                refiner_candidate_count,
            ) = self._refine_candidates(
                obs_img, candidates, candidate_density,
                **({"selected_only": True} if self.disco is not None else {}),
            )
        stage1_world = self._map_pose_to_world(stage1_pose.detach().cpu().numpy())
        final_world = self._map_pose_to_world(final_pose.detach().cpu().numpy())
        return ImageGoalLocalizationEstimate(
            pose=final_world.astype(np.float32),
            stage1_pose=stage1_world.astype(np.float32),
            refiner_used=self.refiner is not None,
            refiner_score=refiner_score,
            refiner_candidate_count=refiner_candidate_count,
            disco_used=self.disco is not None,
            disco_pose=disco_pose,
            disco_candidate_count=disco_info.get("candidate_count", 0),
            disco_match_logit=disco_info.get("match_logit", float("nan")),
        )

    @torch.inference_mode()
    def localize_topk(self, observation, top_k=3, seed=None):
        """Rank the unchanged DisCo pool, refine each pose, and keep global rank.

        No GT pose is accepted. Do not rank distant rooms by local refiner score.
        """
        if self.disco is None or not 1 <= top_k <= self.disco.config['modes']:
            raise ValueError('top_k requires DisCo and must fit its candidate pool')
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.random.default_generator.manual_seed(int(seed))
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(int(seed))
            obs, samples, stage1, density = self._sample_pose_candidates(observation)
            candidates, infos = self.disco.rank(self._disco_image_tokens, samples, density, self._raw_map)
            estimates = []
            stage1_world = self._map_pose_to_world(stage1.cpu().numpy()).astype(np.float32)
            for candidate, info in zip(candidates[:top_k], infos[:top_k]):
                refined, score, count = candidate, float('nan'), 0
                if self.refiner is not None:
                    refined, score, count = self._refine_candidates(
                        obs, candidate[None], torch.ones(1, device=self.device), selected_only=True)
                estimates.append(ImageGoalLocalizationEstimate(
                    pose=self._map_pose_to_world(refined.cpu().numpy()).astype(np.float32),
                    stage1_pose=stage1_world.copy(), refiner_used=self.refiner is not None,
                    refiner_score=score, refiner_candidate_count=count, disco_used=True,
                    disco_pose=self._map_pose_to_world(candidate.cpu().numpy()).astype(np.float32),
                    disco_candidate_count=info['candidate_count'], disco_match_logit=info['match_logit']))
            return estimates

    @torch.inference_mode()
    def update(self, observation, odometry_delta=None):
        _, pose_samples, stage1_pose, _ = self._sample_pose_candidates(observation)
        world_candidates = np.stack(
            [
                self._map_pose_to_world(pose)
                for pose in pose_samples.detach().cpu().numpy()
            ]
        )
        stage1_world = self._map_pose_to_world(stage1_pose.detach().cpu().numpy())
        return self.filter.update(
            world_candidates, odometry_delta=odometry_delta, stage1_pose=stage1_world,
        )
