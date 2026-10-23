"""Train goal-sensitive route scoring with a frozen PointGoal bridge controller."""
import argparse
import json
import hashlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'diffusion_policy'))

import torch
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed, send_to_device, broadcast
from omegaconf import OmegaConf

from model.topk_route_bridge import TopKRouteBridge as PointBridgePlanner, CHECKPOINT_FORMAT
from datasets.pointgoal_bridge_dataset import load_manifest, select_training_manifest
from datasets.hypothesis_route_dataset import HypothesisRouteLMDB as PointBridgeLMDB
from utils.validate_topk_routes import validate_routes
from datasets.lmdb_dataset import assign_lmdb_manifest


class BridgeSystem(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.get('route_task')=='object':
            from model.object_topk_bridge import ObjectTopKRouteBridge
            self.planner=ObjectTopKRouteBridge(config)
        else:self.planner = PointBridgePlanner(config)

    def forward(self, batch):
        planner = self.planner(batch)
        loss = planner['loss']
        metrics = {'planner_loss': loss.detach(), 'planner_bridge': planner['bridge_loss'],
                   'planner_gate': planner['gate_loss']}
        metrics['shallow_weight'] = planner['shallow_weight']
        for key in ('route_loss','bridge_coverage','label_coverage','predicted_compatible','route_entropy','reject_loss','reject_accuracy','goal_coverage','active_fraction'):
            metrics[key] = planner[key]
        metrics['failed_astar_fraction'] = (~batch['candidate_mask'].any(-1)).float().mean()
        return loss, metrics


def build_loader(config, accelerator, max_batches=None, max_trajectories=None):
    batch_size = int(config['batch_size'])
    workers = int(config.get('num_workers', 6))
    if config['data_backend'] == 'lmdb':
        workers = int(config.get('lmdb_num_workers', 1))
        if workers not in (0, 1):
            raise ValueError('LMDB requires num_workers 0 or 1')
        # Filter/count before DDP assignment; source LMDB stays read-only.
        manifest = select_training_manifest(load_manifest(config['lmdb_path'], config.get('subset_manifest')), config)
        heldout=set(json.loads(Path(config['validation_buildings_file']).read_text()))
        manifest=[m for m in manifest if m['floor_key'].split('::')[-1].rsplit('_',1)[0] not in heldout]
        accelerator.print('Selected trajectories:', len(manifest), 'windows:',
                          sum(item['sample_count'] for item in manifest))
        accelerator.print('LMDB manifest SHA256:', hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()).hexdigest())
        if max_trajectories:
            manifest = manifest[:max_trajectories]
        assignments, totals = assign_lmdb_manifest(manifest, accelerator.num_processes)
        samples = min(totals) // batch_size * batch_size
        if max_batches:
            samples = min(samples, max_batches * batch_size)
        if samples == 0:
            raise ValueError('Not enough samples for every DDP rank')
        dataset = PointBridgeLMDB(config, assignments[accelerator.process_index], samples, int(config.get('shuffle_seed', 0)))
        sampler = None
        accelerator.print('LMDB rank samples:', totals, 'equal samples/rank:', samples)
    elif config['data_backend'] == 'files':
        raise ValueError('Stage-one training currently requires the audited LMDB backend')
    else:
        raise ValueError('Supported bridge backends: files, lmdb')
    manual = config['data_backend'] == 'lmdb'
    if not manual and int(config.get('shuffle_block_size', 0)):
        # Same optional global block sampler as original; Accelerate shards it.
        from datasets.sampler import BlockShuffleDistributedSampler
        sampler = BlockShuffleDistributedSampler(dataset,
            block_size=int(config['shuffle_block_size']), num_replicas=1, rank=0,
            seed=int(config.get('shuffle_seed', 0)),
            shuffle_within_block=bool(config.get('shuffle_within_block', True)))
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        shuffle=not manual and sampler is None, num_workers=workers,
                        pin_memory=True, persistent_workers=bool(workers and not manual),
                        prefetch_factor=int(config.get('prefetch_factor', 2)) if workers else None)
    return loader if manual else accelerator.prepare(loader)


def build_optimizer(system, config):
    name = str(config.get('optimizer', 'adam')).lower()
    if name == 'adam':
        return torch.optim.Adam((p for p in system.parameters() if p.requires_grad), lr=float(config['learning_rate']))
    if name == 'adamw':
        return torch.optim.AdamW((p for p in system.parameters() if p.requires_grad), lr=float(config['learning_rate']),
                                 weight_decay=float(config.get('weight_decay', .01)))
    raise ValueError('Unsupported optimizer: ' + name)


def save_planner(system, accelerator, config, step, path):
    # Evaluation uses the actual trained parameters, matching original (no EMA).
    if accelerator.is_main_process:
        from model.object_topk_bridge import CHECKPOINT_FORMAT as OBJECT_FORMAT
        package = dict(format=OBJECT_FORMAT if config.get('route_task')=='object' else CHECKPOINT_FORMAT, config=config, step=step,
                       weights_kind='raw',
                       planner=accelerator.unwrap_model(system).planner.state_dict())
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(package, temporary)
        os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'config/topk_route_bridge.yaml'))
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--resume', help='Full accelerate state directory from this run')
    parser.add_argument('--stop-after-epoch', type=int, help='Save and stop after this 1-based epoch; config stays unchanged for resume')
    parser.add_argument('--max-steps', type=int, help='Bound optimizer steps for a smoke run')
    parser.add_argument('--max-batches', type=int, help='Bound batches per epoch')
    parser.add_argument('--max-trajectories', type=int, help='Read a manifest subset for smoke tests')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--gradient-accumulation-steps', type=int)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    for name in ['batch_size', 'num_workers', 'gradient_accumulation_steps']:
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if args.num_workers is not None and config['data_backend'] == 'lmdb':
        config['lmdb_num_workers'] = args.num_workers
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= config['epochs']:
        raise ValueError('stop-after-epoch must be in [1, configured epochs]')
    accumulation = int(config['gradient_accumulation_steps'])
    if accumulation < 1 or config['batch_size'] < 1:
        raise ValueError('Batch/accumulation must be positive')
    if args.cpu:
        # Installed Accelerate otherwise queries torch.cpu.device_count (absent in Torch 2.0).
        os.environ['ACCELERATE_TORCH_DEVICE'] = 'cpu'
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=config.get('mixed_precision', 'no'),
                              gradient_accumulation_steps=1,
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])
    set_seed(config['seed'] + accelerator.process_index)
    run = Path(config['run_dir']) / args.run_name
    exists = torch.tensor([int(run.exists() and not args.resume) if accelerator.is_main_process else 0], device=accelerator.device)
    exists = broadcast(exists, from_process=0)
    if exists.item():
        raise FileExistsError('Use a fresh run-name or --resume: ' + str(run))
    if accelerator.is_main_process:
        run.mkdir(parents=True, exist_ok=True)
        (run / 'config.yaml').write_text(OmegaConf.to_yaml(config))
    accelerator.wait_for_everyone()
    system = BridgeSystem(config)
    if config.get('init_planner_checkpoint') and not args.resume:
        saved = torch.load(config['init_planner_checkpoint'], map_location='cpu')
        if saved.get('format') != 'floverse-pointgoal-world-rgb4-planner-v4':
            raise ValueError('Warm start requires an aligned RGB4 PointGoal bridge')
        state = system.planner.state_dict()
        for key, value in saved['planner'].items():
            if key not in state or state[key].shape != value.shape:
                raise ValueError('Incompatible warm-start field: '+key)
        system.planner.load_state_dict(saved['planner'], strict=False)
    optimizer = build_optimizer(system, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'],eta_min=float(config.get('min_learning_rate',1e-6)))
    # Explicitly one scheduler step per epoch; save/resume this plain scheduler.
    system, optimizer = accelerator.prepare(system, optimizer)
    accelerator.register_for_checkpointing(scheduler)
    loader = build_loader(config, accelerator, args.max_batches, args.max_trajectories)
    epoch_start, skip_batches, step = 0, 0, 0
    if args.resume:
        progress = json.loads((Path(args.resume) / 'progress.json').read_text())
        if progress['config'] != config or progress['world_size'] != accelerator.num_processes:
            raise ValueError('Resume requires identical config and DDP world size')
        selection = dict(max_batches=args.max_batches, max_trajectories=args.max_trajectories)
        if progress.get('data_selection', selection) != selection:
            raise ValueError('Resume requires the same dataset subset and epoch length')
        epoch_start, skip_batches, step = progress['epoch'], progress['batch'], progress['step']
        accelerator.load_state(args.resume)
    log_path = run / 'metrics.jsonl'
    stop = False
    validation_log=run/'validation.jsonl'
    best_path=run/'best_validation.json'
    best_validation=json.loads(best_path.read_text())['selection_loss'] if best_path.exists() else float('inf')
    for epoch in range(epoch_start, config['epochs']):
        if hasattr(loader.dataset, 'set_epoch'):
            loader.dataset.set_epoch(epoch)
        accelerator.unwrap_model(system).planner.epoch = epoch
        system.train()
        optimizer.zero_grad(set_to_none=True)
        batches = min(len(loader), args.max_batches or len(loader))
        next_batch = 0
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            if index < skip_batches:
                continue
            begin = time.monotonic()
            batch = send_to_device(batch, accelerator.device, non_blocking=True)
            window = min(accumulation, batches - (index // accumulation) * accumulation)
            loss, metrics = system(batch)
            accelerator.backward(loss / window)
            boundary = (index + 1) % accumulation == 0 or index + 1 == batches
            if boundary:
                accelerator.clip_grad_norm_(system.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            if int(config.get('eval_per_batches',0))>0 and index != 0 and index % int(config['eval_per_batches']) == 0:
                # Training-batch trajectory diagnostic; sampling never reads supervision labels.
                # Keep train mode as in original, but never backpropagate this sample.
                with torch.no_grad():
                    prediction = accelerator.unwrap_model(system).planner(batch, mode='sample')
                    metrics['planner/eval_traj_cosine_sim'] = torch.nn.functional.cosine_similarity(
                        prediction.flatten(1), batch['gt_pose'].flatten(1), dim=1).mean()
                if accelerator.is_main_process:
                    torch.save(dict(prediction=prediction.cpu(), target=batch['gt_pose'].cpu()),
                               run / ('planner_epoch_%d_batch_%d_diagnostic.pt' % (epoch + 1, index)))
            if index != 0 and index % int(config.get('save_pre_batches', 500)) == 0:
                accelerator.wait_for_everyone()
                save_planner(system, accelerator, config, step,
                             run / ('planner_epoch_%d_%d.pth' % (epoch + 1, index)))
            metrics = {key: float(accelerator.reduce(value.float(), reduction='mean').item()) for key, value in metrics.items()}
            metrics.update(epoch=epoch, batch=index, step=step, learning_rate=optimizer.param_groups[0]['lr'], seconds=time.monotonic() - begin)
            if accelerator.is_main_process and (index % config.get('log_every', 10) == 0 or boundary and args.max_steps):
                print(json.dumps(metrics), flush=True)
                with log_path.open('a') as handle:
                    handle.write(json.dumps(metrics) + '\n')
            next_batch = index + 1
            if boundary and args.max_steps and step >= args.max_steps:
                stop = True
                break
        skip_batches = 0
        completed_epoch = next_batch >= batches
        if completed_epoch:
            scheduler.step()
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                validation=validate_routes(accelerator.unwrap_model(system).planner,config,accelerator.device)
                validation.update(epoch=epoch+1,step=step)
                with validation_log.open('a') as handle:handle.write(json.dumps(validation)+'\n')
                accelerator.print('Validation',json.dumps(validation))
                if validation['selection_loss']<best_validation-float(config.get('validation_min_delta',0.)):
                    best_validation=validation['selection_loss']
                    save_planner(system,accelerator,config,step,run/'planner_best_route_validation.pth')
                    (run/'best_validation.json').write_text(json.dumps(validation,indent=2))
            accelerator.wait_for_everyone()
        progress = dict(epoch=epoch + int(completed_epoch), batch=0 if completed_epoch else next_batch,
                        step=step, config=config, world_size=accelerator.num_processes,
                        data_selection=dict(max_batches=args.max_batches, max_trajectories=args.max_trajectories))
        # Full optimizer/scaler/RNG state per rank; saves only at optimizer boundaries.
        state_dir = run / ('state_step_%08d' % step)
        accelerator.save_state(str(state_dir))
        if accelerator.is_main_process:
            (state_dir / 'progress.json').write_text(json.dumps(progress, indent=2))
        save_planner(system, accelerator, config, step, run / 'planner_latest.pth')
        if completed_epoch and config.get('save_epoch_checkpoints', True):
            save_planner(system, accelerator, config, step,
                         run / ('planner_epoch_%d_final.pth' % (epoch + 1)))
        accelerator.print('Saved', state_dir, 'and', run / 'planner_latest.pth')
        accelerator.wait_for_everyone()
        if stop or (completed_epoch and args.stop_after_epoch is not None and epoch + 1 >= args.stop_after_epoch):
            break


if __name__ == '__main__':
    main()
