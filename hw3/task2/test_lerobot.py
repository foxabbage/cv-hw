import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors


def make_delta_timestamps(delta_indices, fps):
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate(policy_path, dataset_repo_id, dataset_root, device="cuda", batch_size=1):
    policy_path = Path(policy_path)

    # 1. 读取数据集 metadata
    meta = LeRobotDatasetMetadata(dataset_repo_id, root=dataset_root)

    # 2. 读取 policy config
    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path
    cfg.device = device

    # 3. ACT 需要 action chunk target
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, meta.fps),
    }

    # observation 一般只要当前帧；ACT 默认 n_obs_steps=1
    if getattr(cfg, "observation_delta_indices", None) is not None:
        if "observation.state" in meta.features:
            delta_timestamps["observation.state"] = make_delta_timestamps(
                cfg.observation_delta_indices, meta.fps
            )
        for cam_key in meta.camera_keys:
            delta_timestamps[cam_key] = make_delta_timestamps(
                cfg.observation_delta_indices, meta.fps
            )

    # 4. 加载环境 D
    dataset = LeRobotDataset(
        dataset_repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        tolerance_s=1e-3,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # 5. 加载 policy
    # ds_meta 只用于检查 feature shape；zero-shot 的 normalization 应来自 checkpoint processor
    policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
    policy.eval()
    policy.to(device)

    # 6. 加载 checkpoint 里保存的 pre/post processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
    )

    abs_errors = []
    sq_errors = []
    per_dim_abs = None
    n = 0

    # ACT 有 action queue；离线逐帧评估时不要跨 episode 泄漏
    last_episode = None

    for batch in loader:
        # 图像如果是 uint8，需要转 float / 255
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].float() / 255.0

        batch = move_to_device(batch, device)

        # episode 边界 reset，避免 ACT 上一个 episode 的 action chunk 影响下一个 episode
        if "episode_index" in batch:
            ep = int(batch["episode_index"][0].item())
            if last_episode is None or ep != last_episode:
                if hasattr(policy, "reset"):
                    policy.reset()
                last_episode = ep

        # 取 observation 输入
        obs = {}
        for k in cfg.input_features.keys():
            if k in batch:
                obs[k] = batch[k]

        obs = preprocessor(obs)

        # 模型输出一个当前要执行的 action
        pred_action = policy.select_action(obs)
        pred_action = postprocessor(pred_action)

        target = batch["action"]

        if target.ndim == 3:
            target = target[:, 0, :]
        
        target = target.to(device=pred_action.device, dtype=pred_action.dtype)


        # 保证 shape: [B, action_dim]
        err = pred_action - target

        abs_errors.append(err.abs().detach().cpu())
        sq_errors.append((err ** 2).detach().cpu())

        if per_dim_abs is None:
            per_dim_abs = err.abs().sum(dim=0).detach().cpu()
        else:
            per_dim_abs += err.abs().sum(dim=0).detach().cpu()

        n += err.shape[0]

    abs_errors = torch.cat(abs_errors, dim=0)
    sq_errors = torch.cat(sq_errors, dim=0)

    result = {
        "num_frames_eval": int(n),
        "mae_all_dims": float(abs_errors.mean().item()),
        "rmse_all_dims": float(torch.sqrt(sq_errors.mean()).item()),
        "mae_per_dim": (per_dim_abs / n).tolist(),
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--dataset-repo-id", default="local/envD")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = evaluate(
        policy_path=args.policy,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        device=args.device,
    )

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))