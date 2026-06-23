### 环境配置

PyTorch  2.8.0
Python  3.12(ubuntu22.04)
CUDA  12.8

```bash
pip install lerobot torch==2.8.0 torchcodec==0.7.0
sudo apt update
sudo apt install ffmpeg
```

### 数据集下载与处理

```bash
hf download huiwon/calvin_task_ABC_D\
  --repo-type dataset \
  --local-dir ./hf-datasets

# 重命名0,1,2,3结尾的四个文件夹分别为splitA, splitB, splitC, splitD
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=huiwon/calvin_task_ABC_D --root=./hf-datasets/splitA
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=huiwon/calvin_task_ABC_D --root=./hf-datasets/splitB
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=huiwon/calvin_task_ABC_D --root=./hf-datasets/splitC
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=huiwon/calvin_task_ABC_D --root=./hf-datasets/splitD
lerobot-edit-dataset \
  --operation.repo_ids "['huiwon/calvin_task_ABC_D', 'huiwon/calvin_task_ABC_D', 'huiwon/calvin_task_ABC_D']" \
  --operation.type merge \
  --new_repo_id ${USER_NAME}/calvin_ABC \
  --operation.roots "['./hf-datasets/splitA', './hf-datasets/splitB', './hf-datasets/splitC']"\
  --new_root ./hf-datasets/splitABC \
  --push_to_hub false
```

### 训练

```bash
lerobot-train \
  --dataset.repo_id=huiwon/calvin_task_ABC_D \
  --dataset.root=./hf-datasets/splitB \
  --tolerance_s=0.001\
  --policy.type=act \
  --output_dir=outputs/train/splitB \
  --job_name=splitB \
  --policy.device=cuda \
  --policy.repo_id=${USER_NAME}/splitB \
  --wandb.enable=true \
  --wandb.mode=offline

lerobot-train \
  --dataset.repo_id=huiwon/calvin_task_ABC_D \
  --dataset.root=./hf-datasets/splitABC \
  --tolerance_s=0.001\
  --policy.type=act \
  --output_dir=outputs/train/splitABC \
  --job_name=splitABC \
  --policy.device=cuda \
  --policy.repo_id=${USER_NAME}/splitABC \
  --wandb.enable=true \
  --wandb.mode=offline
```

训练结束后将wandb日志上传可视化防止网络不稳定

### 测试

使用仓库中提供的test_lerobot.py文件进行离线测试

```bash
python test_lerobot.py \
  --policy outputs/train/splitB/checkpoints/last/pretrained_model \
  --dataset-repo-id huiwon/calvin_task_ABC_D \
  --dataset-root ./hf-datasets/splitD \
  --out eval_splitB_envD.json

python test_lerobot.py \
  --policy outputs/train/splitABC/checkpoints/last/pretrained_model \
  --dataset-repo-id huiwon/calvin_task_ABC_D \
  --dataset-root ./hf-datasets/splitD \
  --out eval_splitABC_envD.json
```