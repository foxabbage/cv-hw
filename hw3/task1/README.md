## threesudio

使用threestudio完成以下任务：text to 3d, single image to 3d

### 环境及仓库的下载与安装

PyTorch  2.1.2
Python  3.10(ubuntu22.04)
CUDA  11.8

```bash
conda create -n threestudio python=3.10 -y && conda activate threestudio
git clone https://github.com/threestudio-project/threestudio.git
pip install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 --index-url https://download.pytorch.org/whl/cu118
pip install setuptools==81.0.0
pip install -r requirements.txt --no-build-isolation  # 使用仓库to_change中提供的requirements.txt 或 使用threestudio中提供的requirements.txt，找到并注释下列两行：lightning==2.0.0 | xformers
pip install pybind11 swanlab
wget https://api.anaconda.org/download/xformers/xformers/0.0.23.post1/linux-64/xformers-0.0.23.post1-py310_cu11.8.0_pyt2.1.2.tar.bz2
conda install xformers-0.0.23.post1-py310_cu11.8.0_pyt2.1.2.tar.bz2
pip install lightning=2.1.0
cd threestudio && pip install -e . && cd ..
```

### 3D训练

使用仓库中的to_change下的两个配置文件替换对应configs文件，并将swanlab.py和banana.png文件夹放到threestudio的根目录下

#### text to 3d

```bash
cd threestudio
python swanlab_launch.py \
  --config configs/magic3d-coarse-sd.yaml \
  --train --gpu 0 \
  system.prompt_processor.prompt="a yellow ripe banana"

python swanlab_launch.py \
  --config configs/magic3d-refine-sd.yaml \
  --train --gpu 0 \
  system.prompt_processor.prompt="a yellow ripe banana" \
  system.geometry_convert_from=./outputs/magic3d-coarse-sd/banana/ckpts/last.ckpt \
  trainer.max_steps=5000

python swanlab_launch.py \
  --config outputs/magic3d-refine-sd/banana/configs/parsed.yaml \
  --export --gpu 0 \
  resume=outputs/magic3d-refine-sd/banana/ckpts/last.ckpt \
  system.exporter_type=mesh-exporter \
  system.exporter.fmt=obj
```

#### single image to 3d

```bash
python swanlab_launch.py \
  --config configs/magic123-coarse-sd.yaml \
  --train --gpu 0 \
  data.image_path=banana.png \
  system.prompt_processor.prompt="a yellow ripe banana"

python swanlab_launch.py \
  --config configs/magic123-refine-sd.yaml \
  --train --gpu 0 \
  system.prompt_processor.prompt="a yellow ripe banana" \
  data.image_path=banana.png \
  system.geometry_convert_from=./outputs/magic123-coarse-sd/banana/ckpts/last.ckpt \
  trainer.max_steps=5000

python swanlab_launch.py \
  --config outputs/magic123-refine-sd/banana/configs/parsed.yaml \
  --export --gpu 0 \
  resume=outputs/magic123-refine-sd/banana/ckpts/last.ckpt \
  system.exporter_type=mesh-exporter \
  system.exporter.fmt=obj
```

## colmap & 2dgs

使用colmap和2dgs完成以下任务：多视角重建；Mip-NeRF 360中的garden重建

### colmap安装

PyTorch  2.7.0
Python  3.12(ubuntu22.04)
CUDA  12.8

```bash
git clone https://github.com/colmap/colmap
cd colmap
sudo apt-get install \
  git \
  cmake \
  ninja-build \
  build-essential \
  libboost-program-options-dev \
  libboost-graph-dev \
  libboost-system-dev \
  libeigen3-dev \
  libopenimageio-dev \
  openimageio-tools \
  libmetis-dev \
  libgoogle-glog-dev \
  libgtest-dev \
  libgmock-dev \
  libsqlite3-dev \
  libglew-dev \
  qt6-base-dev \
  libqt6svgwidgets6 \
  libqt6svg6-dev \
  libqt6svg6 \
  libqt6opengl6-dev \
  libqt6openglwidgets6 \
  libcgal-dev \
  libceres-dev \
  libsuitesparse-dev \
  libcurl4-openssl-dev \
  libssl-dev \
  libmkl-full-dev
sudo mkdir -p /usr/include/opencv4
mkdir build
cd build
cmake .. -GNinja -DBLA_VENDOR=Intel10_64lp
ninja
sudo ninja install
```

### 2dgs安装

PyTorch  2.0.0
Python  3.8(ubuntu20.04)
CUDA  11.8

```bash
git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive
cd 2d-gaussian-splatting
conda env create --file environment.yml
conda activate surfel_splatting
```

### 视频预处理

安装rembg和ffmpeg

```bash
sudo apt update
sudo apt install ffmpeg
pip install "rembg[gpu,cli]"
```

将视频转换为图片并去除背景

```bash
ffmpeg -i b.mp4 -vf "fps=2" frame_%04d.png
cd ..
rembg p frames/ banana/
```

### colmap提取位姿

将去除背景的图片进行人工筛选，除去被去除大量前景和遗留大量背景的图片

```bash
colmap feature_extractor \
  --database_path database.db \
  --image_path banana \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model PINHOLE

colmap sequential_matcher \
  --database_path ./database.db

colmap exhaustive_matcher \
  --database_path ./database.db

colmap mapper \
  --database_path ./database.db \
  --image_path ./banana \
  --output_path ./sparse
```

文件夹./sparse/0下即位提取的位姿

### 2dgs

将先前的图片文件夹和sparse文件夹放入input文件夹下，把input文件夹放在2d-gaussian-splatting文件夹下

```bash
python train.py \
  -s input \
  -m output \
  --iterations 15000

python render.py -m ./output -s ./input
```

后处理，去除噪声,只保留主体部分

```bash
conda create -n ply python=3.12 -y
conda activate ply
pip install plyfile numpy scipy trimesh
python process.py ./multi/input.ply ./multi/output.ply
```

### garden重建

下载garden数据集至garden文件夹，把garden文件夹放在2d-gaussian-splatting文件夹下，使其下包含images和sparse两个文件夹

```bash
python train.py \
  -s ./garden \
  -m ./garden_output

python render.py \
  -s ./garden \
  -m ./garden_output \
  --unbounded \
  --skip_test \
  --skip_train \
  --mesh_res 1024
```

## 导出和渲染

导出三个文件(两个obj文件和两个ply文件)，放入blender进行渲染

## 光滑度测试

使用仓库中的test_smoothness.py进行测试，将ply文件导出为obj文件，并将main中对应的路径修改为你的模型对应的路径
