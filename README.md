# UrbanWorld
The official implementation of the paper "UrbanWorld: An Urban World Model for 3D City Generation"

## Setup
1. Install dependencies
```bash
conda create -n blend python=3.7
conda activate blend
pip install -r requirements-blend.txt
 
conda create -n urbanworld python=3.8
conda activate urbanworld
pip install -r requirements-urbanworld.txt
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html
```

2. Install [blender](https://www.blender.org/) (tested version: blender-3.2.2-linux-x64).

3. Download different versions of ControlNet from [Hugging Face](https://huggingface.co/models?sort=downloads&search=controlnet) and UrbanWorld-UV-ControlNet from [here](https://huggingface.co/Urban-World/UrbanWorld-UV-control).

## Quick Start
1. Modify the settings in `run_osm.sh` and configs in `controlnet/config/`.

2. Run the following command:
```bash
bash run_osm.sh 
