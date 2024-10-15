# UrbanWorld
The official implementation of the paper "UrbanWorld: An Urban World Model for 3D City Generation"

## Setup
1. Install dependencies
```bash
conda env create -f blend.yml
conda env create -f urbanworld.yml
```

2. Install [blender](https://www.blender.org/) (tested version: blender-3.2.2-linux-x64).

3. Download different versions of ControlNet from [Hugging Face](https://huggingface.co/models?sort=downloads&search=controlnet) and UrbanWorld-UV-ControlNet from [here](https://huggingface.co/Urban-World/UrbanWorld-UV-control).

## Quick Start
1. Modify the settings in `run_osm.sh` and configs in `/controlnet/config/`.

2. Run the following command:
```bash
bash run_osm.sh 
