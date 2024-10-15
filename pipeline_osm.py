import sys
import argparse
import os
import cv2
from tqdm import tqdm
import torch
import torchvision
import time
import numpy as np
import random

from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf

from controlnet.diffusers_cnet_txt2img import txt2imgControlNet
from controlnet.diffusers_cnet_inpaint import inpaintControlNet
from rendering import utils
from rendering.models.textured_mesh import TexturedMeshModel
from rendering.dataset import init_dataloaders
from rendering.trainer import dr_eval, forward_texturing

from controlnet.diffusers_cnet_txt2img import txt2imgControlNet
from controlnet.diffusers_cnet_inpaint import inpaintControlNet
from controlnet.diffusers_cnet_img2img import img2imgControlNet
from rendering.dataset import init_dataloaders
from rendering import utils
from rendering.models.textured_mesh import TexturedMeshModel
from rendering.trainer import dr_eval,dr_eval2, forward_texturing

def inpaint_viewpoint(sd_cfg, cnet, save_result_dir, mesh_model, dataloaders, inpaint_view_ids=[(5, 6)]):
    print(f"Project inpaint view {inpaint_view_ids}...")
    view_angle_info = {i:data for i, data in enumerate(dataloaders['train'])}
    inpaint_used_key = ["image", "depth", "uncolored_mask"]
    for i, one_batch_id in tqdm(enumerate(inpaint_view_ids)):
        one_batch_img = []
        for view_id in one_batch_id:
            data = view_angle_info[view_id]
            theta, phi, radius = data['theta'], data['phi'], data['radius']
            outputs = mesh_model.render(theta=theta, phi=phi, radius=radius)
            view_img_info = [outputs[k] for k in inpaint_used_key]
            one_batch_img.append(view_img_info)

        for i, img in enumerate(zip(*one_batch_img)):
            img = torch.cat(img, dim=3)
            if img.size(1) == 1:
                img = img.repeat(1, 3, 1, 1)
            t = '_'.join(map(str, one_batch_id))
            name = inpaint_used_key[i]
            if name == "uncolored_mask":
                img[img>0] = 1
            save_path = os.path.join(save_result_dir, f"view_{t}_{name}.png")
            utils.save_tensor_image(img, save_path=save_path)

    txt_cfg = sd_cfg.txt2img
    img_cfg = sd_cfg.inpaint
    copy_list = ["prompt", "negative_prompt", "seed", ]
    for k in copy_list:
        img_cfg[k] = txt_cfg[k]

    for i, one_batch_id in tqdm(enumerate(inpaint_view_ids)):
        t = '_'.join(map(str, one_batch_id))
        rgb_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[0]}.png")
        depth_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[1]}.png")
        mask_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[2]}.png")

        mask = cv2.imread(mask_path)
        dilate_kernel = 10
        mask = cv2.dilate(mask, np.ones((dilate_kernel, dilate_kernel), np.uint8))
        mask_path = os.path.join(save_result_dir, f"view_{t}_{inpaint_used_key[2]}_d{dilate_kernel}.png")
        cv2.imwrite(mask_path, mask)

        img_cfg.image_path = rgb_path
        img_cfg.mask_path =  mask_path
        img_cfg.controlnet_units[0].condition_image_path = depth_path
        images = cnet.infernece(config=img_cfg)
        for i, img in enumerate(images):
            save_path = os.path.join(save_result_dir, f"view_{t}_rgb_inpaint_{i}.png")
            img.save(save_path)
    return images


def gen_init_view(trial, sd_cfg, cnet, mesh_model, dataloaders, outdir, view_ids=[]):
    print(f"Project init view {view_ids}...")
    init_depth_map = []
    view_angle_info = {i: data for i, data in enumerate(dataloaders['train'])}
    for view_id in view_ids:
        data = view_angle_info[view_id]
        theta, phi, radius = data['theta'], data['phi'], data['radius']
        outputs = mesh_model.render(theta=theta, phi=phi, radius=radius)
        depth_render = outputs['depth']
        init_depth_map.append(depth_render)

    init_depth_map = torch.cat(init_depth_map, dim=0).repeat(1, 3, 1, 1)
    init_depth_map = torchvision.utils.make_grid(init_depth_map, nrow=2, padding=0)
    save_path = os.path.join(outdir, f"init_depth_render.png")
    utils.save_tensor_image(init_depth_map.unsqueeze(0), save_path=save_path)

    depth_dilated = utils.dilate_depth_outline(save_path, iters=5, dilate_kernel=3)
    save_path = os.path.join(outdir, f"init_depth_dilated.png")
    cv2.imwrite(save_path, depth_dilated)

    print("Generating init view...")
    p_cfg = sd_cfg.txt2img
    p_cfg.controlnet_units[0].condition_image_path = save_path

    images = cnet.infernece(config=p_cfg)
    for i, img in enumerate(images):
        save_path = os.path.join(outdir, f'init-img-{i}-{trial}.png')
        img.save(save_path)
    return images

def UV_inpaint(sd_cfg, cnet, mesh_model, outdir,):
    print(f"rendering texture and position map")
    mesh_model.export_mesh(outdir, export_texture_only=True)
    albedo_path = os.path.join(outdir, f"albedo.png")
    UV_pos = mesh_model.UV_pos_render()
    UV_pos_path = os.path.join(outdir, f"UV_pos.png")
    utils.save_tensor_image(UV_pos.permute(0, 3, 1, 2), UV_pos_path)

    mask_dilated = utils.extract_bg_mask(albedo_path, dilate_kernel=15)
    mask_path = os.path.join(outdir, f"mask.png")
    cv2.imwrite(mask_path, mask_dilated)

    p_cfg = sd_cfg.inpaint
    p_cfg.image_path = albedo_path
    p_cfg.mask_path =  mask_path
    p_cfg.controlnet_units[0].condition_image_path = UV_pos_path

    images = cnet.infernece(config=p_cfg)
    res = []
    for i, img in enumerate(images):
        save_path = os.path.join(outdir, f'UV_inpaint_res_{i}.png')
        img.save(save_path)
        res.append((img, save_path))
    return res


def UV_tile(sd_cfg, cnet, mesh_model, outdir,):
    print(f"rendering texture and position map")
    mesh_model.export_mesh(outdir, export_texture_only=True)
    albedo_path = os.path.join(outdir, f"albedo.png")
    UV_pos = mesh_model.UV_pos_render()
    UV_pos_path = os.path.join(outdir, f"UV_pos.png")
    utils.save_tensor_image(UV_pos.permute(0, 3, 1, 2), UV_pos_path)

    # UV inpaint
    p_cfg = sd_cfg.img2img
    p_cfg.image_path = albedo_path
    p_cfg.controlnet_units[0].condition_image_path = UV_pos_path
    p_cfg.controlnet_units[1].condition_image_path = albedo_path

    images = cnet.infernece(config=p_cfg)
    for i, img in enumerate(images):
        save_path = os.path.join(outdir, f'UV_tile_res_{i}.png')
        img.save(save_path)
    return images

def init_process(opt,asset):
    outdir = opt.outdir
    os.makedirs(outdir, exist_ok=True)

    pathdir, filename = Path(opt.render_config).parent, Path(opt.render_config).stem
    sys.path.append(str(pathdir))
    render_cfg = __import__(filename, ).TrainConfig()

    if 'building' not in asset:
        if 'path_road' in asset:
            render_cfg.render.radius = 1.5
            render_cfg.render.base_theta = 10
        else:
            render_cfg.render.radius = 1.5
            render_cfg.render.base_theta = 30

    utils.seed_everything(render_cfg.optim.seed)

    sd_cfg = OmegaConf.load(opt.sd_config)
    sd_cfg2 = OmegaConf.load(opt.sd_config2)
    sd_cfg.txt2img.seed = opt.seed
    sd_cfg2.inpaint.seed = opt.seed
    if 'building' not in asset:
        if 'water' in asset:
            sd_cfg2.inpaint.prompt = 'pure blue lake' 
        if 'forest' in asset or 'vegetation' in asset:
            sd_cfg2.inpaint.prompt = 'pure green grassland'
        if 'area' in asset or 'path_road' in asset: 
            sd_cfg2.inpaint.prompt = 'grey concrete ground'
  
    print('sd_seed:'+str(sd_cfg.txt2img.seed))
    render_cfg.log.exp_path = str(outdir)
    if opt.ip_adapter_image_path is not None:
        sd_cfg.txt2img.ip_adapter_image_path = opt.ip_adapter_image_path
        sd_cfg.inpaint.ip_adapter_image_path = opt.ip_adapter_image_path
        sd_cfg2.img2img.ip_adapter_image_path = opt.ip_adapter_image_path
        sd_cfg2.inpaint.ip_adapter_image_path = opt.ip_adapter_image_path
    if opt.mesh_path is not None:
        render_cfg.guide.shape_path = opt.mesh_path
    if opt.texture_path is not None:
        render_cfg.guide.initial_texture = opt.texture_path
        img = Image.open(opt.texture_path)
        render_cfg.guide.texture_resolution = img.size
    return sd_cfg, sd_cfg2, render_cfg

def parse():
    parser = argparse.ArgumentParser()
    #Rendering params

    parser.add_argument(
        "--sd_config",
        type=str,
        default="controlnet/config/depth_based_osm_template.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--sd_config2",
        type=str,
        default="controlnet/config/UV_based_inpaint_template_2.yaml",
        help="path to refine config which constructs model",
    )
    parser.add_argument(
        "--render_config",
        type=str,
        default="render/config/train_config_render.py",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--obj_save_path",
        type=str,
        help="Obj saving path",
        default=None
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed",
        default=0
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Output file path",
        default=None
    )
    parser.add_argument(
        "--ip_adapter_image_path",
        type=str,
        help="IP adapter image path",
        default=None
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        help="Specific mesh path",
        default=None
    )
    parser.add_argument(
        "--texture_path",
        type=str,
        help="Texture path",
        default=None
    )

    #LLM params
    parser.add_argument(
        "--api_key",
        type=str,
        help="API KEY",
        default=None
    )
    parser.add_argument(
        "--user_prompts",
        type=str,
        help="User instruction of the urban scene",
        default="a university area"
    )
    parser.add_argument(
        "--ip_img",
        type=str,
        help="Ip adapter image",
        default=None
    )
    opt = parser.parse_args()
    return opt


def main():
    opt = parse()

    t1 = time.time()
    llm_plan_dict = {
    'building': [
    "Modern apartment blocks with white and gray facade, large glass windows outlined in black, metal balconies, and wooden doors centered.",
    "Traditional brick buildings with red-brown hue, green shutters, small square windows with white frames, and a central arched wooden door.",
    "Futuristic tower with a silver metallic finish, mirrored windows covering the front, and revolving glass doors at ground level."
    ],
    'path_road': [
        "Smooth asphalt paths winding through the campus, marked with bright white lines for guidance. Adorned with tactile patterns near pedestrian crossings for added safety.",
        "Grey cobblestone roads navigate around the buildings, offering a rustic yet modern feel. The pathways are lined with yellow paint on the edges to highlight bicycle lanes."
    ],
    'forest': [
        "Pure green lush grassland dominated by tall, dark green conifers. The forest floor is covered with light brown fallen needles and the light dances through the canopy."
    ],
    'vegetation': [
        "Vibrant green shrubs and colorful flower beds lining the walkways, adding to the visual appeal. The beds are edged with polished wood borders."
    ],
    'water': [
        "pure blue, a serene water feature with reflective blue surface"
    ],
    'area': [
        "Expansive pedestrian area paved in red clay tiles, providing a vibrant contrast to the greenery. Equipped with dark metal benches under modern street lamps."
    ]  
    }
    for asset in os.listdir(opt.obj_save_path):
        print('Start rendering {}!'.format(asset))
        des = ''
        find = False
        for asset_type, asset_des in llm_plan_dict.items():
            if asset_type in asset:
                des = asset_des[0] if len(asset_des) == 1 else asset_des[random.randrange(0,len(asset_des))]
                print(des)
                opt.mesh_path = os.path.join(opt.obj_save_path,asset,'mesh.obj')
                if opt.outdir is None:
                    opt.outdir = os.path.join(opt.obj_save_path,asset,'output')
                find = True
                break
        if not find:
            print('Skip rendering!')
            opt.outdir = None
            continue
        # print(opt.outdir)
        sd_cfg, sd_cfg2, render_cfg = init_process(opt,asset)
        sd_cfg.txt2img.prompt = des
        sd_cfg2.inpaint.prompt = des
        if opt.ip_img is not None:
            sd_cfg.txt2img.ip_adapter_image_path = opt.ip_img
            sd_cfg.inpaint.ip_adapter_image_path = opt.ip_img
            sd_cfg2.img2img.ip_adapter_image_path = opt.ip_img
            sd_cfg2.inpaint.ip_adapter_image_path = opt.ip_img
        device = torch.device("cuda")
        dataloaders = init_dataloaders(render_cfg, device)
        mesh_model = TexturedMeshModel(cfg=render_cfg, device=device,).to(device)

        depth_cnet = txt2imgControlNet(sd_cfg.txt2img)
        inpaint_cnet = inpaintControlNet(sd_cfg.inpaint)

        total_start = time.time()
        start_t = time.time()
        
        max_trial = 1

        for t in range(max_trial):
            print('Trial '+str(t+1))
            init_images = gen_init_view(
                trial = t+1,
                sd_cfg=sd_cfg,
                cnet=depth_cnet,
                mesh_model=mesh_model,
                dataloaders=dataloaders,
                outdir=opt.outdir,
                view_ids=render_cfg.render.views_init,
            )
            print(f"init view generation time: {time.time() - start_t}")

        for i, init_image in enumerate(init_images):
            outdir = Path(opt.outdir) / f"res-{i}"
            outdir.mkdir(exist_ok=True)
            start_t = time.time()
            mesh_model.initial_texture_path = None
            mesh_model.refresh_texture()
            view_imgs = utils.split_grid_image(img=np.array(init_image), size=(2, 2))
            forward_texturing(
                cfg=render_cfg,
                dataloaders=dataloaders,
                mesh_model=mesh_model,
                save_result_dir=outdir,
                device=device,
                view_imgs=view_imgs,
                view_ids=render_cfg.render.views_init,
                verbose=False,
            )
            print(f"init DR time: {time.time() - start_t}")

            for view_group in render_cfg.render.views_inpaint:   # cloth 4 view
                start_t = time.time()
                print("View inpainting ...")
                outdir = Path(opt.outdir) / f"res-{i}"
                outdir.mkdir(exist_ok=True)
                inpainted_images = inpaint_viewpoint(
                    sd_cfg=sd_cfg,
                    cnet=inpaint_cnet,
                    save_result_dir=outdir,
                    mesh_model=mesh_model,
                    dataloaders=dataloaders,
                    inpaint_view_ids=[view_group],
                )
                print(f"inpaint view generation time: {time.time() - start_t}")


                start_t = time.time()
                view_imgs = []
                for img_t in inpainted_images:
                    view_imgs.extend(utils.split_grid_image(img=np.array(img_t), size=(1, 2)))
                forward_texturing(
                    cfg=render_cfg,
                    dataloaders=dataloaders,
                    mesh_model=mesh_model,
                    save_result_dir=outdir,
                    device=device,
                    view_imgs=view_imgs,
                    view_ids=view_group,
                    verbose=False,
                )
                print(f"inpaint DR time: {time.time() - start_t}")


            print(f"total processed time:{time.time() - total_start}")
            mesh_model.initial_texture_path = f"{outdir}/albedo.png"
            mesh_model.refresh_texture()
            dr_eval(
                cfg=render_cfg,
                sd_cfg = sd_cfg,
                dataloaders=dataloaders,
                mesh_model=mesh_model,
                save_result_dir=outdir,
                valset=True,
                verbose=False,
            )
            mesh_model.empty_texture_cache()
            torch.cuda.empty_cache()
        
        opt.texture_path = opt.outdir+'/res-0/albedo.png'
        opt.outdir += '/refine'
        sd_cfg, sd_cfg2, render_cfg = init_process(opt,asset)
        device = torch.device("cuda")
        mesh_model = TexturedMeshModel(cfg=render_cfg, device=device,).to(device)
        dataloaders = init_dataloaders(render_cfg, device)

        UVInpaint_cnet = inpaintControlNet(sd_cfg2.inpaint)
        UVtile_cnet = img2imgControlNet(sd_cfg2.img2img)

        total_start = time.time()
        start_t = time.time()
        UV_inpaint_res = UV_inpaint(
            sd_cfg=sd_cfg2,
            cnet=UVInpaint_cnet,
            mesh_model=mesh_model,
            outdir=opt.outdir,
        )
        print(f"UV Inpainting time: {time.time() - start_t}")

        outdir = opt.outdir
        mesh_model.initial_texture_path = f"{outdir}/UV_inpaint_res_0.png"
        mesh_model.refresh_texture()
        dr_eval2(
            cfg=render_cfg,
            dataloaders=dataloaders,
            mesh_model=mesh_model,
            save_result_dir=outdir,
            valset=True,
            verbose=False,
        )


        for i, (_, init_img_path) in enumerate(UV_inpaint_res):
            outdir = Path(opt.outdir) / f"tile_res_{i}"
            outdir.mkdir(exist_ok=True)
            start_t = time.time()
            mesh_model.initial_texture_path = init_img_path
            mesh_model.refresh_texture()
            _ = UV_tile(
                sd_cfg=sd_cfg2,
                cnet=UVtile_cnet,
                mesh_model=mesh_model,
                outdir=outdir,
            )
            print(f"UV tile time: {time.time() - start_t}")

            print(f"total processed time:{time.time() - total_start}")
            mesh_model.initial_texture_path = f"{outdir}/UV_tile_res_0.png"
            mesh_model.refresh_texture()
            dr_eval2(
                cfg=render_cfg,
                dataloaders=dataloaders,
                mesh_model=mesh_model,
                save_result_dir=outdir,
                valset=True,
                verbose=False,
            )
            mesh_model.empty_texture_cache()
            torch.cuda.empty_cache()

        opt.outdir = None
    t2 = time.time()
    print('Spend time:',(t2-t1)/60)
if __name__ == '__main__':
    main()
            