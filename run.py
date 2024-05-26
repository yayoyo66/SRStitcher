import torch
from PIL import Image
import os
import cv2
from diffusers import ControlNetModel, AutoencoderKL
import math
from omegaconf import OmegaConf
import argparse

from util.preprocessing import merged_img, preprocess_image, preprocess_map, make_inpaint_condition
from util.weighted_mask import make_mask

# SRStitcher Pipes
from pipes.diff_pipe_inpaint import DiffusionDiffInpaintingPipeline
from pipes.diff_pipe_SD2 import StableDiffusionDiffImg2ImgPipeline
from pipes.diff_pipe_unclip import StableUnCLIPImg2ImgPipeline
from pipes.diff_pipe_control import StableDiffusionControlNetInpaintPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="SRStitcher.")
    parser.add_argument(
        "--config",
        type=str,
        default="inpaint_config.yaml",
    )
    args = parser.parse_args()
    return args


def calculate_k(image_width, lamb):
    k = image_width / lamb
    k = math.ceil(k) * 10
    return (k, k)


def main(cfg):
    device = cfg.device
    if cfg.mode == "SD2-inpaint":
        pipe = DiffusionDiffInpaintingPipeline.from_pretrained(cfg.pretrained_model_name_or_path,
                                                               safety_checker=None,
                                                               torch_dtype=torch.float16).to(device)
    elif cfg.mode == "SD2":
        pipe = StableDiffusionDiffImg2ImgPipeline.from_pretrained(cfg.pretrained_model_name_or_path,
                                                                  safety_checker=None,
                                                                  torch_dtype=torch.float16).to(device)
    elif cfg.mode == "unclipSD2":
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(cfg.pretrained_model_name_or_path,
                                                           safety_checker=None,
                                                           torch_dtype=torch.float16).to(device)
    elif cfg.mode == "controlnet":
        controlnet = ControlNetModel.from_pretrained(cfg.controlnet_path, torch_dtype=torch.float16).to(device)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(cfg.pretrained_model_name_or_path,
                                                                        controlnet=controlnet,
                                                                        torch_dtype=torch.float16,
                                                                        safety_checker=None).to(device)

    path = cfg.datapath
    save_dir = cfg.save_dir
    generator = torch.Generator(device="cuda").manual_seed(cfg.seed)

    # Check if the folder exists
    if not os.path.exists(save_dir):
        # Folder does not exist, so create the folder
        os.makedirs(save_dir)
        print(f"Folder '{save_dir}' created.")

    if not os.path.exists('coarse'+save_dir):
        # Folder does not exist, so create the folder
        os.makedirs('coarse'+save_dir)
        print(f"Folder '{'coarse'+save_dir}' created.")

    R = cfg.R

    warp1_path = os.path.join(path, 'warp1')
    warp2_path = os.path.join(path, 'warp2')
    mask1_path = os.path.join(path, 'mask1')
    mask2_path = os.path.join(path, 'mask2')

    names = sorted(os.listdir(warp1_path))

    for name in names:

        warp1 = cv2.imread(os.path.join(warp1_path, name))
        warp2 = cv2.imread(os.path.join(warp2_path, name))
        mask1 = cv2.imread(os.path.join(mask1_path, name), cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(os.path.join(mask2_path, name), cv2.IMREAD_GRAYSCALE)

        _, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
        _, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

        coarse_fusion_img = merged_img(warp1, warp2, mask1, mask2)

        K = calculate_k(coarse_fusion_img.shape[1], cfg.lamb)

        h, w, c = coarse_fusion_img.shape

        newsize = (cfg.size, cfg.size)

        map, content_mask, map2, map3= make_mask(mask1, mask2, K, cfg.epsilon1, cfg.epsilon2)
        map = preprocess_map(map, newsize).to(device)
        map3 = preprocess_map(map3, newsize).to(device)

        Image.fromarray(cv2.cvtColor(coarse_fusion_img, cv2.COLOR_BGR2RGB)).save(os.path.join('coarse'+save_dir, name))

        coarse_rectangling_img = Image.fromarray(
            cv2.cvtColor(cv2.inpaint(coarse_fusion_img, content_mask, R, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB))

        image = preprocess_image(coarse_rectangling_img, newsize)

        if cfg.mode == "SD2-inpaint":
            map2 = Image.fromarray(map2).resize((512, 512))
            edited_image = pipe(
                prompt=[""],
                image=image,
                guidance_scale=cfg.guidance_scale,
                num_images_per_prompt=1,
                mask_image=map2,
                generator=generator,
                map=map,
                num_inference_steps=cfg.num_inference_steps,
            ).images[0]
        elif cfg.mode == "SD2":
            edited_image = pipe(
                prompt=[""],
                image=image,
                guidance_scale=cfg.guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
                map=map3,
                num_inference_steps=cfg.num_inference_steps,
            ).images[0]
        elif cfg.mode == "unclipSD2":
            edited_image = pipe(prompt=[""],
                                image=image,
                                guidance_scale=7.5,
                                num_images_per_prompt=1,
                                generator=generator,
                                map=map3,
                                num_inference_steps=50).images[0]
        elif cfg.mode == "controlnet":
            map2 = Image.fromarray(map2).resize((512, 512))
            control_image = make_inpaint_condition(image, map2)
            edited_image = pipe(prompt=[""],
                                image=image,
                                guidance_scale=7.5,
                                num_images_per_prompt=1,
                                generator=generator,
                                mask_image=Image.fromarray(mask2).resize((512, 512)),
                                map=map,
                                control_image=control_image.to(device),
                                num_inference_steps=50).images[0]

        edited_image.resize((w, h)).save(os.path.join(save_dir, name))

        print('processing image... %s completed' % name)


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    main(config)