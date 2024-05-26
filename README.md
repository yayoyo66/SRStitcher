# Reconstructing the Image Stitching Pipeline: Integrating Fusion and Rectangling into a Unified Inpainting Model(SRStitcher)
Deep learning-based image stitching pipelines are typically divided into three cascading stages: registration, fusion, and rectangling. Each stage requires its own network training and is tightly coupled to the others, leading to error propagation and posing significant challenges to parameter tuning and system stability. This paper proposes the Simple and Robust Stitcher (SRStitcher), which revolutionizes the image stitching pipeline by simplifying the fusion and rectangling stages into a unified inpainting model, requiring no model training or fine-tuning. We reformulate the problem definitions of the fusion and rectangling stages and demonstrate that they can be effectively integrated into an inpainting task. Furthermore, we design the weighted masks to guide the reverse process in a pre-trained large-scale diffusion model, implementing this integrated inpainting task in a single inference. Through extensive experimentation, we verify the interpretability and generalization capabilities of this unified model, demonstrating that SRStitcher outperforms state-of-the-art methods in both performance and stability.

## Table of Contents

- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)

## Requirements
- Python >= 3.9
- GPU (NVIDIA CUDA compatible)
  
- Create a virtual environment (optional but recommended):

    ```bash
    conda create -n srstitcher python==3.9
    conda activate srstitcher
    ```
    
- Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
  
 **Notice: check `transformers==4.35.2`, other version may report errors**
 
## Dataset
 
 We provide a `examples` document to reproduce Figure 2 from our paper
 
 The complete UDIS-D dataset can be obtained from  [UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching) 
  
Aligned images and masks can be obtained by  [UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching) or 
[UDIS++](https://github.com/nie-lang/UDIS2) 

The datasets should be organized as follows: 

```
dataset
├── warp1
│   ├── 000001.jpg
│   ├── ...
├── warp2
│   ├── 000001.jpg
│   ├── ...
├── mask1
│   ├── 000001.jpg
│   ├── ...
├── mask2
│   ├── 000001.jpg
│   ├── ...
```

## Usage

- Run the script to get SRStitcher results of Figure 2:

    ```bash
    python run.py  --config configs/inpaint_config.yaml
    ```
  
  see results in document `SRStitcherResults`.
  
- Run the script to measure the CCS of stitched image:

    ```bash
    python evaluation_ccs.py
    ```
   
 - Run the script to get SRStitcher-S results of UDIS-D:
 
 Modify the `datapath` in `configs/SD2_config.yaml`
   
 ## Variants
 
 We provide the implementation of three variants.
 
 ### SRStitcher-S
 Implementation version of SRStitcher on the [
stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) 
 
 - Run the script to get SRStitcher-S results:
     ```bash
    python run.py  --config configs/SD2_config.yaml
    ```

 ### SRStitcher-U
 Implementation version of SRStitcher on the [
stable-diffusion-2-1-unclip-small ](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip-small) 

 - Run the script to get SRStitcher-U results:
     ```bash
    python run.py  --config configs/unclipSD2_config.yaml
    ```
   
 ### SRStitcher-C
 Implementation version of SRStitcher on the [
control_v11p_sd15_inpaint ](https://huggingface.co/lllyasviel/control_v11p_sd15_inpaint) 

 - Run the script to get SRStitcher-U results:
     ```bash
    python run.py  --config configs/controlnet_config.yaml
    ```
