# Streamlining the Image Stitching Pipeline: Integrating Fusion and Rectangling into a Unified Model(SRStitcher)
This is the official repository of the paper "Streamlining the Image Stitching Pipeline: Integrating Fusion and Rectangling into a Unified Model".

> Learning-based image stitching techniques typically involve three distinct stages: registration, fusion, and rectangling. These stages are often performed sequentially, each trained independently, leading to potential cascading error propagation and complex parameter tuning challenges. In rethinking the mathematical modeling of the fusion and rectangling stages, we discovered that these processes can be effectively combined into a single, variety-intensity inpainting problem. Therefore, we propose the Simple and Robust Stitcher (SRStitcher), an efficient image stitching pipeline that merges the fusion and rectangling stages into a unified model. By employing the weighted mask and large-scale generative model, SRStitcher can solve the fusion and rectangling problems in a single inference, without additional training or fine-tuning of other models. Our method not only simplifies the stitching pipeline but also enhances fault tolerance towards misregistration errors. Extensive experiments demonstrate that SRStitcher outperforms state-of-the-art (SOTA) methods in both quantitative assessments and qualitative evaluations.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)

## Requirements
- Python >= 3.9
- GPU (NVIDIA CUDA compatible)
  
- Create a virtual environment (optional but recommended):

    ```bash
    conda create -n name python==3.9
    ```
    
- Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage
Aligned images and masks can be obtained from  [UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching)  or [UDIS++](https://github.com/nie-lang/UDIS2) 

The datasets are organized as follows: 

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

- Run the script:

    ```bash
    python run.py
    ```
## Citation

Please cite our paper if you find this code useful for your research.

```
@misc{xie2024streamlining,
      title={Streamlining the Image Stitching Pipeline: Integrating Fusion and Rectangling into a Unified Model}, 
      author={Ziqi Xie},
      year={2024},
      eprint={2404.14951},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
