import argparse
import os
from metrics.ccs import get_CCSscores

def parse_args():
    parser = argparse.ArgumentParser(description="SRStitcher.")
    parser.add_argument(
        "--imgpath",
        type=str,
        default="SRStitcherResults",
    )

    parser.add_argument(
        "--inputpath",
        type=str,
        default="examples",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    img_path = args.imgpath
    coarse_img_path = 'coarse'+args.imgpath
    input1_path = os.path.join(args.inputpath,'input1')
    input2_path = os.path.join(args.inputpath,'input2')

    imgs = sorted(os.listdir(img_path))

    for imgname in imgs:
        imgpath = os.path.join(img_path, imgname)
        coarseimgpath = os.path.join(coarse_img_path, imgname)
        input1path = os.path.join(input1_path, imgname)
        input2path = os.path.join(input2_path, imgname)

        socre = get_CCSscores(imgpath, coarseimgpath, input1path, input2path)
        print('CCS score of', imgname,':',socre)
