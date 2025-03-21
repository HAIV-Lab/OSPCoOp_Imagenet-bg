import torch
import sys
import argparse
import numpy as np
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point
import os
import cv2
import time
def setup_args(parser):
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )

    parser.add_argument(
        "--min", type=float, required=True,
        help="The path to the lama checkpoint.",
    )

    parser.add_argument(
        "--max", type=float, required=True,
        help="The path to the lama checkpoint.",
    )

def get_image_paths(root_folder):
    image_paths = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(subdir, file))
    return image_paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_file = os.path.join("classnames.txt")
    root = args.output_dir
    folder_path = root + "/mask/"
    image_files = get_image_paths(folder_path)
    kernel = np.ones((12,12), np.uint8)


    for imgname in image_files:
        # 记录开始时间
        start_time = time.time()
        imgname = imgname.split('/')[-1]
        mask = load_img_to_array(os.path.join(root + "/mask/",imgname))

        img = load_img_to_array(os.path.join(root + "/raw/", imgname[0:-4]+'.JPEG'))
        mask1 = mask[:,:,0]

        mask_d = cv2.dilate(mask1, kernel, iterations=1)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=0)
            img = np.transpose(img, (1, 2, 0))

        img_inpainted_d = inpaint_img_with_lama(
            img, mask_d, args.lama_config, args.lama_ckpt, device=device)

        mask_d = np.stack((mask_d,) * 3, axis=-1)
        horizontal_concat = np.concatenate((img, img_inpainted_d, mask_d), axis=1)

        img_inpainted_p = os.path.join(root +'/inpaint/',
                                       imgname)
        save_array_to_img(img_inpainted_d, img_inpainted_p)

        visual = os.path.join(root +"/visualization/",
                                       imgname)
        save_array_to_img(horizontal_concat, visual)
        end_time = time.time()

        # 计算运行时间
        elapsed_time = end_time - start_time
        print(elapsed_time)
