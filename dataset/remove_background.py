from skimage import io
import torch, os
from PIL import Image
import sys
import numpy as np
import os
import argparse
from tqdm import tqdm

sys.path.append('../RMBG-1.4')
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image



def remove_bg_from_image(img_name):

    im_path = f"{imgs_dir}/{img_name}"
    bbox_path = im_path.replace("imgs", "bbox_smooth").replace(".png", ".npy")

    net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # prepare input
    model_input_size = [1024, 1024]
    orig_im = io.imread(im_path)
    bbox = np.load(bbox_path)
    orig_im = orig_im[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
    orig_im_size = orig_im.shape[0:2]
    image = preprocess_image(orig_im, model_input_size).to(device)

    # inference
    result = net(image)

    # post process
    result_image = postprocess_image(result[0][0], orig_im_size)

    # save result
    pil_im = Image.fromarray(result_image)
    no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
    # orig_image = Image.open(im_path)
    no_bg_image.paste(Image.fromarray(orig_im), mask=pil_im)
    no_bg_image.save(f"{no_bg_imgs_dir}/{img_name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="path to input video file")
    args = vars(ap.parse_args())

    data_dir = '../data/processed_data'

    if args["video"] is not None:
        videos = [args["video"]]
    else:
        videos = os.listdir(data_dir)

    for video in videos:

        imgs_dir = f'{data_dir}/{video}/imgs'
        no_bg_imgs_dir = f'{data_dir}/{video}/no_bg'

        if not os.path.exists(no_bg_imgs_dir):
            os.makedirs(no_bg_imgs_dir, exist_ok=True)

            imgs = sorted(os.listdir(imgs_dir), key=lambda x: int(x.split('.')[0].split('_')[-1]))
            for img_name in tqdm(imgs, desc=video):
                remove_bg_from_image(img_name)