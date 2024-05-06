import cv2
import numpy as np
import os
from tqdm import tqdm


if __name__ == "__main__":

    processed_data_path = '../data/processed_data'
    training_data_dir = '../data/training_data'
    videos = os.listdir(training_data_dir)

    cropped_dir = '../data/training_data_cropped'

    for video in tqdm(videos):

        frames = os.listdir(os.path.join(training_data_dir, video))
        os.makedirs(os.path.join(cropped_dir, video), exist_ok=True)

        for frame in frames:

            im_path = os.path.join(processed_data_path, video, 'imgs', frame)
            bbox_path = os.path.join(processed_data_path, video, 'bbox_smooth', frame.replace('png', 'npy'))

            # prepare input
            orig_im = cv2.imread(im_path)
            bbox = np.load(bbox_path)
            orig_im = orig_im[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]

            cv2.imwrite(os.path.join(cropped_dir, video, frame), orig_im)