import os

import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

""" bbox = [x, y, x + w, y + h] """


def moving_average(signal, window_size):
    """Calculate the moving average of a signal."""
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')


def get_center_and_size(date_dir):
    centers = []
    sizes = []
    frames = sorted(os.listdir(date_dir), key=lambda x: int(x.split('.')[0].split('_')[-1]))

    for box_name in frames:
        bbox = np.load(os.path.join(date_dir, box_name))

        # square the bbox
        w, h = bbox[2], bbox[3]

        x_center, y_center = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
        size = max(h, w) # * 1.2
        x_min, y_min = max(x_center - (size//2), 0), max(y_center - (size//2), 0)
        s_bbox = [x_min, y_min, size, size]

        centers.append(np.array([s_bbox[0] + (s_bbox[2] // 2), s_bbox[1] + (s_bbox[3] // 2)]))
        sizes.append(size)

    return frames, np.array(centers), np.array(sizes)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="path to input video file")
    args = vars(ap.parse_args())

    if args["video"] is not None:
        videos = [args["video"]]
    else:
        videos = os.listdir('../data/processed_data')

    for video in videos:
        data_dir = f'../data/processed_data/{video}'

        if os.path.exists(f'{data_dir}/bbox_smooth'):
            print(f'skipping {video}')

        else:
            imgs_dir = os.path.join(data_dir, 'imgs')
            bbox_dir = os.path.join(data_dir, 'bboxes')
            new_data_dir = os.path.join(data_dir, 'bbox_smooth')
            os.makedirs(new_data_dir, exist_ok=True)

            frames, centers, sizes = get_center_and_size(bbox_dir)

            # Smooth the signal using moving average
            smoothed_sizes = moving_average(sizes, window_size=10)
            smoothed_x = moving_average(centers[:, 0], window_size=10)
            smoothed_y = moving_average(centers[:, 1], window_size=10)

            plt.figure()
            plt.plot(sizes, '-x')
            plt.plot(smoothed_sizes, '-o')
            plt.title('sizes')

            plt.figure()
            plt.plot(centers[:, 0], '-x')
            plt.plot(smoothed_x, '-o')
            plt.title('x center')

            plt.figure()
            plt.plot(centers[:, 1], '-x')
            plt.plot(smoothed_y, '-o')
            plt.title('y center')

            # plt.show()
            plt.close('all')

            for i in range(len(smoothed_sizes)):
                c_x, c_y, s = smoothed_x[i], smoothed_y[i], smoothed_sizes[i]
                new_bbox = [int(c_x - (s//2)), int(c_y - (s//2)), int(s), int(s)]
                np.save(os.path.join(new_data_dir, frames[i]), new_bbox)


            # plot frames with bbox on them
            images = sorted(os.listdir(imgs_dir), key=lambda x: int(x.split('.')[0].split('_')[-1]))
            for frame in images:

                if os.path.isfile(os.path.join(new_data_dir, frame.replace(".png", ".npy"))):
                    bbox_org = np.load(os.path.join(bbox_dir, frame.replace(".png", ".npy")))
                    bbox_new = np.load(os.path.join(new_data_dir, frame.replace(".png", ".npy")))
                    x, y, w, h = bbox_new

                    img = cv2.imread(os.path.join('../data/processed_data', video, 'imgs', frame))
                    img = cv2.rectangle(img, (bbox_org[0], bbox_org[1]), (bbox_org[0] + bbox_org[2], bbox_org[1] + bbox_org[3]), (0, 255, 0), 5)
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
                    img = cv2.resize(img, (img.shape[1]//2, img.shape[0  ]//2))
                    cv2.imshow("Frame", img)
                    cv2.waitKey(1)