import os
import cv2
from tqdm import tqdm


raw_date_dir = f'../data/raw_data'
processed_data_dir = '../data/processed_data/'

videos = os.listdir('../data/raw_data/videos')
videos = [v for v in videos if v.endswith('.mp4')]

for video in tqdm(videos):

    # create a folder to save the pngs
    os.makedirs(os.path.join(processed_data_dir, video.split('.')[0]), exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(os.path.join(raw_date_dir, 'videos', video))

    # Initialize frame count
    frame_count = 0

    # Read until video is completed
    while video_capture.isOpened():
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % 8 == 0:
            # Save frame as PNG
            cv2.imwrite(os.path.join(processed_data_dir, video.split('.')[0], f'{video.split(".")[0]}_frame_{frame_count:04d}.png'), frame)

        # Increment frame count
        frame_count += 1

    # Release the video capture object
    video_capture.release()
