# bat_recognition

## Dataset pre-process

The dataset is handled using DVC to pull the raw and processed data. To pull the data, use the command:
`dvc pull data.dvc`

1. Save raw videos at `data/raw/data/videos`.
2. To create a bounding box (bbox) around the bat, run:
>python dataset/opencv_object_tracker.py -v <video_name> -i <frames_save_interval>
   - Select an initial bbox.
   - If you want to adjust the bbox during the video, press "s", and choose a new bbox.

The images will be saved at `data/processed_data/<video>/imgs`, and the bounding boxes will be saved at `data/processed_data/<video>/bboxes`.
Note: Bounding box coordinates are represented as `[x, y, w, h]`.\
To crop the bat from the image: `crop = img[y: y + h, x: x + w, :]`.

3. To smooth the bounding box size and `[x, y]` location, run:\
>python dataset/smooth_bbox.p -v <video_name>

The processed bounding boxes will be saved at `data/processed_data/<video>/bbox_smooth`.

4. To remove the background, run:\
`dataset/remove_background.py -v <video>`

Note: If `-v` is not specified, the script will go through all the videos in the folder that have not been processed yet.
