# bat_recognition



## Project Structure:

    └── bat_recognition/
        ├── data/
        │   ├── raw_data/
        │   │   └── videos/
        │   │       ├── video1.mp4
        │   │       ├── video2.mp4
        │   │       └── ...
        │   └── processed_dataset/
        │       └── video1/
        │           ├── imgs/
        │           ├── bboxes/
        │           ├── bbox_smooth/
        │           └── np_bg/
        └── RMBG-1.4 (see Dataset pre-process section 4)

## Dataset Pre-Process

The dataset is handled using DVC.\
To pull the data, use the command:
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
>python dataset/smooth_bbox.py -v <video_name>

The processed bounding boxes will be saved at `data/processed_data/<video>/bbox_smooth`.

4. To remove the background:
   - First, clone the following repository inside the project folder (source: [RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4).):\
   >  git clone https://huggingface.co/briaai/RMBG-1.4
   - Then run:
>python dataset/remove_background.py -v <video_name>

The processed images will be saved at `data/processed_data/<video>/no_bg`.


Note: If `-v` is not specified, the script will go through all the videos in the folder that have not been processed yet.

## model
https://github.com/chou141253/FGVC-HERBS/tree/master