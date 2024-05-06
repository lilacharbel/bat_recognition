import cv2
import os


def png_to_video(input_folder, output_video, fps=30):
    # Get all the png files from the input folder
    img_array = []
    files = sorted(os.listdir(input_folder), key=lambda x: int(x.split('.')[0]))
    if not files:
        print("No PNG files found in the directory.")
        return

    # Read the first image to set the frame size
    first_image_path = os.path.join(input_folder, files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"Failed to load the first image: {first_image_path}")
        return
    height, width, layers = first_image.shape
    size = (width, height)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, size)

    # Add frames to the video
    for filename in files:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        if img.shape[0] != height or img.shape[1] != width:
            print(f"Skipping {filename}: size mismatch.")
            continue
        video.write(img)
    video.release()

    print("Video created successfully.")


folder = '../../data/figures/20230805_135856/smoothed_bbox'
output = '../../data/figures/20230805_135856/bbox_vid.mp4'
# Example usage
png_to_video(folder, output, fps=15)
