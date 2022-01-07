import argparse
import os
import numpy as np
import cv2
from landmark_utils import detect_frames_track


def detect_track(input_path, video, use_visualization, visualize_path):
    vidcap = cv2.VideoCapture(os.path.join(input_path, video))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        success, image = vidcap.read()
        if success:
            frames.append(image)
        else:
            break
    raw_data = detect_frames_track(frames, fps, use_visualization, visualize_path, video)

    vidcap.release()
    return np.array(raw_data)


def main():
    
    input_path = "./fake_videos/"
    # input_path = "./true_videos_actor"
    # input_path = "./true_videos_youtube"
    
    output_path = "./fake_videos_txt/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    
    visualize = False
    visualize_path = " "
    """
    Prepare the environment
    """
    i = 0
    videos = os.listdir(input_path)
    for video in videos:
        if video.startswith("."):
            continue
        
        print("Extract landmarks from {}.".format(video))
        raw_data = detect_track(input_path, video, visualize, visualize_path)
        i = i + 1
        i_str = str(i).zfill(5)
        if len(raw_data) == 0:
            print("No face detected", video)
        else:
            np.savetxt(output_path + i_str + ".txt", raw_data, fmt='%1.5f')
        print("Landmarks data saved!" + i_str)
    return


if __name__ == "__main__":
    main()