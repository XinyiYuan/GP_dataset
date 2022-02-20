import argparse
import os
import numpy as np
import cv2
from landmark_utils import detect_frames_track
import shutil


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
    visualize = False
    visualize_path = " "
    
    video_path = ["./real", "./fake"]
    
    for path in video_path:
        input_path = path + "/"
        output_path = path + "_calib_txt/"
        
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        videos = os.listdir(input_path)
        for video in videos:
            if video.startswith("."):
                continue
            
            txt_name = video.split('.')[0] + ".txt"
            
            if not os.path.exists(output_path + txt_name):
                print("Extract landmarks from {}.".format(video))
                
                raw_data = detect_track(input_path, video, visualize, visualize_path)
                if len(raw_data) == 0:
                    print("No face detected", video)
                else:
                    np.savetxt(output_path + txt_name, raw_data, fmt='%1.5f')
                    print("Landmarks data saved!", video)
    return


if __name__ == "__main__":
    main()