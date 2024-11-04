import cv2
import numpy as np

def create_panorama(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return None

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for consistent processing
        frame = cv2.resize(frame, (480, 270))  

        # Store every 10th frame for stitching
        if frame_count % 10 == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()

    # Stitch frames together
    if len(frames) > 1:
        stitcher = cv2.Stitcher_create()
        status, pano = stitcher.stitch(frames)

        if status == cv2.STITCHER_OK:
            return pano
        else:
            print("Panorama stitching failed.")
            return None
    else:
        print("Not enough frames for stitching.")
        return None

# Example usage:
video_path = "test3.mp4" 
panorama = create_panorama(video_path)

if panorama is not None:
    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("panorama.jpg", panorama)
else:
    print("No panorama created.")