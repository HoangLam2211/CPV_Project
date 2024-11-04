# Image and Video Stitching with OpenCV

This project provides scripts for stitching multiple images into a panoramic image and for stitching frames from two video streams in real-time. By leveraging OpenCV's feature detection, the program aligns and stitches images, creating seamless panoramas from both static images and live video streams.

## Features
- **Multi-Image Stitching**: Creates a panoramic image by stitching together multiple static images.
- **Real-Time Video Stitching**: Stitches frames from two video streams in real-time to generate a dynamic panorama.
- **Motion Detection**: Detects movement within stitched frames in real-time using a basic motion detector.
- **Timestamp Overlay**: Displays the current timestamp on the real-time stitched frames.

## Requirements
- Python 3.x
- OpenCV
- imutils
- NumPy

You can install the necessary libraries with the following command:
```bash
pip install opencv-python imutils numpy
```
## Project Files
- **`multi_image_stitching.py`**: Script for stitching multiple static images.
- **`realtime_stitching.py`**: Script for stitching frames from two live video streams.
- **Custom modules in `pyimagesearch`**:
  - **`Stitcher`**: Class for handling image stitching.
  - **`BasicMotionDetector`**: Class for motion detection within frames.


**Usage**

**1. Multi-Image Stitching**
This script stitches multiple overlapping images into a single panoramic image.

**Usage**:
```bash
python multi_image_stitching.py --images path/to/images
```
**Parameters:**
- **`--images`**: Directory containing images to be stitched (ensure sufficient overlap between images).

**Workflow**:

**`1. Load Images`**: Reads and processes images from the specified directory.

**`2. Stitch Images`**: Uses the Stitcher class to align and merge images based on detected features and homography estimation.

**`3. Display or Save Result`**: Shows the final stitched panorama, or saves it if specified.

**Error Handling**:
- **`If images lack sufficient overlap, the stitching may fail or yield poor results.`**

**2. Real-Time Video Stitching**

This script stitches frames from two video files in real-time to produce a live panorama.

Usage:
```bash
python realtime_stitching.py
```
**Parameters in realtime_stitching.py:**
- **`rightStream`**: Path to the right camera video file (e.g., "la_202_rs.mp4")
- **`leftStream`**: Path to the left camera video file (e.g., "la_202_ls.mp4").

**Workflow:**

**1. Initialize Video Streams:**  Loads video files and prepares frames.

**2. Frame Stitching:** Uses the Stitcher class to merge left and right frames into a panoramic frame.

**3. Motion Detection:** Tracks motion in the stitched frame using the BasicMotionDetector class.

**4. Display with Timestamp:** Shows stitched frame with an overlayed timestamp, along with individual left and right frames.

**Exit:** Press q to exit the real-time stitching view.

**Error Handling:**
- **`If homography cannot be computed, the script will display a message and terminate.`**

**Examples**

**1. Multi-Image Stitching:**

- **`Run multi_image_stitching.py to create a panorama from a directory of overlapping images.`**
- 
**2. Real-time Video Stitching:**
  
- **`Use realtime_stitching.py to generate a live panorama from two video feeds.`**

**Troubleshooting**
- **`Insufficient Overlap: For multi-image stitching, ensure images have enough overlap.`**
- **`Low Frame Rate: For real-time stitching, resize frames to reduce processing time.`**
- **`Stitching Failure: If homography fails, check that the videos/images have strong feature points.`**

