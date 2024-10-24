# Real-Time Image Stitching with SIFT and FLANN

This project provides a solution for stitching multiple images together to create panoramic views. Using **SIFT** (Scale-Invariant Feature Transform) for keypoint detection and **FLANN** (Fast Library for Approximate Nearest Neighbors) for matching, this tool allows users to automatically detect feature points and combine multiple images into one cohesive panorama.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Algorithms Overview](#algorithms-overview)
  - [SIFT (Scale-Invariant Feature Transform)](#sift)
  - [FLANN (Fast Library for Approximate Nearest Neighbors)](#flann)
- [Installation](#installation)
- [Usage](#usage)
  - [Image Folder Structure](#image-folder-structure)
  - [Running the Stitching Functions](#running-the-stitching-functions)
- [File Structure](#file-structure)
- [Performance Considerations](#performance-considerations)
- [Examples](#examples)
  - [Stitching Shanghai Images](#stitching-shanghai-images)
  - [Stitching Street Images](#stitching-street-images)
  - [Stitching Grail Images](#stitching-grail-images)
- [Customization](#customization)


## Introduction
Image stitching refers to the process of combining multiple overlapping images to produce a seamless panorama or a composite image. This is commonly used in applications like panoramic photography, satellite image stitching, and computer vision tasks requiring a wide field of view. 

This project is built using Python’s OpenCV library, utilizing advanced algorithms such as SIFT for feature detection and FLANN for efficient matching of those features between images. The system can stitch images from various directories, automatically aligning and blending them to create high-quality panoramic outputs.

## Features
- **Automated Keypoint Detection**: Automatically identifies and detects key points (corners, edges, and patterns) in images that remain consistent regardless of scaling and rotation.
- **Image Matching with FLANN**: Efficiently matches features between images using FLANN, allowing for fast and robust alignment of images.
- **Panorama Creation**: Combines images into a seamless panorama with controlled cropping and adjustment to remove any unaligned edges.
- **Cropping Functionality**: Allows users to specify cropping dimensions to ensure the final stitched image is clean and optimized for viewing.
- **Supports Multiple Sets of Images**: Functions for stitching different sets of images, such as "shanghai," "street," "building," and "grail" image sets.

## Technologies Used
- **Python 3.x**: The main programming language used for the project.
- **OpenCV**: A powerful open-source computer vision library that provides tools for image processing, feature detection, and stitching.
- **NumPy**: For efficient array operations and mathematical calculations.
  
## Algorithms Overview

### SIFT (Scale-Invariant Feature Transform)
**SIFT** is an algorithm used to detect and describe local features in images. SIFT extracts keypoints and descriptors that are invariant to scaling, rotation, and lighting conditions. This makes it particularly well-suited for image stitching tasks, as it can accurately match corresponding points between images, even if the images have different orientations or scales.

### FLANN (Fast Library for Approximate Nearest Neighbors)
**FLANN** is a fast and efficient library used to perform approximate nearest neighbor searches. In this project, FLANN is used to match SIFT descriptors between images. This matching process identifies corresponding points in overlapping image regions, which are crucial for aligning and stitching the images together.

## Installation
Before using the project, ensure that Python 3.x and the necessary libraries are installed.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/HoangLam2211/CPV_Project
   cd real-time-image-stitching
