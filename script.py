# Operating System module for interacting with the operating system
import os

# Module for generating random numbers
import random

# Module for numerical operations
import numpy as np

# OpenCV library for image processing
import cv2

# Python Imaging Library for image processing
from PIL import Image, ImageDraw, ImageFont

# PyTorch library for deep learning
import torch

# Dataset class for creating custom datasets in PyTorch
from torch.utils.data import Dataset

# Module for image transformations
import torchvision.transforms as transforms

# Neural network module in PyTorch
import torch.nn as nn

# Optimization algorithms in PyTorch
import torch.optim as optim

# Function for padding sequences in PyTorch
from torch.nn.utils.rnn import pad_sequence

# Function for saving images in PyTorch
from torchvision.utils import save_image

# Module for plotting graphs and images
import matplotlib.pyplot as plt

# Module for displaying rich content in IPython environments
from IPython.display import clear_output, display, HTML

# Module for encoding and decoding binary data to text
import base64

# Create a directory named 'training_dataset'
os.makedirs('training_dataset', exist_ok=True)

# Define the number of videos to generate for the dataset
num_videos = 10000

# Define the number of frames per video (1 Second Video)
frames_per_video = 10

# Define the size of each image in the dataset
img_size = (64, 64)

# Define the size of the shapes (Circle)
shape_size = 10

# Define text prompts and corresponding movements for circles
prompts_and_movements = [
    ("circle moving down", "circle", "down"),  # Move circle downward
    ("circle moving left", "circle", "left"),  # Move circle leftward
    ("circle moving right", "circle", "right"),  # Move circle rightward
    ("circle moving diagonally up-right", "circle", "diagonal_up_right"),  # Move circle diagonally up-right
    ("circle moving diagonally down-left", "circle", "diagonal_down_left"),  # Move circle diagonally down-left
    ("circle moving diagonally up-left", "circle", "diagonal_up_left"),  # Move circle diagonally up-left
    ("circle moving diagonally down-right", "circle", "diagonal_down_right"),  # Move circle diagonally down-right
    ("circle rotating clockwise", "circle", "rotate_clockwise"),  # Rotate circle clockwise
    ("circle rotating counter-clockwise", "circle", "rotate_counter_clockwise"),  # Rotate circle counter-clockwise
    ("circle shrinking", "circle", "shrink"),  # Shrink circle
    ("circle expanding", "circle", "expand"),  # Expand circle
    ("circle bouncing vertically", "circle", "bounce_vertical"),  # Bounce circle vertically
    ("circle bouncing horizontally", "circle", "bounce_horizontal"),  # Bounce circle horizontally
    ("circle zigzagging vertically", "circle", "zigzag_vertical"),  # Zigzag circle vertically
    ("circle zigzagging horizontally", "circle", "zigzag_horizontal"),  # Zigzag circle horizontally
    ("circle moving up-left", "circle", "up_left"),  # Move circle up-left
    ("circle moving down-right", "circle", "down_right"),  # Move circle down-right
    ("circle moving down-left", "circle", "down_left"),  # Move circle down-left
]
We’ve defined several movements of our circle using these prompts. Now, we need to code some mathematical equations to move that circle based on the prompts.

# Define function with parameters
def create_image_with_moving_shape(size, frame_num, shape, direction):
  
    # Create a new RGB image with specified size and white background
    img = Image.new('RGB', size, color=(255, 255, 255))  

    # Create a drawing context for the image
    draw = ImageDraw.Draw(img)  

    # Calculate the center coordinates of the image
    center_x, center_y = size[0] // 2, size[1] // 2  

    # Initialize position with center for all movements
    position = (center_x, center_y)  

    # Define a dictionary mapping directions to their respective position adjustments or image transformations
    direction_map = {  
        # Adjust position downwards based on frame number
        "down": (0, frame_num * 5 % size[1]),  
        # Adjust position to the left based on frame number
        "left": (-frame_num * 5 % size[0], 0),  
        # Adjust position to the right based on frame number
        "right": (frame_num * 5 % size[0], 0),  
        # Adjust position diagonally up and to the right
        "diagonal_up_right": (frame_num * 5 % size[0], -frame_num * 5 % size[1]),  
        # Adjust position diagonally down and to the left
        "diagonal_down_left": (-frame_num * 5 % size[0], frame_num * 5 % size[1]),  
        # Adjust position diagonally up and to the left
        "diagonal_up_left": (-frame_num * 5 % size[0], -frame_num * 5 % size[1]),  
        # Adjust position diagonally down and to the right
        "diagonal_down_right": (frame_num * 5 % size[0], frame_num * 5 % size[1]),  
        # Rotate the image clockwise based on frame number
        "rotate_clockwise": img.rotate(frame_num * 10 % 360, center=(center_x, center_y), fillcolor=(255, 255, 255)),  
        # Rotate the image counter-clockwise based on frame number
        "rotate_counter_clockwise": img.rotate(-frame_num * 10 % 360, center=(center_x, center_y), fillcolor=(255, 255, 255)),  
        # Adjust position for a bouncing effect vertically
        "bounce_vertical": (0, center_y - abs(frame_num * 5 % size[1] - center_y)),  
        # Adjust position for a bouncing effect horizontally
        "bounce_horizontal": (center_x - abs(frame_num * 5 % size[0] - center_x), 0),  
        # Adjust position for a zigzag effect vertically
        "zigzag_vertical": (0, center_y - frame_num * 5 % size[1]) if frame_num % 2 == 0 else (0, center_y + frame_num * 5 % size[1]),  
        # Adjust position for a zigzag effect horizontally
        "zigzag_horizontal": (center_x - frame_num * 5 % size[0], center_y) if frame_num % 2 == 0 else (center_x + frame_num * 5 % size[0], center_y),  
        # Adjust position upwards and to the right based on frame number
        "up_right": (frame_num * 5 % size[0], -frame_num * 5 % size[1]),  
        # Adjust position upwards and to the left based on frame number
        "up_left": (-frame_num * 5 % size[0], -frame_num * 5 % size[1]),  
        # Adjust position downwards and to the right based on frame number
        "down_right": (frame_num * 5 % size[0], frame_num * 5 % size[1]),  
        # Adjust position downwards and to the left based on frame number
        "down_left": (-frame_num * 5 % size[0], frame_num * 5 % size[1])  
    }

    # Check if direction is in the direction map
    if direction in direction_map:  
        # Check if the direction maps to a position adjustment
        if isinstance(direction_map[direction], tuple):  
            # Update position based on the adjustment
            position = tuple(np.add(position, direction_map[direction]))  
        else:  # If the direction maps to an image transformation
            # Update the image based on the transformation
            img = direction_map[direction]  

    # Return the image as a numpy array
    return np.array(img)
The function above is used to move our circle for each frame based on the selected direction. We just need to run a loop on top of it up to the number of videos times to generate all videos.

# Iterate over the number of videos to generate
for i in range(num_videos):
    # Randomly choose a prompt and movement from the predefined list
    prompt, shape, direction = random.choice(prompts_and_movements)
    
    # Create a directory for the current video
    video_dir = f'training_dataset/video_{i}'
    os.makedirs(video_dir, exist_ok=True)
    
    # Write the chosen prompt to a text file in the video directory
    with open(f'{video_dir}/prompt.txt', 'w') as f:
        f.write(prompt)
    
    # Generate frames for the current video
    for frame_num in range(frames_per_video):
        # Create an image with a moving shape based on the current frame number, shape, and direction
        img = create_image_with_moving_shape(img_size, frame_num, shape, direction)
        
        # Save the generated image as a PNG file in the video directory
        cv2.imwrite(f'{video_dir}/frame_{frame_num}.png', img)
