from collections import defaultdict
import glob
import json
from pathlib import Path
import os
import random
import re
import shutil

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression 
from dataset_handler.util.write_annotation_and_delete_files import write_annotation_and_delete_files
from util.read_exr_image import read_exr_image, select_pixels_within_range
from util.velocity_towards_camera import calculate_max_velocity_towards_camera, get_world_velocity_subset
os.environ["OPENCV_IO_ENABLE_OPENEXR"]='1'

import cv2
import os
import pandas as pd
import matplotlib
import numpy as np
from tqdm import tqdm
import argparse
from annotation_handler_hslu import AnnotationHandlerHslu 
from util.get_sementation_contour import get_segmentation_contour
from util.write_movie import write_movie
from vca_blender_config_reader import VcaBlenderConfigReader
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-c','--config_path',default=r"D:\projects\phd_utils\blender_environments\20_09_24_fisheye_person\generated\tests") 
parser.add_argument('-w','--write_movie',default=False) 
parser.add_argument('-d','--delete_annotation',default=True) 
parser.add_argument('--debug',default=True) 
parser.add_argument('--min_contour_area',default=100) 
parser.add_argument('--contour_approx_eps',default=0.8) 
parser.add_argument('--combine_convex',default=False) 
parser.add_argument('--delete_empty_images',default=True) 
parser.add_argument('--min_amount_of_images',default=8) 
parser.add_argument('--output_folder',default='D:\gen4_3_2') 
parser.add_argument('--depth_map',default=r"distances\gen4_distance.exr") 

parser.add_argument('--version', help='version of selection', default='frame', type=str)

parser.add_argument('--door_opening_velocity_max_ms',default=1.5) 
parser.add_argument('--door_opening_acceleration_ms2',default=0.5) 
parser.add_argument('--door_opening_distance_m',default=1.5) 
parser.add_argument('--camera_height_m',default=2) 

parser.add_argument('--max_positive_sampler_per_sequence',default=2) 
parser.add_argument('--entrance_difference_m',default=0.2) 

parser.add_argument('--adjusted_framerate',default=10)
args = parser.parse_args()


# Draw the door into the map - use centerpoint and opening distance m

paths = [{"in":1},
        {"out":0}]

final_data = []

def extend_list(label,path_to_element):
    final_data.append({"id": path_to_element.name,
                        "label": label})



max_velocities = []

def main():
            

    # Search all images
    # check if keypoints and mask are available
    # Get them and store them
    # Add directory in annotation.csv
    available_annotation_files = write_annotation_and_delete_files(args,args.delete_annotation)
                    


if __name__ == '__main__':
    main()