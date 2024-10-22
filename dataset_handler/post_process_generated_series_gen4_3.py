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

parser.add_argument('-c','--config_path',default=r"D:\gen4_2_1") 
parser.add_argument('-w','--write_movie',default=False) 
parser.add_argument('-d','--delete_annotation',default=False) 
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
            
    print("Loaded Config in PostProcessing")

    
    region_selected_visual_height = np.sqrt(args.camera_height_m**2+args.door_opening_distance_m**2)
    # iterate over all annotation files - create label files
    depth_clipped_to_20m = read_exr_image(args.depth_map,20.0)
    mm,selected_environment = select_pixels_within_range(depth_clipped_to_20m,0,region_selected_visual_height)
    # get max left pixel and max right pixel, project to ground    
    nonzero_indices = np.nonzero(selected_environment)
    x_coordinates = nonzero_indices[0]
    door_left_end = np.min(x_coordinates)
    door_right_end = np.max(x_coordinates)

    # Search all images
    # check if keypoints and mask are available
    # Get them and store them
    # Add directory in annotation.csv
    available_annotation_files = write_annotation_and_delete_files(args)
                    

    
    def extract_frame_number(imageName):
        match = re.search(r'image_(\d+)\.png', imageName)
        if match:
            return int(match.group(1))
        return -1

    def is_point_inside_image(point, h, w):
        x, y = point
        return 0 <= x < w and 0 <= y < h
    
    def write_group(group,label,path):
        for index,group_chunks in enumerate(group):
            origine_path = Path(group_chunks[0].file_path).parent
            origin_name = origine_path.name
            dataset_path = path.joinpath(origin_name+'_'+label+'_'+str(index).zfill(3))
            
            dataset_path.mkdir(exist_ok=True)
            # os.mkdir(dataset_path.as_posix())
            for group_element in group_chunks:
                target_file_path = dataset_path.joinpath(group_element.imageName.split(':')[0])
                # avoid overwriting
                if target_file_path.exists():
                    continue
                origin_file_path = origine_path.joinpath(group_element.imageName.split(':')[0])
                if origin_file_path.exists():
                    shutil.copyfile(origin_file_path.as_posix(),target_file_path.as_posix())
            extend_list(label,dataset_path)
            
    def extrapolate_line(points, ymax):
        """Given a list of points, fit a line and extrapolate to intersect y=ymax."""
        points = np.array(points)
        X = points[:, 0].reshape(-1, 1)  # x-values
        y = points[:, 1]  # y-values
        
        # Fit linear regression model
        model = LinearRegression()
        model.fit(X, y)
        
        # Get line parameters
        slope = model.coef_[0]
        intercept = model.intercept_
        
        if slope == 0:
            return -1
        # Calculate x-coordinate where y = ymax
        x_intersect = (ymax - intercept) / slope
        
        return x_intersect

     
    Path(args.output_folder).mkdir(exist_ok=True)
    # os.mkdir(args.output_folder,exist_ok=True)
    
    
    opening_group = []
    other_group = []
    
    velocities = []
    # Read Annotations. 
    for annotation_file in tqdm(available_annotation_files):
        annotation_handler = AnnotationHandlerHslu()
        # annotation_items = annotation_handler.readAnnotationFile(annotation_file.as_posix())
        annotation_handler.readAnnotations(annotation_file.as_posix())
        annotation_items = annotation_handler.getAnnotationItemsList()
        smaller_list_of_elements = []
        regular_elements_with_id = []
        world_elements_with_id = []
        for annotation_item in annotation_items:
            # find both world and other, save both with its id on the same dictionary
            if type(annotation_item) == AnnotationHandlerHslu.annotationItemPoint and 'walking_root'in annotation_item.label and not 'world' in  annotation_item.label :
                # smaller_list_of_elements.append(annotation_item)
                regular_elements_with_id.append({'id':int(annotation_item.labelId),'annotation':annotation_item})
            if type(annotation_item) == AnnotationHandlerHslu.annotationItemPoint and 'world walking_root'in annotation_item.label :
                # smaller_list_of_elements.append(annotation_item)
                world_elements_with_id.append({'id':int(annotation_item.labelId),'annotation':annotation_item})
        
        
        
        if len(regular_elements_with_id) == 0:
            continue
        # Sort the list based on the frame number
        regular_image = Path(regular_elements_with_id[0]['annotation'].file_path).parent.joinpath(regular_elements_with_id[0]['annotation'].imageName.split(':')[0])
        actual_image = cv2.imread(regular_image.as_posix())
        h,w,c = actual_image.shape
        
        # SORT OUT POINTS THAT ARE NOT IN THE IMAGE, 
        
        # sortedd_annotations = sorted(filtered_annotations, key=lambda x: extract_frame_number(x['annotation'].imageName))
        
        # sort both
        regular_elements_with_id_sorted = sorted(regular_elements_with_id, key=lambda x: extract_frame_number(x['annotation'].imageName))
        world_elements_with_id_sorted = sorted(world_elements_with_id, key=lambda x: extract_frame_number(x['annotation'].imageName))

        filtered_annotations = []
        for a, b in zip(regular_elements_with_id_sorted, world_elements_with_id_sorted):
            if is_point_inside_image(a['annotation'].point, h, w):
                filtered_annotations.append({'id':a['id'],'world':b['annotation'],'normal':a['annotation']})

        
        grouped_annotations = defaultdict(list)

        # Group elements by 'id'
        for item in filtered_annotations:
            grouped_annotations[item['id']].append({'normal':item['normal'],'world':item['world']})

        
        # Iterate over every single person and their detections. 
        for id_, group in grouped_annotations.items():
            # sort group
            sorted_detections = sorted(group, key=lambda x: extract_frame_number(x['world'].imageName))
            
            # measure speed
            world_items = [item['world'] for item in sorted_detections]
            normal_items = [item['normal'] for item in sorted_detections]
            # world velocities in blender coordinates -> meaning y and x are twisted
            # in blender (image) pos y is positive x and (image) positive x is positive y
            velocities_kmh,velocities_x_kmh,velocities_y_kmh,velocities_ms = get_world_velocity_subset(world_items,framerate=args.adjusted_framerate)
            world_positions_x_y = [item['world'].point for item in sorted_detections]
            image_positions_x_y = [item['normal'].point for item in sorted_detections]
            
            
            batch_size = 8
            count_from_same_dataset = 0
            for i in range(1, len(image_positions_x_y) - batch_size + 1, 1):      
                # if count_from_same_dataset == args.max_positive_sampler_per_sequence :
                #     break
                # b for batched
                b_annotation_items = normal_items[i:i+batch_size]
                b_velocities_x_kmh = velocities_x_kmh[i-1 : i-1+batch_size]
                b_velocities_y_kmh = velocities_y_kmh[i-1 : i-1+batch_size]
                b_velocities_kmh = velocities_kmh[i-1 : i-1+batch_size]
                
                b_image_positions_x_y = image_positions_x_y[i:i+batch_size]
                b_world_positions_x_y = world_positions_x_y[i:i+batch_size]
                b_world_positions_y = [item[1] for item in b_world_positions_x_y]
                b_world_positions_x = [item[0] for item in b_world_positions_x_y]
                intersection_point = extrapolate_line(b_image_positions_x_y,h)
                if door_left_end < intersection_point < door_right_end :
                    # check if y velocity is positive
                    velocity_is_positive = np.mean(b_velocities_x_kmh) > 0
                    
                    if not velocity_is_positive:
                        continue
                    # check how far away it is
                    world_mean_x = np.mean(b_world_positions_x)
                    world_mean_y = np.mean(b_world_positions_y)
                    world_front_pos_x = b_world_positions_x[-1]
                    world_front_pos_y = b_world_positions_x[-1]
                    world_x_front_velocity_ms = b_velocities_x_kmh[-1]/3.6
                    world_front_velocity_ms = b_velocities_kmh[-1]/3.6
                    
                    def calculate_distance_to_door(v_person_ms, v_max, a, m):
                        term1 = v_max / (2 * a)
                        term2 = m / v_max
                        distance_to_door = v_person_ms * (term1 + term2)
                        return distance_to_door
                                        
                    person_distance_to_door = np.sqrt((0 - world_front_pos_x)**2 + (0 - world_front_pos_y)**2)
                    distance_to_door = calculate_distance_to_door(world_front_velocity_ms,
                                                                  args.door_opening_velocity_max_ms,
                                                                  args.door_opening_acceleration_ms2,
                                                                  args.door_opening_distance_m)
                    
                    # otherwise too many close exampler are visible
                    if distance_to_door+args.entrance_difference_m >= person_distance_to_door and distance_to_door-args.entrance_difference_m <= person_distance_to_door:   
                        # regular_image = Path(world_items[i].file_path).parent.joinpath(world_items[i].imageName.split(':')[0])
                        # actual_image = cv2.imread(regular_image.as_posix())
                        count_from_same_dataset = count_from_same_dataset + 1
                        opening_group.append(b_annotation_items)
                        velocities.append(np.mean(b_velocities_kmh))
                    else:
                        other_group.append(b_annotation_items)
        print(str(len(other_group)) + ' vs '+ str(len(opening_group)))
    plt.hist(velocities, bins=10, color='blue', edgecolor='black')
    # Add labels and title
    plt.title('Histogram of Provided Data')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # Show plot
    plt.savefig("velocities.png")

    write_group(opening_group,'entering',Path(args.output_folder))
    # reduce size
    other_group = random.sample(other_group, len(opening_group))
    write_group(other_group,'none',Path(args.output_folder))

    # only require walking_root
    random.shuffle(final_data)
    
    # separate by args.version
    label_to_data = {}
    for item in final_data:
        label = item['label']
        if label not in label_to_data:
            label_to_data[label] = []
        label_to_data[label].append(item)
    min_count = min(len(items) for items in label_to_data.values())
        
    balanced_data = []
    for label, items in label_to_data.items():
        sampled_items = np.random.choice(items, min_count, replace=False)
        balanced_data.extend(sampled_items)
        
    train_data, val_data, test_data = [], [], []
    for label in label_to_data.keys():
        label_items = [item for item in balanced_data if item['label'] == label]

        # Split into train+val and test
        train_val_items, test_items = train_test_split(label_items, test_size=0.2, random_state=42, stratify=[item['label'] for item in label_items])
        
        # Split train_val_items into train and val
        train_items, val_items = train_test_split(train_val_items, test_size=0.25, random_state=42, stratify=[item['label'] for item in train_val_items])  # 0.25 * 0.8 = 0.2

        train_data.extend(train_items)
        val_data.extend(val_items)
        test_data.extend(test_items)

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    def save_to_json(data, filename):
        with open(filename, 'w') as f:
            json.dump({"Data": data}, f, indent=4)

    save_to_json(train_data, Path(args.output_folder).joinpath('annotations_train_{}.json'.format(args.version)))
    save_to_json(val_data, Path(args.output_folder).joinpath('annotations_validation_{}.json'.format(args.version)))
    save_to_json(test_data, Path(args.output_folder).joinpath('annotations_test_{}.json'.format(args.version)))


if __name__ == '__main__':
    main()