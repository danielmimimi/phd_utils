import glob
import json
from pathlib import Path
import os
import random
import re
import shutil 
from util.read_exr_image import read_exr_image, select_pixels_within_range
from util.velocity_towards_camera import calculate_max_velocity_towards_camera
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

parser.add_argument('-c','--config_path',default=r"D:\gen4_2") 
parser.add_argument('-w','--write_movie',default=True) 
parser.add_argument('-d','--delete_annotation',default=False) 
parser.add_argument('--debug',default=True) 
parser.add_argument('--min_contour_area',default=100) 
parser.add_argument('--contour_approx_eps',default=0.8) 
parser.add_argument('--combine_convex',default=False) 
parser.add_argument('--delete_empty_images',default=True) 
parser.add_argument('--min_amount_of_images',default=8) 
parser.add_argument('--depth_map',default=r"distances\gen4_distance.exr") 
parser.add_argument('--door_opening_distance_to_camera',default=3) 
parser.add_argument('--version', help='version of selection', default='frame', type=str)
args = parser.parse_args()



paths = [{"in":1},
        {"out":0}]

final_data = []

def extend_list(label,path_to_element):
    final_data.append({"id": path_to_element.name,
                        "label": label})



max_velocities = []

def main():
            
    print("Loaded Config in PostProcessing")
    args_config_path = Path(args.config_path)
    if not args_config_path.exists():
        raise Exception("Not Exists {}".format(args.config_path))
    dataset_folders = os.listdir(args_config_path.as_posix())


    csv_template = "pose_frame_{}.csv"
    mask_template = "mask_p{}_{}.jpg"

    # Search all images
    # check if keypoints and mask are available
    # Get them and store them
    # Add directory in annotation.csv
    mask_exists = False

    available_annotation_files = []

    for folder in tqdm(dataset_folders):
        
        image_path = args_config_path.joinpath(folder)
        if not image_path.is_dir():
            continue
        
        print(folder)
        if image_path.joinpath("annotation.csv").exists():
            available_annotation_files.append(image_path.joinpath("annotation.csv"))
            continue

        image_paths = glob.glob(str(image_path.joinpath("image_*.png")))
        keypoints_csv = []
        annotations = []
        mask_paths = []
        for image_dir in image_paths:
            # get name 
            pathed = Path(image_dir)
            image_number = pathed.name.split("_")[-1].split(".")[0]
            
            csv_file_path = image_path.joinpath(csv_template.format(image_number))
            if csv_file_path.exists() and csv_file_path.stat().st_size != 0:
                keypoints_csv.append(csv_file_path)
                keypoints = pd.read_csv(csv_file_path.as_posix())
                for person_id in keypoints.id.unique():
                    person_mask_path = image_path.joinpath(mask_template.format(person_id,image_number))
                    if person_mask_path.exists():
                        mask_exists = True
                        mask = cv2.imread(person_mask_path.as_posix())
                        contour = get_segmentation_contour(mask,args)
                        if contour is not None:
                            approx_contour = cv2.approxPolyDP(contour,args.contour_approx_eps, True)
                            mask_paths.append(person_mask_path)
                        else:
                            mask_exists = False
                    else:
                        mask_exists = False
                    current_keypoints = keypoints[keypoints.id == person_id]
                    current_keypoints['limb'] = current_keypoints['limb'].str.split('_').str[-2:].apply(lambda x: '_'.join(x))
                    current_keypoints = current_keypoints.drop(columns=['d_cam'])
                    current_keypoints = current_keypoints.drop(columns=['id'])
                    current_keypoints.reset_index(drop=True, inplace=True)
                    if mask_exists:
                        annotations.append({'mask_exists':True,'Id':person_id,'Contour':contour,'Approx':approx_contour,'Frame':pathed.as_posix(),"Keypoints":current_keypoints.to_dict()})
                    else:
                        annotations.append({'mask_exists':False,'Id':person_id,'Frame':pathed.as_posix(),"Keypoints":current_keypoints.to_dict()})


        annotations_pd = pd.DataFrame(annotations)
        if 'Frame' in annotations_pd.columns:
            frame_grouped = annotations_pd.groupby(by='Frame')

            frameSize = (640, 480)
            if args.write_movie:
                write_movie(frame_grouped,folder,frameSize)
                
            if True:
                annotation_handler = AnnotationHandlerHslu()
                for group_name,group in frame_grouped:
                    image = cv2.imread(group_name)
                    count_per_frame = len(group)
                    for index, row in group.iterrows():
                        for cx,cy,name,visible in zip(row['Keypoints']['x'].values(),row['Keypoints']['y'].values(),row['Keypoints']['limb'].values(),row['Keypoints']['visible'].values()):
                            frame_name = Path(row['Frame']).name
                            annotation_handler.addPoint(frame_name,point=[int(cx),int(cy)],label="{} - {}".format(name, visible),label_id=row['Id'])
                        for cx,cy,name,visible in zip(row['Keypoints']['wx'].values(),row['Keypoints']['wy'].values(),row['Keypoints']['limb'].values(),row['Keypoints']['visible'].values()):
                            frame_name = Path(row['Frame']).name
                            annotation_handler.addPoint(frame_name,point=[cx,cy],label="world {} - {}".format(name, visible),label_id=row['Id'])
                        if row['mask_exists']:
                            frame_name = Path(row['Frame']).name
                            annotation_handler.addPolygon(frame_name,polygon=row['Approx'].ravel().tolist(),label='mask',label_id=row['Id'])
                    # annotation_handler.addMetadata(group_name,"FrameCount",str(count_per_frame))
                annotation_handler.writeAnnotation(image_path.joinpath("annotation.csv"))
                
            if image_path.joinpath("annotation.csv").exists():
                available_annotation_files.append(image_path.joinpath("annotation.csv"))
                if False: # DONT DELETE 
                    for csv in keypoints_csv:
                        os.remove(csv.as_posix())
                    for mask_path in mask_paths:
                        os.remove(mask_path.as_posix())
                    
    # iterate over all annotation files - create label files
    depth_clipped_to_20m = read_exr_image(args.depth_map,20.0)
    mm,selected_environment = select_pixels_within_range(depth_clipped_to_20m,0,args.door_opening_distance_to_camera)
    
    
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
            origine_path = Path(group_chunks[0][1].iloc[0].annotation.file_path).parent
            origin_name = origine_path.name
            dataset_path = path.joinpath(origin_name+'_'+label+'_'+str(index).zfill(3))
            
            dataset_path.mkdir(exist_ok=True)
            # os.mkdir(dataset_path.as_posix())
            for group_element in group_chunks:
                target_file_path = dataset_path.joinpath(group_element[0].split(':')[0])
                # avoid overwriting
                if target_file_path.exists():
                    continue
                origin_file_path = origine_path.joinpath(group_element[0].split(':')[0])
                if origin_file_path.exists():
                    shutil.copyfile(origin_file_path.as_posix(),target_file_path.as_posix())
            extend_list(label,dataset_path)

    output_foder = 'D:\gen4_21'
    
    Path(output_foder).mkdir(exist_ok=True)
    # os.mkdir(output_foder,exist_ok=True)
    
    
    # Read Annotations. 
    # Check each footpoint if it is consisting in the region
    # if not, put into not folder groups of 8 - label as outside
    # if yes, put 7 of NO and XXX of YES into crossing folder, label as crossing in
    # if yes first and then X NO, into crossing folder, label as crossing out
    # if all 8 are yes, label as inside
    # if yes, put into yes folder, all the prev 8
    # print(annotation_file)
    for annotation_file in tqdm(available_annotation_files):
        annotation_handler = AnnotationHandlerHslu()
        # annotation_items = annotation_handler.readAnnotationFile(annotation_file.as_posix())
        annotation_handler.readAnnotations(annotation_file.as_posix())
        annotation_items = annotation_handler.getAnnotationItemsList()
        smaller_list_of_elements = []
        smaller_list_of_elements_with_id = []
        for annotation_item in annotation_items:
            if type(annotation_item) == AnnotationHandlerHslu.annotationItemPoint and 'walking_root'in annotation_item.label and 'world' not in annotation_item.label:
                smaller_list_of_elements.append(annotation_item)
                smaller_list_of_elements_with_id.append({'id':int(annotation_item.labelId),'annotation':annotation_item})
        # Sort the list based on the frame number
        regular_image = Path(smaller_list_of_elements_with_id[0]['annotation'].file_path).parent.joinpath(smaller_list_of_elements_with_id[0]['annotation'].imageName.split(':')[0])
        h,w,c = cv2.imread(regular_image.as_posix()).shape
        
        # SORT OUT POINTS THAT ARE NOT IN THE IMAGE, 
        filtered_annotations = [a for a in smaller_list_of_elements_with_id if is_point_inside_image(a['annotation'].point, h, w)]
        sortedd_annotations = sorted(filtered_annotations, key=lambda x: extract_frame_number(x['annotation'].imageName))


        group_1 = []
        group_2 = []
        group_3 = []
        group_4 = []
        

        
        
        def create_debug_image(image,mask):
            mask_color = (0, 0, 255)  # Red
            # Create a color mask
            color_mask = np.zeros_like(image)
            color_mask[mask > 0] = mask_color
            alpha = 0.5  # Transparency factor
            overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
            return overlay
        
        chunk_size = 8
        df = pd.DataFrame(sortedd_annotations)
        df['imageName'] = df['annotation'].apply(lambda x: x.imageName)
        groups = df.groupby('imageName')
        
        
        def check_image_sequence(data):
            # Extract the starting number from the first entry
            first_image_name = data[0][0]  # e.g., 'image_00002.png:rot0'
            
            # Ensure the format is correct
            if not first_image_name.startswith('image_') or not first_image_name.endswith(':rot0'):
                return False

            # Extract the starting number (e.g., 2 from 'image_00002.png:rot0')
            try:
                starting_number = int(first_image_name.split('_')[1].split('.')[0])
            except ValueError:
                return False  # If the number extraction fails

            # Loop through the entries to check the sequence
            for i, entry in enumerate(data):
                expected_number = starting_number + i  # Dynamically calculate the expected number

                # Extract the actual number from the string
                image_name = entry[0]
                try:
                    actual_number = int(image_name.split('_')[1].split('.')[0])
                except ValueError:
                    return False  # If the number extraction fails

                # Check if the numbers match
                if actual_number != expected_number:
                    return False

            return True
        
        def slice_groups(groups, n):
            for i in range(0, len(groups), n):
                yield groups[i:i + n]
        # Iterate over each group
        groups = list(df.groupby('imageName'))
        for group_chunk in slice_groups(groups, chunk_size):
            if check_image_sequence(group_chunk) and len(group_chunk) == 8:
                
                annotation_per_id = {}
                for name, group in group_chunk:
                    for index, subrow in group.iterrows():
                    # Convert to groups of persons
                        if subrow['id'] not in annotation_per_id:
                            annotation_per_id[subrow['id']] = []
                            annotation_per_id[subrow['id']].append(subrow['annotation'])
                        else:
                            annotation_per_id[subrow['id']].append(subrow['annotation'])
                                  
                single_all_outside = []
                single_entering = []
                single_exiting = []
                single_all_inside_zone = []
                # moving_0_1003
                for person_id, items in annotation_per_id.items():

                    single_outside = all(not selected_environment[int(annotation.point[1])][int(annotation.point[0])] for annotation in items)
                    single_at_least_one_inside = any(selected_environment[int(annotation.point[1])][int(annotation.point[0])] for annotation in items)
                    single_first_inside = selected_environment[int(items[0].point[1])][int(items[0].point[0])] 
                    single_all_inside = all(selected_environment[int(annotation.point[1])][int(annotation.point[0])] for annotation in items)
                    
                    single_going_inside = not single_first_inside and single_at_least_one_inside
                    sincle_leaving_region = single_first_inside and not single_all_inside
                    
                    if single_outside:# all outside
                        single_all_outside.append(person_id)
                    if single_going_inside: # entry - last inside, rest outside
                        single_entering.append(person_id)
                    if sincle_leaving_region:  # exit -first inside, rest outside
                        single_exiting.append(person_id)
                    if single_all_inside:# all inside
                        single_all_inside_zone.append(person_id)
                        
                all_outside =len(single_all_outside) > 0  and len(single_entering) == 0 and len(single_exiting)== 0 and len(single_all_inside_zone) == 0
                going_inside = len(single_entering) > 0
                # leaving_region = any(single_exiting) and not going_inside
                # all_inside = any(single_all_inside_zone) and not going_inside and not all_outside
                
                
                
                # going_inside = not first_inside and at_least_one_inside
                # leaving_region = first_inside and not all_inside
                
                if going_inside:# all outside
                    group_1.append(group_chunk)
                    if args.debug:
                        debug_image = create_debug_image(cv2.imread(Path(group_chunk[0][1].annotation.iloc[0].file_path).parent.joinpath(group_chunk[0][1].annotation.iloc[0].imageName.split(':')[0]).as_posix()),selected_environment)
                if all_outside: # entry - last inside, rest outside
                    group_2.append(group_chunk)
                    if args.debug:
                        debug_image = create_debug_image(cv2.imread(Path(group_chunk[0][1].annotation.iloc[0].file_path).parent.joinpath(group_chunk[0][1].annotation.iloc[0].imageName.split(':')[0]).as_posix()),selected_environment)
            

                write_group(group_1,'entering',Path(output_foder))
                write_group(group_2,'outside',Path(output_foder))
        # write_group(group_3,'leaving',Path(output_foder))
        # write_group(group_4,'inside',Path(output_foder))
        # pass
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

    save_to_json(train_data, Path(output_foder).joinpath('annotations_train_{}.json'.format(args.version)))
    save_to_json(val_data, Path(output_foder).joinpath('annotations_validation_{}.json'.format(args.version)))
    save_to_json(test_data, Path(output_foder).joinpath('annotations_test_{}.json'.format(args.version)))


if __name__ == '__main__':
    main()