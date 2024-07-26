import glob
from pathlib import Path
import os

from dataset_handler.util.read_exr_image import read_exr_image
from dataset_handler.util.velocity_towards_camera import calculate_max_velocity_towards_camera
os.environ["OPENCV_IO_ENABLE_OPENEXR"]='1'

import cv2
import os
import pandas as pd
import matplotlib
import numpy as np
from tqdm import tqdm
import argparse
from annotation_handler_hslu import AnnotationHandlerHslu 
from dataset_handler.util.get_sementation_contour import get_segmentation_contour
from dataset_handler.util.write_movie import write_movie
from vca_blender_config_reader import VcaBlenderConfigReader


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-c','--config_path',default=r"D:\gen4") 
parser.add_argument('-w','--write_movie',default=False) 
parser.add_argument('-d','--delete_annotation',default=False) 
parser.add_argument('--min_contour_area',default=100) 
parser.add_argument('--contour_approx_eps',default=0.8) 
parser.add_argument('--combine_convex',default=False) 
parser.add_argument('--delete_empty_images',default=True) 
parser.add_argument('--depth_map',default=r"D://projects//phd_utils//distances//gen4_distance.exr") 
args = parser.parse_args()


max_velocities = []

def main():
            

    depth_clipped_to_20m = read_exr_image(args.depth_map,20.0)

    print("Loaded Config in PostProcessing")
    args_config_path = Path(args.config_path)
    if not args_config_path.exists():
        raise Exception("Not Exists {}".format(args.config_path))
    dataset_folders = os.listdir(args_config_path.as_posix())


    csv_template = "pose_frame_{}.csv"
    mask_template = "mask_p{}_{}.png"

    # Search all images
    # check if keypoints and mask are available
    # Get them and store them
    # Add directory in annotation.csv
    mask_exists = False


    for folder in tqdm(dataset_folders):
        
        image_path = args_config_path.joinpath(folder)
        if not image_path.is_dir():
            continue
        
        if image_path.joinpath("annotation.csv").exists():
            continue

        image_paths = glob.glob(str(image_path.joinpath("image_*.png")))
        keypoints_csv = []
        annotations = []
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
                        contour = get_segmentation_contour(mask)
                        approx_contour = cv2.approxPolyDP(contour[0],args.contour_approx_eps, True)
                    else:
                        mask_exists = False
                    current_keypoints = keypoints[keypoints.id == person_id]
                    current_keypoints['limb'] = current_keypoints['limb'].str.split('_').str[-2:].apply(lambda x: '_'.join(x))
                    current_keypoints = current_keypoints.drop(columns=['d_cam'])
                    current_keypoints = current_keypoints.drop(columns=['id'])

                    if mask_exists:
                        annotations.append({'Id':person_id,'Contour':contour,'Approx':approx_contour,'Frame':pathed.as_posix(),"Keypoints":current_keypoints.to_dict()})
                    else:
                        annotations.append({'Id':person_id,'Frame':pathed.as_posix(),"Keypoints":current_keypoints.to_dict()})


        annotations_pd = pd.DataFrame(annotations)
        if 'Frame' in annotations_pd.columns:
            frame_grouped = annotations_pd.groupby(by='Frame')

            frameSize = (640, 480)
            if args.write_movie:
                write_movie(frame_grouped,folder,mask_exists,frameSize)
                
            if False:
                annotation_handler = AnnotationHandlerHslu()
                for group_name,group in frame_grouped:
                    image = cv2.imread(group_name)
                    count_per_frame = len(group)
                    for index, row in group.iterrows():
                        for cx,cy,name,visible in zip(row['Keypoints']['x'].values(),row['Keypoints']['y'].values(),row['Keypoints']['limb'].values(),row['Keypoints']['visible'].values()):
                            frame_name = Path(row['Frame']).name
                            annotation_handler.addPoint(frame_name,point=[int(cx),int(cy)],label="{} - {}".format(name, visible),label_id=row['Id'])
                        if mask_exists:
                            frame_name = Path(row['Frame']).name
                            annotation_handler.addPolygon(frame_name,polygon=row['Approx'].ravel().tolist(),label='mask',label_id=row['Id'])
                    annotation_handler.addMetadata(group_name,"FrameCount",str(count_per_frame))
            if False:
                annotation_handler.writeAnnotation(image_path.joinpath("annotation.csv"))
            
            max_velocity,is_not_empty = calculate_max_velocity_towards_camera(frame_grouped,depth_clipped_to_20m)
            if is_not_empty:
                max_velocities.append({'folder':folder,'max_velocity':max_velocity})
            # END
            if image_path.joinpath("annotation.csv").exists():
                for csv in keypoints_csv:
                    os.remove(csv.as_posix())
    max_velocities_collected = pd.DataFrame(max_velocities)
    max_velocities_collected.to_csv("max_velocities_collected.csv",index=False)
        # if args.delete_empty_images:
        #     annotation_handler = AnnotationHandlerHslu()
        #     annotations = annotation_handler.readAnnotationFile(image_path.joinpath("annotation.csv").as_posix())
        #     image_size_of_series = None
        #     for annotation in annotations:
        #         if image_size_of_series == None:
        #             img = cv2.imread(image_path.joinpath(annotation[0].split(":")[0]).as_posix())
        #             image_size_of_series = img.shape[0:2]

if __name__ == '__main__':
    main()