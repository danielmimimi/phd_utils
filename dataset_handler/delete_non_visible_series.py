import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm

from annotation_handler_hslu import AnnotationHandlerHslu


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-c','--config_path',default=r"D:\generated_annotated") 
args = parser.parse_args()

keypoints_to_inspect = ["head_end","foot_l","foot_r"]

print("Loaded Config in PostProcessing")
args_config_path = Path(args.config_path)
if not args_config_path.exists():
    raise Exception("Not Exists {}".format(args.config_path))
dataset_folders = os.listdir(args_config_path.as_posix())

unvalid = 0
for folder in tqdm(dataset_folders):
    image_path = args_config_path.joinpath(folder)

    if not image_path.joinpath("annotation.csv").exists():
        print("{} does not contain an annotation.csv".format(image_path.as_posix()))
        continue
    # READ ANNOTATION:CSV
    new_annotation_handler = AnnotationHandlerHslu()

    annotation_handler = AnnotationHandlerHslu()
    annotation_handler.readAnnotations(image_path.joinpath("annotation.csv").as_posix())
    annotations = annotation_handler.getAnnotationItemsList()
    image_size_of_series = None

    not_valid_indexes = []
    for index,annotation in enumerate(annotations):
        if image_size_of_series == None:
            correct_path = None
            # CHECK PATHING. 
            if Path(annotation.imageName.strip(":rot0")).exists():
                correct_path = Path(annotation.imageName.strip(":rot0")).name
            else:
                 correct_path = annotation.imageName.strip(":rot0")
            img = cv2.imread(image_path.joinpath(correct_path).as_posix())
            image_size_of_series = img.shape[0:2]

        # CHECK VISIBILITY OF PERSON IN IMAGE - SELECT FEET AND HEAD TIP
        if annotation.label.split("-")[0].strip() in keypoints_to_inspect:
            x = annotation.point[0]
            y = annotation.point[1]
            if x < 0 or y < 0 or y > image_size_of_series[0] or x > image_size_of_series[1]:
                not_valid_indexes.append(annotation.imageName)
                continue
        pass
    if len(not_valid_indexes) > 0:
        unvalid = unvalid + 1

    print(unvalid)
pass
    
    # UPDATE ANNOTATION 
    # WRITE ANNOTATION
    # DELETE NOT EXISTING IMAGES
    # DELETE IMAGES
