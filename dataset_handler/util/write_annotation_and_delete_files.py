import os
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
import glob
from dataset_handler.annotation_handler_hslu import AnnotationHandlerHslu
from dataset_handler.util.get_sementation_contour import get_segmentation_contour
from dataset_handler.util.write_movie import write_movie

def write_annotation_and_delete_files(args,delete_in_the_end:bool=False):
    
    csv_template = "pose_frame_{}.csv"
    mask_template = "mask_p{}_{}.jpg"


    available_annotation_files = []

    args_config_path = Path(args.config_path)
    if not args_config_path.exists():
        raise Exception("Not Exists {}".format(args.config_path))
    dataset_folders = os.listdir(args_config_path.as_posix())
    
    mask_exists = False
    
    for folder in tqdm(dataset_folders):
        
        image_path = args_config_path.joinpath(folder)
        if not image_path.is_dir():
            continue
        
        print(folder)
        if image_path.joinpath("annotation.csv").exists():
            # os.remove(image_path.joinpath("annotation.csv").as_posix())
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
            if False:
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
                if delete_in_the_end: # DONT DELETE 
                    for csv in keypoints_csv:
                        os.remove(csv.as_posix())
                    for mask_path in mask_paths:
                        os.remove(mask_path.as_posix())
    return available_annotation_files