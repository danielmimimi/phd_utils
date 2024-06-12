import glob
from pathlib import Path
import cv2
import os
import pandas as pd
import matplotlib
import numpy as np
from tqdm import tqdm
import argparse
from annotation_handler_hslu import AnnotationHandlerHslu 
from vca_blender_config_reader import VcaBlenderConfigReader

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-c','--config_path',default=r"D:\generated_annotated") 
parser.add_argument('-w','--write_movie',default=False) 
parser.add_argument('-d','--delete_annotation',default=True) 
parser.add_argument('--min_contour_area',default=100) 
parser.add_argument('--contour_approx_eps',default=0.8) 
parser.add_argument('--combine_convex',default=False) 
parser.add_argument('--delete_empty_images',default=True) 

args = parser.parse_args()


def _get_segmentation_contour(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    saved_contours = []
    sizes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > args.min_contour_area:
            saved_contours.append(contour)
            sizes.append(area)

    if args.combine_convex:
        for index, contour in enumerate(saved_contours):
            if index == 0:
                final_contour = contour
            else:
                final_contour = np.vstack([final_contour,contour])

            cv2.drawContours(img_gray,[cv2.convexHull(final_contour)],0, 255, -1)
        if len(saved_contours) > 0:
            return [cv2.convexHull(final_contour)]
    else:
        max_index = np.array(sizes).argmax()
        saved_contours = saved_contours[max_index]
    return saved_contours

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
        if csv_file_path.exists():
            keypoints_csv.append(csv_file_path)
            keypoints = pd.read_csv(csv_file_path.as_posix())
            for person_id in keypoints.id.unique():
                person_mask_path = image_path.joinpath(mask_template.format(person_id,image_number))
                if person_mask_path.exists():
                    mask_exists = True
                    mask = cv2.imread(person_mask_path.as_posix())
                    contour = _get_segmentation_contour(mask)
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
    frame_grouped = annotations_pd.groupby(by='Frame')

    frameSize = (640, 480)
    if args.write_movie:
        movel_real = []
        
        for group_name,group in frame_grouped:
            image_real = cv2.imread(group_name)
            movel_real.append(image_real)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(folder+"_frames.mp4",fourcc, 10, frameSize)

        for img in movel_real:
            out.write(img)
        out.release()


    if args.write_movie:
        movie_ann = []
        rgb_colors = {}
        for name, hex in matplotlib.colors.cnames.items():
            rgb_colors[name] = matplotlib.colors.to_rgb(hex)
        accessible_colors = list(rgb_colors.values())
        
        for group_name,group in frame_grouped:
            
            image = cv2.imread(group_name)
            for index, row in group.iterrows():
                color = accessible_colors[int(row['Id']) % (len(accessible_colors)-1)]
                color = (int(color[0]*255),int(color[1]*255),int(color[2]*255))
                color_v = (int(color[0]*125),int(color[1]*0),int(color[2]*125))

                keypoint_length = []
                key_point_names = ["x","y","limb","visible"]
                for key in row.Keypoints:
                    keypoint_length.append(len(row.Keypoints[key]))
                
                assert all(x == keypoint_length[0] for x in keypoint_length)

                for index in range(0,keypoint_length[0]):
                    x = row.Keypoints[key_point_names[0]][index]
                    y = row.Keypoints[key_point_names[1]][index]
                    limb = row.Keypoints[key_point_names[2]][index]
                    visible = row.Keypoints[key_point_names[3]][index]
                    if visible:
                        cv2.drawMarker(image,(int(x),int(y)),color=color,line_type=cv2.LINE_4)
                    else:
                        cv2.drawMarker(image,(int(x),int(y)),color=color_v,line_type=cv2.LINE_4)
                    
                
                # cv2.drawMarker(image,(int(row['Cx']),int(row['Cy'])),color=color,line_type=cv2.LINE_4)
                # cv2.putText(image, str(row['Id']), (int(row['Cx']),int(row['Cy'])), cv2.FONT_HERSHEY_SIMPLEX, 
                #     0.5, color, 1, cv2.LINE_AA)
                if mask_exists:
                    cv2.drawContours(image=image,contours=[row['Approx']],contourIdx=-1,color=color,thickness=1)    
            # plt.imshow(image)
            movie_ann.append(image)
            movel_real.append(image)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(folder+"_ann.mp4",fourcc, 10, frameSize)

        for img in movie_ann:
            out.write(img)
        out.release()

    if True:
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
    annotation_handler.writeAnnotation(image_path.joinpath("annotation.csv"))
    # END
    if image_path.joinpath("annotation.csv").exists():
        for csv in keypoints_csv:
            os.remove(csv.as_posix())

    # if args.delete_empty_images:
    #     annotation_handler = AnnotationHandlerHslu()
    #     annotations = annotation_handler.readAnnotationFile(image_path.joinpath("annotation.csv").as_posix())
    #     image_size_of_series = None
    #     for annotation in annotations:
    #         if image_size_of_series == None:
    #             img = cv2.imread(image_path.joinpath(annotation[0].split(":")[0]).as_posix())
    #             image_size_of_series = img.shape[0:2]

 