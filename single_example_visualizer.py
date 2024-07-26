

import argparse
import glob
from pathlib import Path

import cv2
import pandas as pd

from dataset_handler.util.get_sementation_contour import get_segmentation_contour
from dataset_handler.util.keypoint_getter import get_keypoint_position
from dataset_handler.util.read_exr_image import read_exr_image, select_pixels_within_range
from dataset_handler.util.velocity_towards_camera import calculate_max_velocity_towards_camera, get_velocity_towards_camera, get_world_velocity


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
# verical 'D:\gen4\moving_0_617' - Speed 105
# 
# Horizontal D:\gen4\moving_0_1142 -   Speed 65

# New 
# D:\gen4_1\moving_0_0 Speed 80
parser.add_argument('-c','--sample',default=r"D:\gen4_1\moving_0_0") 
parser.add_argument('-i','--frame_index',default=40) 
parser.add_argument('-w','--write_movie',default=False) 
parser.add_argument('-d','--delete_annotation',default=False) 
parser.add_argument('--min_contour_area',default=100) 
parser.add_argument('--contour_approx_eps',default=0.8) 
parser.add_argument('--combine_convex',default=False) 
parser.add_argument('--delete_empty_images',default=True) 
parser.add_argument('--depth_map',default=r"D://projects//phd_utils//distances//gen4_distance.exr") 
args = parser.parse_args()

csv_template = "pose_frame_{}.csv"
mask_template = "mask_p{}_{}.png"

def main():

    
    depth_clipped_to_20m = read_exr_image(args.depth_map,20.0)
    
    mm,mask= select_pixels_within_range(depth_clipped_to_20m,0,3)
    
    image_path = Path(args.sample)
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
        tracking_points = []
        for group_name,group in frame_grouped:
            
            for index, row in group.iterrows():
                
                selected_frame = cv2.imread(row['Frame'])
                kmh,ms = get_velocity_towards_camera(frame_grouped,depth_clipped_to_20m)

                kmh_w,kmh_w_x,kmh_w_y,ms_w = get_world_velocity(frame_grouped,framerate=1)
                if index < 2:
                    continue
                
                person_xy = get_keypoint_position(row,'walking_root')
                tracking_points.append(person_xy)
                
                selected_kmh,selected_ms = kmh[index-1],ms [index-1]
            
                image_to_draw = selected_frame.copy()
                door_center_point = (selected_frame.shape[1]//2,selected_frame.shape[0])
                cv2.circle(image_to_draw,door_center_point,6,(255,255,0),-1)     
                
                # Draw the previous points and connect them with lines
                for i in range(1, len(tracking_points)):
                    # Draw lines connecting previous points
                    cv2.line(image_to_draw, tracking_points[i - 1], tracking_points[i], (0, 255, 0), 2)
                    # Draw previous points
                    cv2.circle(image_to_draw, tracking_points[i - 1], 4, (0, 255, 0), -1)
                cv2.circle(image_to_draw,person_xy,6,(128,255,128),-1)
                
                
                # Add text to the top left corner
                text1 = "P - Dcp V"
                text2 = "y kmh: {:.2f}".format(sum(kmh[:index-1])/len(kmh[:index-1]))  # Replace ... with actual km/h value if available
                text3 = "y ms: {:.2f}".format(sum(ms[:index-1])/len(ms[:index-1]))   # Replace ... with actual m/s value if available

                text4 = "world kmh: {:.2f}".format(sum(kmh_w[:index-1])/len(kmh_w[:index-1]))
                text5 = "world y kmh: {:.2f}".format(sum(kmh_w_y[:index-1])/len(kmh_w_y[:index-1]))
                text6 = "world x kmh: {:.2f}".format(sum(kmh_w_x[:index-1])/len(kmh_w_x[:index-1]))
                
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_color = (0, 0, 255)
                line_type = 2

                # Position of the text
                # position1 = (10, 30)
                # position2 = (10, 60)
                position3 = (10, 90)
                position4 = (10, 120)
                position5 = (10, 150)
                position6 = (10, 180)
                # Draw the text
                # cv2.putText(image_to_draw, text1, position1, font, font_scale, font_color, line_type)
                # cv2.putText(image_to_draw, text2, position2, font, font_scale, font_color, line_type)
                # cv2.putText(image_to_draw, text3, position3, font, font_scale, font_color, line_type)
                cv2.putText(image_to_draw, text4, position4, font, font_scale, font_color, line_type)
                cv2.putText(image_to_draw, text5, position5, font, font_scale, font_color, line_type)
                cv2.putText(image_to_draw, text6, position6, font, font_scale, font_color, line_type)
                
                
        overlay = image_to_draw.copy()
        overlay[mask] = (0,255,0)
        combined_image = cv2.addWeighted(overlay, 0.3, image_to_draw, 1 - 0.3, 0)
                            
                
        cv2.imwrite("simple_setup.png",combined_image)
        pass
        
        
        pass
    
if __name__ == '__main__':
    main()