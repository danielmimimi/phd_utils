
import cv2
import matplotlib


def write_movie(frame_grouped,folder,frameSize = (640, 480)):
        
        movel_real = []
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
                if row['mask_exists']:
                    cv2.drawContours(image=image,contours=[row['Approx']],contourIdx=-1,color=color,thickness=1)    
            # plt.imshow(image)
            movie_ann.append(image)
            movel_real.append(image)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(folder+"_ann.mp4",fourcc, 10, frameSize)

        for img in movie_ann:
            out.write(img)
        out.release()