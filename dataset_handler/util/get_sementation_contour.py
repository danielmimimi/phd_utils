import cv2
import numpy as np


# def get_segmentation_contour(image,args):
#     img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     saved_contours = []
#     sizes = []
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > args.min_contour_area:
#             saved_contours.append(contour)
#             sizes.append(area)

#     if args.combine_convex:
#         for index, contour in enumerate(saved_contours):
#             if index == 0:
#                 final_contour = contour
#             else:
#                 final_contour = np.vstack([final_contour,contour])

#             cv2.drawContours(img_gray,[cv2.convexHull(final_contour)],0, 255, -1)
#         if len(saved_contours) > 0:
#             return [cv2.convexHull(final_contour)]
#     else:
#         max_index = np.array(sizes).argmax()
#         saved_contours = saved_contours[max_index]
#     return saved_contours

def get_segmentation_contour(image, args):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > args.min_contour_area]
    
    max_contours = 10
    if len(filtered_contours) > max_contours:
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)[:max_contours]
    
    if not filtered_contours:
        return None  # No contours found that match the criteria
    
    if args.combine_convex:
        combined_contour = np.vstack(filtered_contours)
        convex_hull = cv2.convexHull(combined_contour)
        cv2.drawContours(img_gray, [convex_hull], 0, 255, -1)
        return [convex_hull]
    else:
        # Find the largest contour by area
        largest_contour = max(filtered_contours, key=cv2.contourArea)
        return largest_contour