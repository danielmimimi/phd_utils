import cv2
from pathlib import Path
from annotation_handler_hslu import AnnotationHandlerHslu


def _find_longest_sequence(numbers):
    # Find the longest sequence of consecutive numbers
    longest_sequence = []
    current_sequence = []

    for i in range(len(numbers)):
        if i == 0 or numbers[i] == numbers[i - 1] + 1:
            current_sequence.append(numbers[i])
        else:
            if len(current_sequence) > len(longest_sequence):
                longest_sequence = current_sequence
            current_sequence = [numbers[i]]
    # Check the last sequence
    if len(current_sequence) > len(longest_sequence):
        longest_sequence = current_sequence
    return longest_sequence

def get_start_and_stop_frame(config_path:Path):

    keypoints_to_inspect = ["head_end","foot_l","foot_r"]

    # READ ANNOTATION.CSV
    annotation_handler = AnnotationHandlerHslu()
    annotation_handler.readAnnotations(config_path.joinpath("annotation.csv").as_posix())
    annotations = annotation_handler.getAnnotationItemsList()
    image_size_of_series = None

    not_valid_indexes = []
    valid_indexes = []
    for index,annotation in enumerate(annotations):
        if image_size_of_series == None:
            correct_path = None
            # CHECK PATHING. 
            if Path(annotation.imageName.strip(":rot0")).exists():
                correct_path = Path(annotation.imageName.strip(":rot0")).name
            else:
                correct_path = annotation.imageName.strip(":rot0")
            img = cv2.imread(config_path.joinpath(correct_path).as_posix())
            image_size_of_series = img.shape[0:2]

        # CHECK VISIBILITY OF PERSON IN IMAGE - SELECT FEET AND HEAD TIP
        if annotation.label.split("-")[0].strip() in keypoints_to_inspect:
            x = annotation.point[0]
            y = annotation.point[1]
            if x < 0 or y < 0 or y > image_size_of_series[0] or x > image_size_of_series[1]:
                not_valid_indexes.append(annotation.imageName)
            else:
                valid_indexes.append(annotation.imageName)
         
        pass
    unique_valid_indexes = set(valid_indexes)
    unique_not_valid_indexes = set(not_valid_indexes)
    unique_valid_indexes.difference_update(unique_not_valid_indexes)
    numbers = sorted(int(item.split('_')[1].split('.')[0]) for item in unique_valid_indexes)
    longest_sequence = _find_longest_sequence(numbers)

    valid = True
    if len(longest_sequence) < 8:
        valid = False
        return 0, 0,valid
    return longest_sequence[0], longest_sequence[-1],valid

    
    # UPDATE ANNOTATION 
    # WRITE ANNOTATION
    # DELETE NOT EXISTING IMAGES
    # DELETE IMAGES
