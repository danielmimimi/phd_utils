


def get_keypoint_position(data_dict,searched_limb_name='walking_root'):
    length_of_data = len(data_dict['Keypoints']['x'])
    for index in range(0,length_of_data):
        x = data_dict['Keypoints']['x'][index]
        y = data_dict['Keypoints']['y'][index]
        limb_name = data_dict['Keypoints']['limb'][index]
        
        if limb_name == searched_limb_name:
            return (int(x),int(y))
    return (0,0)