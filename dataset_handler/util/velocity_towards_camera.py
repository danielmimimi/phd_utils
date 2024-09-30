


import numpy as np


def get_world_velocity_subset(dataset, framerate=1):
    time_interval = 1 / framerate
    velocities = []
    velocities_ms = []
    velocitie_x_ms = []
    velocities_y_ms = []

    previous_wx = None
    previous_wy = None
    
    
    for element in dataset:
        current_wx,current_wy = element.point

        # current_wx = row['Keypoints']['wx'][0]
        # current_wy = row['Keypoints']['wy'][0]
        
        if previous_wx is not None and previous_wy is not None:
            # Calculate the change in position
            delta_x = current_wx - previous_wx
            delta_y = current_wy - previous_wy
            
            # Calculate the velocity
            velocity_x = delta_x / time_interval
            velocity_y = delta_y / time_interval
            
            # Calculate the magnitude of the velocity vector
            velocity = (velocity_x**2 + velocity_y**2)**0.5
            
            # Append the velocity to the list
            velocities.append({
                'frame_index': element.imageName,
                'velocity_x': velocity_x,
                'velocity_y': velocity_y,
                'velocity': velocity
            })
            
            velocities_ms.append(velocity)
            velocitie_x_ms.append(velocity_x)
            velocities_y_ms.append(velocity_y)
            
        # Update the previous position
        previous_wx = current_wx
        previous_wy = current_wy
    velocities_kmh = [v * 3.6 for v in velocities_ms]
    velocities_x_kmh = [v * 3.6 for v in velocitie_x_ms]
    velocities_y_kmh = [v * 3.6 for v in velocities_y_ms]
    return velocities_kmh,velocities_x_kmh,velocities_y_kmh,velocities_ms

def get_world_velocity(dataset, framerate=1):
    time_interval = 1 / framerate
    velocities = []
    velocities_ms = []
    velocitie_x_ms = []
    velocities_y_ms = []

    previous_wx = None
    previous_wy = None
    for group_name, group in dataset:
        for index, row in group.iterrows():
            current_wx = row['Keypoints']['wx'][0]
            current_wy = row['Keypoints']['wy'][0]
            
            if previous_wx is not None and previous_wy is not None:
                # Calculate the change in position
                delta_x = current_wx - previous_wx
                delta_y = current_wy - previous_wy
                
                # Calculate the velocity
                velocity_x = delta_x / time_interval
                velocity_y = delta_y / time_interval
                
                # Calculate the magnitude of the velocity vector
                velocity = (velocity_x**2 + velocity_y**2)**0.5
                
                # Append the velocity to the list
                velocities.append({
                    'group_name': group_name,
                    'frame_index': index,
                    'velocity_x': velocity_x,
                    'velocity_y': velocity_y,
                    'velocity': velocity
                })
                
                velocities_ms.append(velocity)
                velocitie_x_ms.append(velocity_x)
                velocities_y_ms.append(velocity_y)
            
            # Update the previous position
            previous_wx = current_wx
            previous_wy = current_wy
    velocities_kmh = [v * 3.6 for v in velocities_ms]
    velocities_x_kmh = [v * 3.6 for v in velocitie_x_ms]
    velocities_y_kmh = [v * 3.6 for v in velocities_y_ms]
    return velocities_kmh,velocities_x_kmh,velocities_y_kmh,velocities_ms
               
                      
def get_velocity_towards_camera(dataset,depthmap,framerate=1,camera_tilt_angle=35):
    """returns kmh, ms"""
    floor_distance_to_camera = []
    for group_name,group in dataset:
        for index, row in group.iterrows():
            walking_root_x = row['Keypoints']['x'][0]
            walking_root_y = row['Keypoints']['y'][0]
            if 0 <= walking_root_x < depthmap.shape[1] and 0 <= walking_root_y < depthmap.shape[0]:      
                depth_of_walking_root = depthmap[int(walking_root_y), int(walking_root_x)]
                floor_distance_root_camera = np.sin(np.radians(90-camera_tilt_angle))*depth_of_walking_root
                floor_distance_to_camera.append(floor_distance_root_camera)
            
    velocities_ms = []
    for i, distances in enumerate(floor_distance_to_camera[1:]):
        delta_distance = floor_distance_to_camera[i] - distances
        delta_time = 1/framerate       
        velocity = delta_distance / delta_time
        velocities_ms.append(velocity)
    velocities_kmh = [v * 3.6 for v in velocities_ms]
    return velocities_kmh,velocities_ms
    

def calculate_max_velocity_towards_camera(dataset,depthmap,framerate=1,camera_tilt_angle=35):    
    # grouped in frames
    velocities_kmh,velocities_ms = get_velocity_towards_camera(dataset,depthmap,framerate,camera_tilt_angle)
    if(len(velocities_kmh)> 0):
        return max(velocities_kmh),True
    return 0,False