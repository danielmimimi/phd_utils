import OpenEXR
import Imath
import numpy as np

def read_exr_image(path_to_image,clip_distance:float=20.0):
    depth_exr = OpenEXR.InputFile(path_to_image)
    header = depth_exr.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    r_str = depth_exr.channel('V', pt)
    
    r = np.frombuffer(r_str, dtype=np.float32).reshape((height, width))
    
    depth_clipped_to_20m = np.clip(r, None, clip_distance)
    return depth_clipped_to_20m


def select_pixels_within_range(depth_map, min_distance: float, max_distance: float):
    mask = (depth_map >= min_distance) & (depth_map <= max_distance)
    selected_pixels = depth_map[mask]
    return selected_pixels, mask