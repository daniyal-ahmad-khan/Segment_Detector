from process_image import read_image
from process_image import create_masks, remove_enclosed_masks, remove_small_masks
from process_image import tile_image

from config import *

image = read_image(image_path)
masks = create_masks(image)
masks = remove_enclosed_masks(masks)
masks = remove_small_masks(masks, image)
tile_image(image, image_path, masks,floor_tile_path, wall_tile_path, floor_rotation_angle, wall_rotation_angle,floor_percentage, wall_percentage)