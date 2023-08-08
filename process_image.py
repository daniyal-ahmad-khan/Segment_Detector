## Import the necessary modules

import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"  # Choose what to use for infrerence ('cpu', 'cuda'), CPU recommended if GPU not available
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


## Function to read the image
def read_image(path, max_size=(1920, 1080)):
    image = cv2.imread(path)
    assert image is not None, 'Image could not be read'

    # Get the aspect ratio of the image
    h, w = image.shape[:2]
    aspect_ratio = w/h

    if w>h: # If the image is wide
        if w > max_size[0]: # If the width is greater than the maximum width
            w = max_size[0] # Set the width to the maximum width
            h = int(w / aspect_ratio) # Compute the corresponding height
    else: # If the image is tall or square
        if h > max_size[1]: # If the height is greater than the maximum height
            h = max_size[1] # Set the height to the maximum height
            w = int(h * aspect_ratio) # Compute the corresponding width

    image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)

    return image

import cv2
import torch

def modify_bounding_boxes(bboxes, img):
    '''
    bboxes: PyTorch tensor of bounding boxes each of (xmin, ymin, xmax, ymax) on a CUDA device
    img: numpy array read using cv2 (HxWxC)
    '''

    # Move the bounding boxes to CPU for calculations
    bboxes = bboxes.to('cpu')

    height, width, _ = img.shape
    img_center_x, img_center_y = width / 2, height / 2
    img_area = width * height

    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox

        # Calculate center of the bounding box
        bbox_center_x, bbox_center_y = (xmin + xmax) / 2, (ymin + ymax) / 2

        # Calculate the area of the bounding box
        bbox_area = (xmax - xmin) * (ymax - ymin)

        # Check if the center of the bounding box is below the center of the image
        # and the bounding box covers more than 20% of the image
        if bbox_center_y > img_center_y and bbox_area / img_area > 0.2:
            # Change the dimension of the bounding box such that it covers
            # the entire width of the image (without changing its height)
            xmin = 0
            xmax = width
            # Update bounding box
            bboxes[i] = torch.tensor([xmin, ymin, xmax, ymax])

    # Move the modified bounding boxes back to the original device
    bboxes = bboxes.to(bbox.device)

    return bboxes


## The following function returns a list of dictionaries for one image in the following format: [{'class':'floor', 'segmentation': H x W binary mask},
##                                                                                      {'class':'wall', 'segmentation': H x W binary mask}, . . .  ]

def create_masks(image, yolo_path="best_9.pt"):
# def create_masks(image, yolo_path="best.pt"):

  model = YOLO(yolo_path)
  objects = model(image, save = False, line_width=1, stream=False, conf = 0.5)
  classes = []
  input_points = None
  for o in objects:
    if o is None:
      print('No Detections Found...')
      return None, None
    else:
      masks = o.masks
      boxes = o.boxes
      polygons = masks.xy
      for i in range(len(polygons)):
          if len(polygons[i]) == 0:
              polygons[i] = [[0.0, 0.0],[0,image.shape[0]], [image.shape[1],0], [image.shape[1], image.shape[0]]]
      input_points = [find_inaccessible_point(Polygon(polygon)) for polygon in polygons]
      cls = boxes.cls.type(torch.int)
      for i, (c, bbox) in enumerate(zip(cls, boxes.xyxy.tolist())):
          class_names = ['floor', 'wall']
          output_index = c
          class_name = class_names[output_index]
          classes.append(class_name)

  input_points = torch.tensor(input_points, device=device)
  input_boxes = torch.tensor(boxes.xyxy, device=device)
  input_boxes = modify_bounding_boxes(input_boxes, image)
  input_boxes = torch.tensor(input_boxes, device=device)
  input_points = input_points.type(torch.int64)
  input_points = input_points.unsqueeze(1)
  if len(input_boxes) == 0:
    return None, None


  predictor = SamPredictor(sam)
  predictor.set_image(image)
  transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
  transformed_points = predictor.transform.apply_coords_torch(input_points, image.shape[:2])

  MASKS = []

  masks, scores, logits = predictor.predict_torch(
  point_coords=transformed_points,
  point_labels=None,
  boxes = transformed_boxes,
  multimask_output=False)

  MASKS = masks.detach().cpu().numpy()
  MASKS = MASKS[:,0,:,:]

  MASKS = cleanup_masks(MASKS, min_size=0.01*image.shape[0]*image.shape[1])
  masks_dict_list = []

  for cls, mask in zip(classes, MASKS):
      masks_dict_list.append({
          'class': cls,
          'segmentation': mask.astype(bool)
      })
  return masks_dict_list

## Helper function for create_masks function
def find_inaccessible_point(polygon, precision=10):
    # Generate a grid of points within the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds
    x_coords = list(np.linspace(minx, maxx, precision))
    y_coords = list(np.linspace(miny, maxy, precision))
    grid_points = [Point(x, y) for x in x_coords for y in y_coords]

    # Find the point in the grid that's contained in the polygon and is furthest from the exterior
    max_distance = 0
    best_point = None
    for point in grid_points:
        if polygon.contains(point):
            distance = polygon.exterior.distance(point)
            if distance > max_distance:
                max_distance = distance
                best_point = point

    return best_point.x, best_point.y  # Return as a tuple of coordinates

## Helper function for create_masks function
from skimage import morphology
def cleanup_masks(masks, min_size=1000):
  cleaned_masks = []
  for mask in masks:
      # Remove small holes
      mask = morphology.remove_small_holes(mask, min_size)
      # Remove small objects
      mask = morphology.remove_small_objects(mask, min_size)
      cleaned_masks.append(mask)
  return cleaned_masks

def remove_enclosed_masks(masks):
    masks_to_keep = []

    for i in range(len(masks)):
        if masks[i]['class'] != 'wall':
            masks_to_keep.append(masks[i])
            continue

        mask_i = masks[i]['segmentation']
        is_enclosed = False

        # Get bounding box of mask i
        mask_i_indices = np.where(mask_i)
        if mask_i_indices[0].size == 0:  # If there are no True values in mask, skip to next mask
            continue
        min_y_i, max_y_i = np.min(mask_i_indices[0]), np.max(mask_i_indices[0])
        min_x_i, max_x_i = np.min(mask_i_indices[1]), np.max(mask_i_indices[1])

        for j in range(len(masks)):
            if i != j and masks[j]['class'] == 'wall':
                mask_j = masks[j]['segmentation']

                # Get bounding box of mask j
                mask_j_indices = np.where(mask_j)
                if mask_j_indices[0].size == 0:  # If there are no True values in mask, skip to next mask
                    continue
                min_y_j, max_y_j = np.min(mask_j_indices[0]), np.max(mask_j_indices[0])
                min_x_j, max_x_j = np.min(mask_j_indices[1]), np.max(mask_j_indices[1])

                # Check if mask i is enclosed by mask j
                if (min_y_i >= min_y_j) and (max_y_i <= max_y_j) and (min_x_i >= min_x_j) and (max_x_i <= max_x_j):
                    is_enclosed = True
                    break

        if not is_enclosed:
            masks_to_keep.append(masks[i])

    return masks_to_keep

def remove_small_masks(masks, img):
    total_pixels = img.shape[0] * img.shape[1]
    threshold = 0.001 * total_pixels

    masks_to_keep = []

    for mask in masks:
        if np.sum(mask['segmentation']) > threshold:
            masks_to_keep.append(mask)

    return masks_to_keep

def create_individual_tile_image(tile,image_width, image_height, rotation_angle):
  from PIL import Image, ImageDraw, ImageOps
  import math

  # Load your tile
  tile = Image.fromarray(np.uint8(tile))

  # Crop out the border of the tile, if necessary
  border = 10  # Set this to the width of the border
  tile = tile.crop((border, border, tile.size[0] - border, tile.size[1] - border))

  # Set the dimensions of the final image
  final_img_width, final_img_height = image_width, image_height

  # Calculate the size of the larger canvas. This will need to be the diagonal of the final image.
  larger_size = int(math.sqrt(final_img_width**2 + final_img_height**2))

  # Create a larger blank canvas
  final_img = Image.new('RGB', (larger_size, larger_size))

  # Calculate the number of full tiles in each dimension
  num_full_tiles_width = larger_size // tile.size[0]
  num_full_tiles_height = larger_size // tile.size[1]

  # Calculate the remainder pixels in each dimension
  remainder_width = larger_size % tile.size[0]
  remainder_height = larger_size % tile.size[1]

  # Loop over each row and column and paste full tiles
  for i in range(num_full_tiles_height):
      for j in range(num_full_tiles_width):
          # Calculate the position of this tile
          x = j * tile.size[0]
          y = i * tile.size[1]

          # Paste the tile into the final image at this position
          final_img.paste(tile, (x, y))

  # If there is a remainder width, paste a cropped tile for each row
  if remainder_width > 0:
      for i in range(num_full_tiles_height):
          x = num_full_tiles_width * tile.size[0]
          y = i * tile.size[1]
          cropped_tile = tile.crop((0, 0, remainder_width, tile.size[1]))
          final_img.paste(cropped_tile, (x, y))

  # If there is a remainder height, paste a cropped tile for each column
  if remainder_height > 0:
      for j in range(num_full_tiles_width):
          x = j * tile.size[0]
          y = num_full_tiles_height * tile.size[1]
          cropped_tile = tile.crop((0, 0, tile.size[0], remainder_height))
          final_img.paste(cropped_tile, (x, y))

  # If there are remainder width and height, paste a cropped tile at the bottom right corner
  if remainder_width > 0 and remainder_height > 0:
      x = num_full_tiles_width * tile.size[0]
      y = num_full_tiles_height * tile.size[1]
      cropped_tile = tile.crop((0, 0, remainder_width, remainder_height))
      final_img.paste(cropped_tile, (x, y))

  # Now let's rotate
  angle = rotation_angle  # Set your rotation angle
  final_img = final_img.rotate(angle, expand=True)

  # And then crop
  left = (final_img.width - final_img_width) / 2
  top = (final_img.height - final_img_height) / 2
  right = (final_img.width + final_img_width) / 2
  bottom = (final_img.height + final_img_height) / 2
  final_img = final_img.crop((left, top, right, bottom))

  return np.array(final_img)

def create_tile_image(masks_dict_list, image, floor_tile_path, wall_tile_path, floor_rotation_angle, floor_percentage, wall_percentage, wall_rotation_angle):
  from PIL import Image, ImageDraw, ImageOps
  import math

  floor_tile_img = cv2.imread(floor_tile_path)
  wall_tile_img = cv2.imread(wall_tile_path)

  # Determine if floor and wall exist in masks
  floor_in_masks = any(mask_dict['class'] == 'floor' for mask_dict in masks_dict_list)
  wall_in_masks = any(mask_dict['class'] == 'wall' for mask_dict in masks_dict_list)

  # Calculate total areas of floor and wall in the image
  floor_area = sum(mask_dict['segmentation'].sum() for mask_dict in masks_dict_list if mask_dict['class'] == 'floor') if floor_in_masks else 0
  wall_area = sum(mask_dict['segmentation'].sum() for mask_dict in masks_dict_list if mask_dict['class'] == 'wall') if wall_in_masks else 0

  # Calculate desired tile area for floor and wall
  floor_tile_area = floor_area * floor_percentage / 100 if floor_in_masks else 0
  wall_tile_area = wall_area * wall_percentage / 100 if wall_in_masks else 0

  # Compute scale factors for floor and wall tile images
  floor_scale_factor = np.sqrt(floor_tile_area / (floor_tile_img.shape[0] * floor_tile_img.shape[1])) if floor_in_masks and floor_tile_img is not None else 0
  wall_scale_factor = np.sqrt(wall_tile_area / (wall_tile_img.shape[0] * wall_tile_img.shape[1])) if wall_in_masks and wall_tile_img is not None else 0

  # Calculate new dimensions for floor and wall tile images
  floor_tile_new_size = (int(floor_tile_img.shape[1] * floor_scale_factor), int(floor_tile_img.shape[0] * floor_scale_factor)) if floor_in_masks and floor_tile_img is not None else (0, 0)
  wall_tile_new_size = (int(wall_tile_img.shape[1] * wall_scale_factor), int(wall_tile_img.shape[0] * wall_scale_factor)) if wall_in_masks and wall_tile_img is not None else (0, 0)

  # Resize tile images
  floor_tile_img = cv2.resize(floor_tile_img, floor_tile_new_size) if floor_in_masks and floor_tile_img is not None else None
  wall_tile_img = cv2.resize(wall_tile_img, wall_tile_new_size) if wall_in_masks and wall_tile_img is not None else None

  floor_tiling = create_individual_tile_image(floor_tile_img, image.shape[1], image.shape[0], floor_rotation_angle) if floor_in_masks and floor_tile_img is not None else None
  wall_tiling = create_individual_tile_image(wall_tile_img, image.shape[1], image.shape[0], wall_rotation_angle) if wall_in_masks and wall_tile_img is not None else None
  return floor_tiling, wall_tiling

def save_and_blend_images(output_dir, image, floor_in_masks, wall_in_masks, tiled_floor_img, tiled_wall_img, binary_floor_mask, binary_wall_mask):
    """
    A helper function that saves and blends the tiled images.
    """
    # Save and blend images for each mask class

    if floor_in_masks and tiled_floor_img is not None:
        save_and_blend_image('floor', output_dir, image, tiled_floor_img)
    if wall_in_masks and tiled_wall_img is not None:
        save_and_blend_image('wall', output_dir, image, tiled_wall_img)
    if floor_in_masks and wall_in_masks and tiled_floor_img is not None and tiled_wall_img is not None:
        # Create an image with both the floor and wall tiles applied
        tiled_floor_and_wall_img = image.copy()
        tiled_floor_and_wall_img[binary_floor_mask == 255] = tiled_floor_img[binary_floor_mask == 255]
        tiled_floor_and_wall_img[binary_wall_mask == 255] = tiled_wall_img[binary_wall_mask == 255]
        binary_floor_wall_mask = binary_floor_mask.copy()
        binary_floor_wall_mask[binary_wall_mask == 255] = binary_wall_mask[binary_wall_mask == 255]
        smoothed_tiling = smooth_outer_edges(tiled_floor_and_wall_img, binary_floor_wall_mask, (25,25))
        save_and_blend_image('floor_and_wall', output_dir, image, smoothed_tiling)

def save_and_blend_image(mask_class, output_dir, original_image, tiled_image):
    """
    A helper function that saves and blends a single image.
    """
    # Save tiled image
    # cv2.imwrite(os.path.join(output_dir, f'tiled_{mask_class}.png'), tiled_image)

    # Blend the lightness channel with the original image
    original_lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)
    tiled_lab = cv2.cvtColor(tiled_image, cv2.COLOR_BGR2Lab)
    lightness_blend = cv2.addWeighted(tiled_lab[:,:,0], 0.8, original_lab[:,:,0], 0.2, 0)
    tiled_lab[:,:,0] = lightness_blend
    blended_image = cv2.cvtColor(tiled_lab, cv2.COLOR_Lab2BGR)

    # Save blended image
    cv2.imwrite(os.path.join(output_dir, f'blended_{mask_class}.png'), blended_image)

def smooth_outer_edges(tiled_img, binary_mask, kernel_size):
    """
    Blur the edges of the tiled areas in the image.
    Args:
        tiled_img (ndarray): The image with tiled areas.
        binary_mask (ndarray): The binary mask of the tiled areas.
        kernel_size (tuple): The size of the kernel for the blur.

    Returns:
        smoothed_img (ndarray): The image with smoothed tiled areas.
    """

    # Detect the edges of the mask
    edge_mask = cv2.Canny(binary_mask, 100, 200)

    # Create a boundary mask of the image
    boundary_mask = np.ones_like(binary_mask)
    boundary_mask[:5, :] = 0
    boundary_mask[-5:, :] = 0
    boundary_mask[:, :5] = 0
    boundary_mask[:, -5:] = 0

    # Dilate the edge mask to get a wider boundary
    dilated_edge_mask = cv2.dilate(edge_mask, np.ones((15,15), np.uint8), iterations = 1)

    # Bitwise AND with the boundary mask to ensure dilation stays within the image dimensions
    dilated_edge_mask = cv2.bitwise_and(dilated_edge_mask, boundary_mask)

    # Blur the entire tiled image
    blurred_tiled_img = cv2.GaussianBlur(tiled_img, kernel_size, 0)

    # Create a boolean mask for the dilated edge area
    dilated_edge_mask_bool = dilated_edge_mask > 0

    # Create a copy of the tiled image
    smoothed_img = tiled_img.copy()

    # Blend the blurred edge area with the original tiled image
    smoothed_img[dilated_edge_mask_bool] = blurred_tiled_img[dilated_edge_mask_bool]

    return smoothed_img

def tile_image(image, path, masks_dict_list, floor_tile_path, wall_tile_path, floor_rotation_angle, wall_rotation_angle, floor_percentage, wall_percentage):
    # Load and rotate tile images
    floor_in_masks = any(mask_dict['class'] == 'floor' for mask_dict in masks_dict_list)
    wall_in_masks = any(mask_dict['class'] == 'wall' for mask_dict in masks_dict_list)
    floor_tile_img, wall_tile_img = create_tile_image(masks_dict_list, image, floor_tile_path, wall_tile_path,
                                                        floor_rotation_angle, floor_percentage,
                                                        wall_percentage, wall_rotation_angle)
    # Initialize the output directory
    output_dir = os.path.join('Tiled Outputs', os.path.basename(path.split(".")[0]))  # Create output directory based on timestamp
    os.makedirs(output_dir, exist_ok=True)  # Creates the directory if it doesn't exist

    # Initialize tiled floor and wall images and binary masks
    tiled_floor_img, tiled_wall_img = None, None
    binary_floor_mask, binary_wall_mask = None, None
    if floor_in_masks and floor_tile_img is not None:
        tiled_floor_img = image.copy()
        binary_floor_mask = np.zeros(image.shape[:2], dtype='uint8')
    if wall_in_masks and wall_tile_img is not None:
        tiled_wall_img = image.copy()
        binary_wall_mask = np.zeros(image.shape[:2], dtype='uint8')

    # Process all masks
    for mask_dict in masks_dict_list:
        mask_class = mask_dict['class']
        mask = mask_dict['segmentation']  # Assuming segmentation is already a grayscale image

        # Skip mask if it's empty or None
        if mask is None or mask.size == 0:
            print("The mask is None or empty.")
            continue

        # Convert mask to grayscale if it is a color image
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask * 255).astype('uint8')   # Ensure mask has correct data type

        # Threshold the mask to binary form
        _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Perform tiling for each mask class
        if mask_class in ["wall", "floor"] and (floor_in_masks or wall_in_masks):
            # Determine current class and set the tile image and dimensions
            tile_img = wall_tile_img if mask_class == "wall" else floor_tile_img
            if tile_img is None:
                continue
            tile_height, tile_width = tile_img.shape[:2]
            binary_mask_current = binary_wall_mask if mask_class == "wall" else binary_floor_mask
            tiled_img_current = tiled_wall_img if mask_class == "wall" else tiled_floor_img

            # Apply tiling
            binary_mask_current[binary_mask == 255] = 255
            y_indices, x_indices = np.where(binary_mask == 255)
            y_tile_indices = y_indices % tile_height
            x_tile_indices = x_indices % tile_width
            tiled_img_current[y_indices, x_indices] = tile_img[y_tile_indices, x_tile_indices]

        # Smooth the outer edges of the tiled images
    smoothed_floor_tiling = smooth_outer_edges(tiled_floor_img, binary_floor_mask, (15,15)) if floor_in_masks and floor_tile_img is not None else None
    smoothed_wall_tiling = smooth_outer_edges(tiled_wall_img, binary_wall_mask, (15,15)) if wall_in_masks and wall_tile_img is not None else None

    # Call the save_and_blend_images function with the smoothed images
    save_and_blend_images(output_dir, image, floor_in_masks, wall_in_masks, smoothed_floor_tiling, smoothed_wall_tiling, binary_floor_mask, binary_wall_mask)