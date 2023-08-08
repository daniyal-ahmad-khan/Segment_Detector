# README

## Overview

This is a Python application that reads an image, creates masks, and overlays tiles on the image based on the provided configuration in `config.py`.

## Pre-requisites

To run this application, you need Python version 3.9 or higher (3.11.3 is used for development).

## Installation

1. First, download the zip file to your local machine and unzip it. Open your terminal in the unzipped folder.

2. Create a new conda environment with Python 3.11.3 (or any version above 3.9):

    ```
    conda create -n <env_name> python=3.11.3
    ```

3. Activate the newly created environment:

    ```
    conda activate <env_name>
    ```

4. Install the required packages using the `requirements.txt` file:

    ```
    pip install -r requirements.txt
    ```

Replace `<env_name>` with the name of your environment.

## Usage

The application reads its configuration from a file named `config.py`. To use this application, ensure you have set the following parameters in `config.py`:

- `image_path`: Path to the image you want to edit.
- `floor_tile_path`: Path to the tile image you want to overlay on the floor.
- `wall_tile_path`: Path to the tile image you want to overlay on the image.
- `floor_rotation_angle`: Angle by which you would like to rotate the floor tile.
- `wall_rotation_angle`: Angle by which you would like to rotate the wall tile.
- `floor_percentage`: Percentage of the floor mask you would like for one tile to cover.
- `wall_percentage`: Percentage of the wall mask you would like for one tile to cover.

After setting the configuration, run the `main.py` file:

```
python main.py
```

This will process the image based on your settings and save the resulting image.

Please ensure that the paths to the images are correct and the images are accessible from your working directory.

