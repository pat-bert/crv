import csv
import os
from pathlib import Path
from typing import Union, List, Optional, Iterator, Tuple

import cv2

ROOT_PATH = Path(r'D:\Nutzer\Videos\CRV')
IMAGE_PATHS = [Path(ROOT_PATH, f'images_320-240_{i}') for i in range(1, 6)]
CURRENT_DIR = Path('.').absolute()
LABELS_ROOT_PATH = Path(CURRENT_DIR, 'labels-final-revised1')
DL_PARENT_PATH = Path(CURRENT_DIR, 'ressource')

# Labels grouped by topic
scroll_hand = [i for i in range(1, 7)]
zoom_fists = [8, 9]
rotate_fists = [10, 11]
zoom_fingers = [12, 13]
rotate_fingers = [14, 15]
sweep = [i for i in range(17, 21)]
digits = [i for i in range(24, 34)]

# Subjects grouped by category
TRAINING_SUBJECTS = [3, 4, 5, 6, 8, 10, 15, 16, 17, 20, 21, 22, 23, 25, 26, 27, 30, 32, 36, 38, 39, 40, 42, 43, 44, 45,
                     46, 48, 49, 50]
VALIDATION_SUBJECTS = [1, 7, 12, 13, 24, 29, 33, 34, 35, 37]
TEST_SUBJECTS = [2, 9, 11, 14, 18, 19, 28, 31, 41, 47]

ALLOWED_LABELS = digits


def get_label(filepath: str, reduce_intvl: float = 1) -> Union[None, int]:
    """
    Check the label for a given image in the database
    :param filepath: Absolute filepath of an image
    :param reduce_intvl: Inner share of the interval to be used
    :return: Label as int, None if no label assigned
    """
    # Interval share to be clipped on each side
    clip_intvl = 0.5 * (1 - reduce_intvl)

    # Convert to Path
    path = Path(filepath)
    parts = path.parts

    # Split into components starting from last component since relative path is invariant
    frame = int(parts[-1].split('.')[0])
    group, scene, subject = parts[-2], parts[-4], parts[-5]

    # Construct the path for the CSV file containing the label information
    label_path = Path(LABELS_ROOT_PATH, subject.lower(), scene, f'Group{group[-1]}.csv')

    # Search for label in content of CSV file
    with open(label_path) as csv_file:
        for row in csv.reader(csv_file, delimiter=','):
            # Each row lists a label and its corresponding frame range
            if all(len(i) > 0 for i in row):
                label, min_frame, max_frame = [int(i) for i in row]
                diff_frame = max_frame - min_frame
                if min_frame + round(clip_intvl * diff_frame) <= frame <= max_frame - round(clip_intvl * diff_frame):
                    return label
    return None


def get_images(allowed_labels: List[int], color: Optional[bool] = True, reduce_intvl: float = 1) \
        -> Iterator[Tuple[int, str]]:
    """
    Construct a generator of image filepaths to be considered
    :param allowed_labels: Labels that should be processed, other image filepaths are not returned
    :param color: Flag to indicate whether RGB or depth images should be used
    :param reduce_intvl:
    :return: Generator of filepaths for images matching the filters
    """
    for image_dir in IMAGE_PATHS:
        for root, dirs, files in os.walk(image_dir):
            if color:
                dirs[:] = [d for d in dirs if d != 'Depth']
            else:
                dirs[:] = [d for d in dirs if d != 'Color']

            for file in files:
                filepath = Path(root, file)
                img_label = get_label(filepath, reduce_intvl=reduce_intvl)
                if img_label in allowed_labels:
                    yield img_label, str(filepath)


if __name__ == '__main__':
    # Create folder structure for deep learning
    DL_ROOT_PATH = Path(DL_PARENT_PATH, 'DL')
    DL_ROOT_PATH.mkdir(exist_ok=True)

    training_dir = Path(DL_ROOT_PATH, 'Training')
    training_dir.mkdir(exist_ok=True)

    validation_dir = Path(DL_ROOT_PATH, 'Validation')
    validation_dir.mkdir(exist_ok=True)

    test_dir = Path(DL_ROOT_PATH, 'Test')
    test_dir.mkdir(exist_ok=True)

    # Load relevant images by RGB/Depth and label
    cnt = 0
    for label, img_path_str in get_images(ALLOWED_LABELS, color=False, reduce_intvl=0.3):
        cnt += 1
        print(f'Processing image #{cnt}')
        # Preprocess each image with OpenCV
        img = cv2.imread(img_path_str)
        img_path = Path(img_path_str)

        # Determine category
        subject_part = [part for part in img_path.parts if part.startswith('Subject')]
        subject_number = int(subject_part[0].split('Subject')[-1])
        scene_part = [part for part in img_path.parts if part.startswith('Scene')]
        scene_number = int(scene_part[0].split('Scene')[-1])
        type_part = img_path.parts[-2]

        # Create folder for label
        if subject_number in TRAINING_SUBJECTS:
            label_dir = Path(training_dir, str(label))
        elif subject_number in VALIDATION_SUBJECTS:
            label_dir = Path(validation_dir, str(label))
        else:
            label_dir = Path(test_dir, str(label))
        label_dir.mkdir(exist_ok=True)

        # Save ROI of image into correct folder
        destination_path = Path(label_dir, f'Subject{subject_number}_Scene{scene_number}_{type_part}_{img_path.name}')
        cv2.imwrite(str(destination_path), img)

    # Deep Learning
