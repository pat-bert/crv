import csv
from pathlib import Path
from typing import Union

ROOT_PATH = r'D:\Nutzer\Videos\CRV'
LABELS_ROOT_PATH = Path(ROOT_PATH, 'labels-final-revised1')


def get_label(filepath: str) -> Union[None, int]:
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
            label, min_frame, max_frame = [int(i) for i in row]
            if min_frame <= frame <= max_frame:
                return label
    return None


if __name__ == '__main__':
    image_filepath = r'D:\Nutzer\Videos\CRV\images_320-240_1\Subject05\Scene3\Color\rgb2\000196.jpg'
    image_label = get_label(image_filepath)
    print(image_label)
