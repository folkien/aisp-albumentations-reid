#!/usr/bin/python3
from copy import copy
import os
from pathlib import Path
import sys
import random
import argparse
import logging
from tqdm import tqdm
from engine.AnnoterReid import AnnoterReid
from engine.ImageData import ImageData
from engine.ReidFileInfo import ReidFileInfo
from helpers.augumentations import Augment, transform_color, transform_shape, transform_all
from helpers.files import FixPath, GetFileLocation 

def Process(path: str, arguments: argparse.Namespace):
    ''' Process directory'''
    # Check : Path is None or empty
    if (path is None) or (path == ''):
        logging.error('Path is None or empty!')
        return

    # Generated : Create output directory
    outputPath = os.path.join(path, 'generated')
    Path(outputPath).mkdir(parents=True, exist_ok=True)

    # Annoter : Create
    annoter = AnnoterReid(dirpath=FixPath(GetFileLocation(args.input)),
                          args=args,
                          )

    # Counter : Of processed images
    processed_counter = 0
    # Preview: ProgressBar : Create
    progress = tqdm(total=args.iterations, 
                    desc='Augumentation', 
                    unit='images')
    
    # Albumentations per identity : Calculate
    albumentations_per_image = max(1, round(args.iterations / len(annoter.identities)))

    # Identities : Loop over every identity
    for identity_id in annoter.indentities_ids:
        # Identity : Get identity
        identity = annoter.identities[identity_id]

        # Original identitiy images list : get copy
        original_images = copy(identity.images)
        random.shuffle(original_images)

        # Images : For every identity image
        for image in original_images[:albumentations_per_image]:
            # Next frame number : Get from identity
            next_frame_number = identity.last_frame + 1

            # Output name : 
            outputName = ReidFileInfo.toPath(identity_number=identity.number,
                                camera_number=image.camera,
                                frame_number=next_frame_number,
                                dataset=identity.dataset,)

            # Augmentate image
            if (arguments.augumentColor):
                createdPath = Augment(image.path, outputName, outputPath, transform_color)
            elif (arguments.augumentShape):
                createdPath = Augment(image.path, outputName, outputPath, transform_shape)
            else:
                createdPath = Augment(image.path, outputName, outputPath, transform_all)

            # Identity : Append image
            identity.AddImage(ImageData(path=createdPath,
                                        camera=image.camera,
                                        frame=next_frame_number))

            # Counter : Increment
            progress.update(1)

            # Check : Maximum number of created images
            processed_counter += 1
            if (processed_counter >= arguments.iterations):
                logging.info('Finished. Maximum number of created images reached!')
                sys.exit(0)


if (__name__ == '__main__'):
    # Logging : Enable
    if (__debug__ is True):
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.debug('Logging enabled!')

    # Arguments and config
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        required=True, help='Input path')
    parser.add_argument('-n', '--iterations', type=int, nargs='?', const=100, default=100,
                        required=False, help='Maximum number of created images')
    parser.add_argument('-a', '--all', action='store_true',
                        required=False, help='All images (annotated and not annotated). Defaut is only annotated.')
    parser.add_argument('-aa', '--augumentAll', action='store_true',
                        required=False, help='All image augmentations.')
    parser.add_argument('-as', '--augumentShape', action='store_true',
                        required=False, help='Process extra image shape augmentation.')
    parser.add_argument('-ac', '--augumentColor', action='store_true',
                        required=False, help='Process extra image color augmentation.')
    args = parser.parse_args()

    # Process
    Process(args.input, args)
