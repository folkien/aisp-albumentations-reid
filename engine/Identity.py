'''
    Single ReID identity with all images.

'''
from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
import shutil
from engine.ReidFileInfo import ReidDataset, ReidFileInfo
from engine.ImageData import ImageData
import numpy as np


@dataclass
class Identity:
    ''' Class representing identity with all images.'''
    # Identity number :
    number: int = field(init=True, default=None)
    # Identity ImageData list
    images: list = field(init=True, default=None)
    # Identity dataset type
    dataset: ReidDataset = field(init=True, default=ReidDataset.AispReid)

    def __post_init__(self):
        ''' Post init.'''
        # Check : Invalid images list
        if (self.images is None):
            self.images = []

        # Similarity matrix : Default is None

    @property
    def image(self) -> ImageData:
        ''' Return first image.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        return self.images[0]

    @property
    def images_count(self) -> int:
        ''' Count of images.'''
        return len(self.images)

    @property
    def last_frame(self) -> int:
        ''' Return last frame number.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return 0

        # Maximum frame number
        return  max([image.frame for image in self.images])

    @cached_property
    def hue(self) -> float:
        ''' Return average hue of all images.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get hue
        hue = [image.visuals.hue for image in self.images]
        return np.mean(hue)

    @cached_property
    def brightness(self) -> float:
        ''' Return average brightness of all images.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get brightness
        brightness = [image.visuals.brightness for image in self.images]
        return np.mean(brightness)

    @cached_property
    def saturation(self) -> float:
        ''' Return average saturation of all images.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get saturation
        saturation = [image.visuals.saturation for image in self.images]
        return np.mean(saturation)

    @cached_property
    def imhash(self) -> float:
        ''' Return average imhash of all images.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get imhash
        imhash = [image.visuals.dhash for image in self.images]
        return np.mean(imhash)

    @cached_property
    def features(self) -> np.array:
        ''' Return average features of all np.arrays.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get features
        features = [image.features for image in self.images]
        # Get average
        average = np.median(features, axis=0)

        return average


    def AddImage(self, image: ImageData):
        ''' Add image to identity.'''
        # Check : Image is not None
        if (image is None):
            return None

        # Add image
        self.images.append(image)
        sorted(self.images, key=lambda image: image.frame)
