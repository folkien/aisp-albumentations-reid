'''
   Single Image data with features and visuals.
'''
from dataclasses import dataclass, field
import numpy as np
import os



@dataclass
class ImageData:
    ''' Dataclass representing image and features and visuals.'''
    # Image path
    path: str = field(init=True, default=None)
    # Camera number
    camera: int = field(init=True, default=1)
    # Image/Frame number
    frame: int = field(init=True, default=1)
    # Image features
    features: np.array = field(init=True, default=None)

    @property
    def location(self) -> str:
        ''' Return location of image.'''
        return os.path.dirname(self.path)

    @property
    def name(self) -> str:
        ''' Return name of image.'''
        return os.path.basename(self.path)
