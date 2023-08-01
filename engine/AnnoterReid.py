'''
Created on 17 lis 2020

@author: spasz
'''
from __future__ import annotations
from dataclasses import dataclass, field
import os
import time
import numpy as np
import logging
from tqdm import tqdm
from engine.ReidFileInfo import ReidFileInfo
from helpers.files import IsImageFile,  GetFilename
from engine.Identity import Identity
from engine.ImageData import ImageData


@dataclass
class AnnoterReid:
    ''' Class reading all images and annotations.'''
    # Path to directory with images
    dirpath: str = field(init=True, default=None)
    # Arguments : Namespace from argparse
    args: object = field(init=True, default=None)
    # Found identities list
    identities: dict = field(init=False, default_factory=list)

    # Matrix of Identity.features x Identity.features similarities
    similarity_matrix: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        ''' Post init method.'''

        # Location : Open and parse data
        self.OpenLocation(self.dirpath)

    @property
    def indentities_ids(self) -> list:
        ''' Return list of identities ids.'''
        return list(self.identities.keys())

    @property
    def identities_count(self) -> int:
        ''' Count of identities.'''
        return len(self.identities)

    @property
    def images_count(self) -> int:
        ''' Count of images.'''
        return sum([len(identity.images) for identity in self.identities])

    @property
    def consistency_avg(self) -> float:
        ''' Return average consistency.'''
        # Consistency : Get all consistency
        consistency = [
            self.identities[identityID].consistency for identityID in self.identities]
        # Return average
        return sum(consistency) / len(consistency)

    @property
    def similarity_avg(self) -> float:
        ''' Return average similarity.'''
        return np.mean(self.similarity_matrix)

    @property
    def similarity_min(self) -> float:
        ''' Return minimum similarity.'''
        return np.min(self.similarity_matrix)

    @property
    def similarity_max(self) -> float:
        ''' Return maximum similarity.'''
        return np.max(self.similarity_matrix)

    @property
    def separation_avg(self) -> float:
        ''' Return average separation.'''
        return 1 - self.similarity_avg

    @property
    def separation_min(self) -> float:
        ''' Return minimum separation.'''
        return 1 - self.similarity_max

    @property
    def separation_max(self) -> float:
        ''' Return maximum separation.'''
        return 1 - self.similarity_min

    @staticmethod
    def ImagenameToReidInfo(imagename: str) -> int:
        ''' Convert imagename to identity number.'''
        # Filename : Get filename
        filename = GetFilename(imagename)
        # Identity : Get identity number
        identity = int(filename.split('_')[0])
        return identity

    def Remove(self, identity: Identity):
        ''' Remove identity.'''
        # Identity : Get index from keys
        index = self.indentities_ids.index(identity.number)

        # Identity : Remove identity
        self.identities.pop(identity.number)
        # Similarity matrix : Remove row and column
        self.similarity_matrix = np.delete(self.similarity_matrix, index, 0)
        self.similarity_matrix = np.delete(self.similarity_matrix, index, 1)

    def Similarities(self, identity: Identity) -> dict:
        ''' Return identity (to other identities) similarities as dict.'''
        # Row index of identity in matrix
        index = self.indentities_ids.index(identity.number)

        # Similarities dict : Create from matrix
        similarities = {}
        for index, value in enumerate(self.similarity_matrix[index, :]):
            similarities[self.indentities_ids[index]] = value

        return similarities

    def SeparationAvg(self, identity: Identity) -> float:
        ''' Return separation of identity.'''
        # Index position in matrix
        index = self.indentities_ids.index(identity.number)
        # Similarity : Get similarity
        similarity = np.mean(self.similarity_matrix[index, :])
        return 1 - similarity

    def OpenLocation(self, path: str):
        ''' Open images/annotations location.'''
        # Check : Check if path exists
        if (not os.path.exists(path)):
            logging.error('(Annoter) Path `%s` not exists!', path)
            return

        # Dirpath : Store
        self.dirpath = path

        # Excludes : List of excludes
        excludes = ['.', '..', './', '.directory']
        # Images : List all directory images.
        images = [filename for filename in os.listdir(path)
                  if (filename not in excludes) and (IsImageFile(filename))]

        # ProgressBar : Create
        progress = tqdm(total=len(images),
                        desc=f'Loading reid images',
                        unit='images',
                        leave=False)

        # Identities : Create identities
        self.identities = {}
        # Processing all files
        for index, imagename in enumerate(images):
            # Filepath : Create filepath
            imagepath = f'{path}{imagename}'

            # ReidInfo : Get reid info
            reidInfo = ReidFileInfo.FromFilename(imagename)

            # Identity : Create identity if not exists
            if (reidInfo.identity not in self.identities):
                self.identities[reidInfo.identity] = Identity(number=reidInfo.identity,
                                                              images=[],
                                                              dataset=reidInfo.dataset,
                                                              )

            # Identity : Append image
            self.identities[reidInfo.identity].images.append(ImageData(path=imagepath,
                                                                       camera=reidInfo.camera,
                                                                       frame=reidInfo.frame,
                                                                       ))

            # Progress : Update
            progress.update(1)

        # Progress : Close
        progress.close()
