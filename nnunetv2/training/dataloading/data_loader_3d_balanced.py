
from typing import Union, Tuple

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D

class nnUNetDataLoader3D_Balanced(nnUNetDataLoader3D):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager,  # LabelManager,
                 classes,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager, oversample_foreground_percent,
                         sampling_probabilities, pad_sides, probabilistic_oversampling)
        self.indices = []
        for class_id in classes:
            class_indices = [data_id for data_id in data.keys() if data_id.startswith(class_id)]
            self.indices.append(class_indices)

    def get_indices(self):
        indices = []
        if self.infinite:
            for class_indices in self.indices:
                indices = [*indices,
                           *np.random.choice(class_indices, int(self.batch_size / len(self.indices)), replace=True,
                                             p=self.sampling_probabilities)]
            np.random.shuffle(indices)
            return indices
