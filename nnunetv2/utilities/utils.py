#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import Union

from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import re


def get_identifiers_from_splitted_dataset_folder(folder: str, file_ending: str):
    files = subfiles(folder, suffix=file_ending, join=False)
    # all files must be .nii.gz and have 4 digit channel index
    crop = len(file_ending) + 5
    files = [i[:-crop] for i in files]
    # only unique image ids
    files = np.unique(files)
    return files


def create_lists_from_splitted_dataset_folder(folder: str, file_ending: str, identifiers: List[str] = None) -> List[List[str]]:
    """
    does not rely on dataset.json
    """
    if identifiers is None:
        identifiers = get_identifiers_from_splitted_dataset_folder(folder, file_ending)
    files = subfiles(folder, suffix=file_ending, join=False, sort=True)
    list_of_lists = []
    for f in identifiers:
        p = re.compile(f + "_\d\d\d\d" + file_ending)
        list_of_lists.append([join(folder, i) for i in files if p.fullmatch(i)])
    return list_of_lists