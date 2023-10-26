# nnSAM: Plug-and-play Segment Anything Model Improves nnUNet Performance

## Our entire code is built based on nnUNet, and you can follow the nnUNet instructions exactly.


Install nnSAM depending on your use case:

```bash
conda create -n nnsam python=3.9
conda activate nnsam
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
pip install timm
pip install git+https://github.com/Kent0n-Li/nnSAM.git
```

```bash
set MODEL_NAME=nnsam
```

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]

nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz

nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD

nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD

nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```


## How to get started?
Read these:
- [Dataset conversion](documentation/dataset_format.md)
- [Usage instructions](documentation/how_to_use_nnunet.md)

Additional information:
- [Region-based training](documentation/region_based_training.md)
- [Manual data splits](documentation/manual_data_splits.md)
- [Pretraining and finetuning](documentation/pretraining_and_finetuning.md)
- [Intensity Normalization in nnU-Net](documentation/explanation_normalization.md)
- [Manually editing nnU-Net configurations](documentation/explanation_plans_files.md)
- [Extending nnU-Net](documentation/extending_nnunet.md)
- [What is different in V2?](documentation/changelog.md)

[//]: # (- [Ignore label]&#40;documentation/ignore_label.md&#41;)



# Acknowledgements

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
