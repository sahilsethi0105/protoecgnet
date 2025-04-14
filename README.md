# ProtoECGNet
Case-based interpretable deep learning for ECG classification. This code implements ProtoECGNet from the following paper: 

> [**ProtoECGNet: Case-Based Interpretable Deep Learning for Multi-Label ECG Classification**](https://arxiv.org/abs/2504.08713)<br/>
 Sahil Sethi, David Chen, Thomas Statchen, Michael C. Burkhart, Nipun Bhandari, Bashar Ramadan, & Brett Beaulieu-Jones. <b>arXiv</b>, preprint under review.

## Installation

Requirements:

- `python==3.10`

```bash
git clone https://github.com/sahilsethi0105/protoecgnet.git
cd protoecgnet
conda env create -f environment.yml
conda activate ecg_env
```

## Accessing the Data
Please install PTB-XL directly from PhysioNet [`here`](https://physionet.org/content/ptb-xl/1.0.3/). 

## Using the Repo with PTB-XL: 
- In [`ecg_utils.py`](https://github.com/sahilsethi0105/bbj_ecg/blob/main/src/ecg_utils.py), update ```DATASET_PATH``` to where you installed the dataset, and update the ```STANDARDIZATION_PATH``` directory (where you want to save preprocessing results 

**UPDATE THE BELOW FOR THIS REPO (is still for scope-mri right now): **
- [`train_test_val_creation.py`](https://github.com/sahilsethi0105/ortho_ml/blob/main/train_test_val_creation.py):
- For all files in this codebase, **your ```preprocessed_folder``` should be the final folder that contains your ```train```, ```val```, and ```test``` subfolder**
  - Note that these each contain subfolders for each MRI_ID, each with one preprocessed .npy array for each sequence in that MRI
- Note that we manually filtered out sequences that were repeated (the repeat was kept and original was removed, with the assumption that the poor image quality let to the repreat sequence) _**UPDATE WITH EXACT LIST OR Instructions**_
- _**UPDATE WITH How to Get Labels and Metadata**_
- Specify a random seed for ```random_state``` in ```stratified_split_data_by_mri_id()``` if desired (eg, 42)

## Using the Repo with MRNet
 - First, fill out the dataset research use agreement with your email [`here`](https://stanfordmlgroup.github.io/competitions/mrnet/), and you should automatically receive a link to download the data 
 - If they are no longer maintaining that website, they have also posted it [`here`](https://aimi.stanford.edu/datasets/mrnet-knee-mris)
 - After unzipping the folder, you should see ```train``` and ```valid``` subfolders
     - Our code uses the `valid` set as a hold-out test set, and dynamically selects a 120-MRI subset of the ```train``` data to monitor progresss as a validation/tuning set
     - You can adjust this by changing ```create_stratified_validation_set()``` and when it is called in ```prepare_datasets()``` in [`loader.py`](https://github.com/sahilsethi0105/ortho_ml/blob/main/loader.py)
 - Their dataset contains three binary labels: 'acl', 'meniscus', and 'abnormal'
     - Labels for each are found in the corresponding CSVs for each spit (eg, train-abnormal.csv and val-abnormal.csv for the 'abnormal' label, which is what we use for pre-training)
 - Simply **pass in the path to the base folder that contains the original ```train``` and ```valid``` subfolders for the ```preprocessed_folder``` argument** in all of the files in this codebase, and the files should all run properly
   - Make sure to adjust the other input arguments as desired (eg, ``model_type``, ``view``, etc.)
   - Arguments specific to SCOPE-MRI, such as ```sequence_type``` and ```contrast_or_no``` will be ignored, so you can set them to any valid value

## Visualizing MRIs
 - [`visualize_MRIs.ipynb`](https://github.com/sahilsethi0105/scope-mri/blob/main/visualize_MRIs.ipynb) is a Jupyter notebook for viewing the MRIs
 - For SCOPE-MRI: it provides code for viewing a target slice from all available sequences for a specific target MRI_ID
     - For ``base_path``, pass in the same directory used for the ```preprocessed_folder``` argument in the other files
 - For MRNet: it provides code for viewing a target slice from the coronal, sagittal, and axial views for a specific target MRI_ID
 - For whichever dataset you are using, pass in the corresponding ```preprocessed_folder``` as the ```base_path``` argument here

## Training, Cross-Validation, Hyperparameter Tuning, and Ensembling
- [`labrum_train.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_train.py): trains models, does cross-validation, and does inference using either MRNet data or SCOPE-MRI
- [`labrum_tune.py`](https://github.com/sahilsethi0105/scope-mri/blob/main/src/labrum_tune.py): tunes models using either MRNet data or SCOPE-MRI
- [`ensemble.ipynb`](https://github.com/sahilsethi0105/scope-mri/blob/main/ensemble.ipynb): combines models trained on separate views, and provides performance metrics at the MRI_ID level
- See [`src/README.md`](https://github.com/sahilsethi0105/scope-mri/tree/main/src#readme) for additional information

## Grad-CAM: Interpreting what a model learned
- [`grad_cam_med.py`](https://github.com/sahilsethi0105/scope-mri/blob/grad_cam/grad_cam/grad_cam_med.py): outputs Grad-CAM heat maps of what the model is "looking at" in each image (one heatmap for each slice in each MRI in the test set)
- See [`grad_cam/README.md`](https://github.com/sahilsethi0105/scope-mri/blob/main/grad_cam/README.md) for additional information

## Additional Notes
 - The commands in [`src/README.md`](https://github.com/sahilsethi0105/scope-mri/tree/main/src#readme) are for directly running the files
 - [`scripts/`](https://github.com/sahilsethi0105/scope-mri/tree/main/scripts) contains the shell scripts used to submit jobs to SLURM if using an HPC
 - The files in [`src/`](https://github.com/sahilsethi0105/scope-mri/tree/main/src) log information to TensorBoard, including learning rate, performance metrics, and the middle slice of the first MRI in the first batch for train/val/test per epoch (helps with inspecting augmentation and verifying data loading code)

  To view TensorBoard logs, after activating your conda environment (with TensorBoard installed), do:
  ```
  tensorboard --logdir=/path/to/logdir/job_name --port 6006
  ```
   - Replace ```'path/to/logdir/'``` with the actual path, and make sure to update it in ```labrum_train.py``` and ```labrum_tune.py ```
   - Use the ```'job_name'``` from when you began training/tuning
   - Then, either access ```http://localhost:6006``` in your browser
   - Or if on an HPC, ssh into the computer with a new terminal tab ```ssh -L 6006:localhost:6006 myaccount@example_computer.edu```, then access ```http://localhost:6006``` in your browser
   - You can use a different port (6006 is chosen as an example)

## Citation

Please cite both papers associated with this repository and dataset **(UPDATE with SCOPE-MRI citation after arXiv post)**:

```bibtex
@misc{sethi2025protoecgnetcasebasedinterpretabledeep,
      title={ProtoECGNet: Case-Based Interpretable Deep Learning for Multi-Label ECG Classification with Contrastive Learning}, 
      author={Sahil Sethi and David Chen and Thomas Statchen and Michael C. Burkhart and Nipun Bhandari and Bashar Ramadan and Brett Beaulieu-Jones},
      year={2025},
      eprint={2504.08713},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.08713}, 
}
```
