# Extracting Camera Movement from Video Data

The bulk of this algorithm comes from https://github.com/gengshan-y/rigidmask, which separates the moving foreground from the static background to accurately estimate the camera rotation matrix and translation vector between video frames. To learn more about how the algorithm accomplishes this, you can read the author's publication which can be found here: https://arxiv.org/abs/2101.03694.

# Carbonate Setup

After logging into Carbonate, use the command:
```
module load anaconda
```
to learn more about modules, visit https://kb.iu.edu/d/bcwy

once anaconda has been loaded perform the following commands to setup a virtual environment:
```
conda env create -f rigidmask.yml
conda activate rigidmask_v0
conda install -c conda-forge kornia=0.5.3 # install a compatible korna version
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.7/index.html
pip install pandas
pip install timm
pip install gdown
 ```
Then, load the weights for the pretrained model with the following commands:
```
mkdir weights
mkdir weights/rigidmask-sf
gdown https://drive.google.com/uc?id=1H2khr5nI4BrcrYMBZVxXjRBQYBcgSOkh -O ./weights/rigidmask-sf/weights.pth
```
