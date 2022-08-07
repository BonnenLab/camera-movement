# Extracting Camera Movement from Video Data

NOTE: must have access to carbonate's GPU partition to run code

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
Compile DCNv2 and NG-RANSAC with the following commands:
```
cd models/networks/DCNv2/; sbatch make.sh; cd -
cd models/ngransac/; sbatch make.sh; cd -
```
 
Then, load the weights for the pretrained model with the following commands:
```
mkdir weights
mkdir weights/rigidmask-sf
gdown https://drive.google.com/uc?id=1H2khr5nI4BrcrYMBZVxXjRBQYBcgSOkh -O ./weights/rigidmask-sf/weights.pth
```
Now that the setup is finished, you can find camera rotations by submitting a job to slurm, which calls the camera_rotations.py file and specifies the directory of video frames and an output directory for the dataframe which contains the camera rotations.

Here is an example:
```
#!/bin/bash

#SBATCH -J get_rotations
#SBATCH -p gpu
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node v100:1
#SBATCH --time=2:00:00

#Run your program
python camera_rotations.py ../../03_JPG_AllMasters/003MT ./rotations/
```
to learn more about submitting jobs to slurm, visit https://kb.iu.edu/d/awrz.

Because the algorithm takes a long time to run (about 30 minutes for every 1000 frames), you may want to perform parallel processing. To accomplish this with slurm, you can use arrays. An example is shown below:
```
#!/bin/bash

#SBATCH -J get_mult_rotations
#SBATCH -p gpu
#SBATCH -o filename_%j.txt
#SBATCH -e filename_%j.err
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node v100:4
#SBATCH --array=1-4
#SBATCH --time=16:00:00

LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p Files.txt)
echo $LINE

#Run your program
python get_video_rot.py $LINE ./rotations/
```
where Files.txt is a file that contains a list of paths to video frames.
To learn more about parallel processing with slurm, visit https://crc.ku.edu/hpc/slurm/how-to/arrays
