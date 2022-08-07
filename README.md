# Extracting Camera Movement from Video Data

The bulk of this algorithm comes from https://github.com/gengshan-y/rigidmask, which separates the moving foreground from the static background to accurately estimate the camera rotation matrix and translation vector between video frames. To learn more about how the algorithm accomplishes this, you can read the author's publication which can be found here: https://arxiv.org/abs/2101.03694.
