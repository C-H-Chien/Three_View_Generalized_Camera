# Minimal Solutions to Generalized Three-View Relative Pose Problem

## Introduction
This repository provides the source code of the paper: <br />
"Minimal Solutions to Generalized Three-View Relative Pose Problem" published in ICCV 2023. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Ding_Minimal_Solutions_to_Generalized_Three-View_Relative_Pose_Problem_ICCV_2023_paper.pdf)][[Supp](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Ding_Minimal_Solutions_to_ICCV_2023_supplemental.pdf)] <br />
Two minimal cases for the generalized three-view relative pose problem were proposed: *(i)* the 4-point case and *(ii)* the 6-line case, which are both solved by [GPU-HC](https://github.com/C-H-Chien/Homotopy-Continuation-Tracker-on-GPU). It has shown to be more robust than the generalized two-view relative pose (which requires 6 points for a minimal case), and are efficient, *e.g.*, ~7 (ms) for NVIDIA Titan V GPU. 

## Change Logs
[Oct. 10th, 2023] A synthetic data for the generalized three-view is provided by Yaqing.
[Mar. 02nd, 2024] Fix the number of solutions issue for the 3-view 4-points case. The tested GPU time is consistent with what was reported in the paper.
[May. 22nd, 2024] Fix the issue from the power term for parameters in the automated data reformator. This does not affect the three-view generalized problem, but could raise a bug for new minimal problems.

## Dependencies
(1) CMake 3.14 or higher <br />
(2) MAGMA 2.6.1 or higher <br />
(3) CUDA 9.0 or higher <br />
(4) cuBlas <br />
(5) openBlas <br />

## Code Usage Instructions
Expected to be updated soon.

## Reference
If you use the code, please cite our paper: <br />
```
@InProceedings{Ding_2023_ICCV,
    author    = {Ding, Yaqing and Chien, Chiang-Heng and Larsson, Viktor and \r{A}str\"om, Karl and Kimia, Benjamin},
    title     = {Minimal Solutions to Generalized Three-View Relative Pose Problem},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {8156-8164}
}
```



