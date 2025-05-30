## Changlogs
[Oct. 10th, 2023] A synthetic data for the generalized three-view is provided by Yaqing. <br />
[Mar. 02nd, 2024] Fix the number of solutions issue for the 3-view 4-points case. The tested GPU time is consistent with what was reported in the paper. <br />
[May. 22nd, 2024] Fix the issue from the power term for parameters in the automated data reformator. This does not affect the three-view generalized problem, but could raise a bug for new minimal problems. This is reflected in the original GPU-HC code. <br />
[May 29th, 2025] Major code organization change: say goodbye to the messy and unstructured code. The flow follows the original GPU-HC code.