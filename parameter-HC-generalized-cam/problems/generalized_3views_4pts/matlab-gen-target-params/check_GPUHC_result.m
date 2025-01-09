
%> Code Description: This script reads the ground truth solution data and 
%                    the results returned from GPU-HC of the 3 views with 
%                    4 points minimal problem in order to check whether
%                    GPU-HC finds the true solution.
%
%> (c) LEMS, Brown University
%> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
%> Last Modified: Jan. 19th, 2023

clear;
close all;

%> Directories
repo_src_dir      = "/path/to/3views_4pts/";
rd_HC_results_dir = strcat(repo_src_dir, "GPU_HC_Results/");
rd_GT_data_dir    = strcat(repo_src_dir, "depth-GT-data/");

%> Read all the files in a structure from the dataset
gpuhc_results     = dir(rd_HC_results_dir);
gt_data           = dir(rd_GT_data_dir);
numOfData         = 1000;       %> Number of data
numOfVars         = 12;         %> Number of Variables

numOfFound        = 0;
GPU_HC_steps_time = zeros(numOfData, 2);
GPU_HC_best_sols  = zeros(12, numOfData);
Sols_not_found_indx = [];
for d = 1:numOfData

    %> Import data
    name = gpuhc_results(d+2).name;
    HC   = importdata(strcat(rd_HC_results_dir, name));
    GT   = importdata(strcat(rd_GT_data_dir, name));
    
    %> Extract the converged, near-real solutions, HC steps, and time.
    %  Compare the solution with the ground truth
    numOfGpuhcSols = size(HC, 1) / (numOfVars + 1);
    sols_cnter     = 2;
    prev_diff      = 1000;
    for i = 1:numOfGpuhcSols
        steps      = HC(sols_cnter-1, 1);
        time       = HC(sols_cnter-1, 2);
        gpuhc_sols = HC(sols_cnter:sols_cnter+numOfVars-1, 1);
        sols_cnter = sols_cnter + numOfVars + 1;
        
        diff = norm(gpuhc_sols - GT);
        if diff < prev_diff
            prev_diff = diff;
            sol_steps = steps;
            sols_val  = gpuhc_sols;
        end
    end
    
    GPU_HC_best_sols(:,d) = sols_val;
    
    if prev_diff < 0.1
        numOfFound = numOfFound + 1;
        GPU_HC_steps_time(d, 1) = sol_steps;
        GPU_HC_steps_time(d, 2) = time;
    else
        Sols_not_found_indx = [Sols_not_found_indx; d];
    end
    
    %> Monitor the progress
    if mod(d, 100) == 0
        fprintf(". ");
    end
end
fprintf("\n");