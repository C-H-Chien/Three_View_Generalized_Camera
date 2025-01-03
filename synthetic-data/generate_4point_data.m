%> This code generates synthetic data for the problem of generalized
%  three-view relative pose estimation using 4 points. The generalized
%  camera (or camera rig) is set to have four cameras whose rotations and
%  translations with respect to the rig coordinate are random. Each 3D
%  point is observed by individual camera of the rig, i.e., one camera
%  views only one point. In practice, the relative rotations and translations 
%  of the camera coordinate with respect to the rig coordinate is determined
%  by the point which camera views. 

clear; clc;

%> Directories
src_dir             = "/path/to/the/folder/to/which/files/are/written/";
wr_params_dir       = strcat(src_dir, "target_params_");
wr_GT_data_dir      = strcat(src_dir, "gt_");
wr_GT_rot_dir       = strcat(src_dir, "rot_");
wr_GT_transl_dir    = strcat(src_dir, "transl_");


Num_Of_Sets = 500;
A = zeros(3,3,4);
tau = zeros(3,4);

i = 1;
while i < Num_Of_Sets

    %> Set the ground-truth relative rotations and translations
    R21 = rotx(40*rand-20)*roty(40*rand-20)*rotz(40*rand-20);
    T21 = rand(3,1);
    R31 = rotx(40*rand-20)*roty(40*rand-20)*rotz(40*rand-20);
    T31 = rand(3,1);

    %> Rotation (A) and translation (tau) transforming 3D points from rig
    %  coordinate to camera coordinate.
    for m = 1:4
        A(:,:,m) = rotx(10*rand-5)*roty(10*rand-5+90*(m-1))*rotz(10*rand-5); 
        tau(:,m) = rand(3,1);
    end

    %> Four 3D points in the first, second, and third rig coordinates
    Pt_rig1 = [4*rand(1,4)-2; 4*rand(1,4)-2; 3*rand(1,4)+3];
    Pt_rig2 = R21*Pt_rig1 + T21;
    Pt_rig3 = R31*Pt_rig1 + T31;

    %> Transform the point in the rig coordinate to the camera coordinate
    for m = 1:4
        Pt_cams_of_rig1(:,m) = A(:,:,m)*Pt_rig1(:,m) + tau(:,m);
        Pt_cams_of_rig2(:,m) = A(:,:,m)*Pt_rig2(:,m) + tau(:,m);
        Pt_cams_of_rig3(:,m) = A(:,:,m)*Pt_rig3(:,m) + tau(:,m);
    end

    %> Normalize points
    pt_n1 = Pt_cams_of_rig1 ./ Pt_cams_of_rig1(3,:);
    pt_n2 = Pt_cams_of_rig2 ./ Pt_cams_of_rig2(3,:);
    pt_n3 = Pt_cams_of_rig3 ./ Pt_cams_of_rig3(3,:);

    %> ground truth depths for the 12x12 formulation
    depth = [Pt_cams_of_rig1(3,:) Pt_cams_of_rig2(3,:) Pt_cams_of_rig3(3,:)]';

    %> ground truth relative rotations
    gt_rot = [R21; R31];

    %> ground truth relative translations
    gt_transl = [T21'; T31'];

    if min(depth)>0

        for m = 1:4
            tau(:,m) = -A(:,:,m)'*tau(:,m);
        end

        s1 = tau(:,1)-tau(:,2); 
        s2 = tau(:,1)-tau(:,3); 
        s3 = tau(:,1)-tau(:,4);

        for m = 1:4
            data1(:,m) = A(:,:,m)'*pt_n1(:,m);
            data2(:,m) = A(:,:,m)'*pt_n2(:,m);
            data3(:,m) = A(:,:,m)'*pt_n3(:,m);
        end

        data = [];
        data = [data1(:); data2(:); data3(:); s1; s2; s3];
        %% data for test
        data = [data zeros(45,1)];

        m_= num2str(i-1,'%06d');

        %> Target parameters
        n_ = strcat(wr_params_dir, m_, '.txt');
        dlmwrite(n_, data, 'Delimiter', '\t', 'precision', 20);

        %> Ground Truths Depths
        gt_=strcat(wr_GT_data_dir, m_, '.txt');
        dlmwrite(gt_, depth, 'Delimiter', '\t', 'precision', 20);

        %> Ground Truth Relative Rotation Matrices
        rot_gt_ = strcat(wr_GT_rot_dir, m_, '.txt');
        dlmwrite(rot_gt_, gt_rot, 'Delimiter', '\t', 'precision', 20);

        %> Ground Truth Relative Translation Vectors
        transl_gt_ = strcat(wr_GT_transl_dir, m_, '.txt');
        dlmwrite(transl_gt_, gt_transl, 'Delimiter', '\t', 'precision', 20);

        i = i + 1;
    end
end
