
%> Code Description: Generate the true solution and the parameters of the
%                    3-views with 4-points problem. This problem arises
%                    from our collaboration with Yaqing Ding, Kale Astrom,
%                    and Viktor Larsson, as part of the minimal problem
%                    using a "generalized camera model". The code writes
%                    the ground truth data and parameters into files for
%                    the use of GPU-HC.
%
%> (c) LEMS, Brown University
%> Chiang-Heng Chien (chiang-heng_chien@brown.edu)
%> Last Modified: Jan. 19th, 2023

clear; clc;

%> Directories
repo_src_dir   = "/path/to/3views_4pts/";
wr_params_dir  = strcat(repo_src_dir, "target-params/");
wr_GT_data_dir = strcat(repo_src_dir, "depth-GT-data/");
numOfData      = 1000;

for i = 1:numOfData

    X01 = [2*rand(1,4)-4; 2*rand(1,4)-4; 3*rand(1,4)+3]; % 3D point - first camera

    R1 = rotx(10)*roty(10)*rotz(20); T1 = [0 0 1]';
    R2 = rotx(1)*roty(92)*rotz(2);   T2 = [1 0 0]';
    R3 = rotx(1)*roty(183)*rotz(2);  T3 = [0 0 -1]';
    R4 = rotx(1)*roty(274)*rotz(2);  T4 = [-1 0 0]';

    X01(:,1) = R1'*X01(:,1);
    X01(:,2) = R2'*X01(:,2);
    X01(:,3) = R3'*X01(:,3);
    X01(:,4) = R4'*X01(:,4);

    X11 = R1*X01(:,1)+T1; x11 = X11./X11(3,:);
    X21 = R2*X01(:,2)+T2; x21 = X21./X21(3,:);
    X31 = R3*X01(:,3)+T3; x31 = X31./X31(3,:);
    X41 = R4*X01(:,4)+T4; x41 = X41./X41(3,:);

    Ra = rotx(40*rand-20)*roty(40*rand-20)*rotz(40*rand-20);
    Ra = Ra';
    Ta = rand(3,1);
    X02 = Ra*X01+Ta;

    X12 = R1*X02(:,1)+T1; x12 = X12./X12(3,:);
    X22 = R2*X02(:,2)+T2; x22 = X22./X22(3,:);
    X32 = R3*X02(:,3)+T3; x32 = X32./X32(3,:);
    X42 = R4*X02(:,4)+T4; x42 = X42./X42(3,:);

    Rb = rotx(40*rand-20)*roty(40*rand-20)*rotz(40*rand-20);
    Rb = Rb';
    Tb = rand(3,1);
    X03 = Rb*X01+Tb;

    X13 = R1*X03(:,1)+T1; x13 = X13./X13(3,:);
    X23 = R2*X03(:,2)+T2; x23 = X23./X23(3,:);
    X33 = R3*X03(:,3)+T3; x33 = X33./X33(3,:);
    X43 = R4*X03(:,4)+T4; x43 = X43./X43(3,:);

    t1 = -R1'*T1; t2 = -R2'*T2; t3 = -R3'*T3; t4 = -R4'*T4;

    data = []; depth_gt = [];
    data = [R1'*x11; R2'*x21; R3'*x31; R4'*x41; R1'*x12; R2'*x22; R3'*x32; R4'*x42; R1'*x13; R2'*x23; R3'*x33; R4'*x43; ...
            t1-t2; t1-t3; t1-t4];
    depth_gt = [X11(3) X21(3) X31(3) X41(3) X12(3) X22(3) X32(3) X42(3) X13(3) X23(3) X33(3) X43(3)]';
    
    %> Write target parameters to files
    fid = fopen(strcat(wr_params_dir, num2str(i,'%05.f'),".txt"),'w');
    for g = 1:size(data, 1)
        fprintf(fid,"%.30f\t0\n", data(g,1));
    end
    fclose(fid);
    
    %> Write GT data to files
    fid = fopen(strcat(wr_GT_data_dir, num2str(i,'%05.f'),".txt"),'w');
    for g = 1:size(depth_gt, 1)
        fprintf(fid,"%.30f\n", depth_gt(g));
    end
    fclose(fid);
end



%% data for test
% data = [data zeros(45,1)];
% 
% p = data(:,1);
% depth = depth_gt;
% 
% q1 = [p(1:3) p(4:6) p(7:9) p(10:12)];
% q2 = [p(13:15) p(16:18) p(19:21) p(22:24)];
% q3 = [p(25:27) p(28:30) p(31:33) p(34:36)];
% 
% s1 = p(37:39); % t1-t2
% s2 = p(40:42); % t1-t3
% s3 = p(43:45); % t1-t4
% 
% A1 = [depth(1)*q1(:,1) depth(2)*q1(:,2) depth(3)*q1(:,3) depth(4)*q1(:,4)];
% A2 = [depth(5)*q2(:,1) depth(6)*q2(:,2) depth(7)*q2(:,3) depth(8)*q2(:,4)];
% A3 = [depth(9)*q3(:,1) depth(10)*q3(:,2) depth(11)*q3(:,3) depth(12)*q3(:,4)];
% 
% 
% eq(1) = (A1(:,1)-A1(:,2)+s1)'*(A1(:,1)-A1(:,2)+s1) - (A2(:,1)-A2(:,2)+s1)'*(A2(:,1)-A2(:,2)+s1);
% eq(2) = (A1(:,1)-A1(:,3)+s2)'*(A1(:,1)-A1(:,3)+s2) - (A2(:,1)-A2(:,3)+s2)'*(A2(:,1)-A2(:,3)+s2);
% eq(3) = (A1(:,1)-A1(:,4)+s3)'*(A1(:,1)-A1(:,4)+s3) - (A2(:,1)-A2(:,4)+s3)'*(A2(:,1)-A2(:,4)+s3);
% eq(4) = (A1(:,2)-A1(:,3)+s2-s1)'*(A1(:,2)-A1(:,3)+s2-s1) - (A2(:,2)-A2(:,3)+s2-s1)'*(A2(:,2)-A2(:,3)+s2-s1);
% eq(5) = (A1(:,2)-A1(:,4)+s3-s1)'*(A1(:,2)-A1(:,4)+s3-s1) - (A2(:,2)-A2(:,4)+s3-s1)'*(A2(:,2)-A2(:,4)+s3-s1);
% eq(6) = (A1(:,3)-A1(:,4)+s3-s2)'*(A1(:,3)-A1(:,4)+s3-s2) - (A2(:,3)-A2(:,4)+s3-s2)'*(A2(:,3)-A2(:,4)+s3-s2);
% 
% eq(7) = (A1(:,1)-A1(:,2)+s1)'*(A1(:,1)-A1(:,2)+s1) - (A3(:,1)-A3(:,2)+s1)'*(A3(:,1)-A3(:,2)+s1);
% eq(8) = (A1(:,1)-A1(:,3)+s2)'*(A1(:,1)-A1(:,3)+s2) - (A3(:,1)-A3(:,3)+s2)'*(A3(:,1)-A3(:,3)+s2);
% eq(9) = (A1(:,1)-A1(:,4)+s3)'*(A1(:,1)-A1(:,4)+s3) - (A3(:,1)-A3(:,4)+s3)'*(A3(:,1)-A3(:,4)+s3);
% eq(10) = (A1(:,2)-A1(:,3)+s2-s1)'*(A1(:,2)-A1(:,3)+s2-s1) - (A3(:,2)-A3(:,3)+s2-s1)'*(A3(:,2)-A3(:,3)+s2-s1);
% eq(11) = (A1(:,2)-A1(:,4)+s3-s1)'*(A1(:,2)-A1(:,4)+s3-s1) - (A3(:,2)-A3(:,4)+s3-s1)'*(A3(:,2)-A3(:,4)+s3-s1);
% eq(12) = (A1(:,3)-A1(:,4)+s3-s2)'*(A1(:,3)-A1(:,4)+s3-s2) - (A3(:,3)-A3(:,4)+s3-s2)'*(A3(:,3)-A3(:,4)+s3-s2);
% eq(:)