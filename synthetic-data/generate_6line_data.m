clear; clc;

for i = 1:1000

    pt0 = [4*rand(1,12)-2; 4*rand(1,12)-2; 3*rand(1,12)+3]; % 3D point for first camera
    pt1 = reshape(pt0,[3,2,6]);

    for m = 1:6
        R_cali(:,:,m) = rotx(10*rand-5)*roty(10*rand-5+60*(m-1))*rotz(10*rand-5); 
        T_cali(:,m) = rand(3,1);
    end

    for m = 1:6
        pt1(:,:,m) = R_cali(:,:,m)'*pt1(:,:,m);
        pt_n1(:,:,m) = R_cali(:,:,m)*pt1(:,:,m)+T_cali(:,m);
        pt_n1(:,:,m) = pt_n1(:,:,m)./pt_n1(3,:,m);
        L1(1:3,m) = cross(pt_n1(:,1,m),pt_n1(:,2,m));
        L1(1:3,m) = L1(1:3,m)./norm(L1(1:3,m));
    end

    gt = [];

    Ra = rotx(40*rand-20)*roty(40*rand-20)*rotz(40*rand-20);
    quat1 = rotm2quat(Ra);
    gt = [quat1(2)/quat1(1); quat1(3)/quat1(1); quat1(4)/quat1(1)];
    Ra = Ra';
    Ta = rand(3,1);
    
    for m = 1:6
        pt2(:,:,m) = Ra*pt1(:,:,m)+Ta;
        pt_n2(:,:,m) = R_cali(:,:,m)*pt2(:,:,m)+T_cali(:,m);
        pt_n2(:,:,m) = pt_n2(:,:,m)./pt_n2(3,:,m);
        L2(1:3,m) = cross(pt_n2(:,1,m),pt_n2(:,2,m));
        L2(1:3,m) = L2(1:3,m)./norm(L2(1:3,m));
    end

    Rb = rotx(40*rand-20)*roty(40*rand-20)*rotz(40*rand-20);
    quat2 = rotm2quat(Rb);
    gt = [gt; quat2(2)/quat2(1); quat2(3)/quat2(1); quat2(4)/quat2(1)];
    Rb = Rb';
    Tb = rand(3,1);
    gt = [gt; Ta; Tb];

    for m = 1:6
        pt3(:,:,m) = Rb*pt1(:,:,m)+Tb;
        pt_n3(:,:,m) = R_cali(:,:,m)*pt3(:,:,m)+T_cali(:,m);
        pt_n3(:,:,m) = pt_n3(:,:,m)./pt_n3(3,:,m);
        L3(1:3,m) = cross(pt_n3(:,1,m),pt_n3(:,2,m));
        L3(1:3,m) = L3(1:3,m)./norm(L3(1:3,m));
    end


    P0 = [eye(3) [0 0 0]'; 0 0 0 1];
    Pa = [Ra Ta; 0 0 0 1]; 
    Pb = [Rb Tb; 0 0 0 1];

    L1(4,:) = 0; L2(4,:) = 0; L3(4,:) = 0;

    N1 = []; N2 = []; N3 = [];
    for m = 1:6
        P_cali(:,:,m) = [R_cali(:,:,m) T_cali(:,m); 0 0 0 1];
        Mo(:,:,m) = [P_cali(:,:,m)'*L1(:,m) P_cali(:,:,m)'*L2(:,m) P_cali(:,:,m)'*L3(:,m)];
        M(:,:,m) = Mo(1:4,:,m);
        M(:,:,m) = M(:,:,m)./M(3,:,m);
        N1 = [N1 M(1:2,1,m)];
        N2 = [N2 M(1:2,2,m)];
        N3 = [N3 M(1:2,3,m)];
    end

    data = [];
    data = [N1(:); N2(:); N3(:)];
    data = [data zeros(36,1)];

    m_= num2str(i-1,'%06d');
    n_=strcat('/path/target_params/','target_params_',m_,'.txt');
    dlmwrite(n_,data,'Delimiter','\t','precision',20)


    gt_=strcat('/path/ground_truth/','gt_',m_,'.txt');
    dlmwrite(gt_,gt,'Delimiter','\t','precision',20)


    trans = [];
    trans = [M(4,:,1) M(4,:,2) M(4,:,3) M(4,:,4) M(4,:,5) M(4,:,6)]';

    tra=strcat('/path/trans/','trans_',m_,'.txt');
    dlmwrite(tra,trans,'Delimiter','\t','precision',20)

end
