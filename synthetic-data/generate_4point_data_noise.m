clear; clc;
foca=1000;
K = [foca 0 0; 0 foca 0; 0 0 1];
for ss = 1:5
    sigma = 0.2*ss;
    for i = 1:1000

        Pt0 = [4*rand(1,4)-2; 4*rand(1,4)-2; 3*rand(1,4)+3]; % 3D point
        Pt1 = Pt0;

        for m = 1:4
            R_cali(:,:,m) = rotx(10*rand-5)*roty(10*rand-5+90*(m-1))*rotz(10*rand-5); 
            T_cali(:,m) = rand(3,1);
            Pt1(:,m) = R_cali(:,:,m)'*Pt1(:,m);
            Pt_n1(:,m) = R_cali(:,:,m)*Pt1(:,m)+T_cali(:,m);
        end

        pt_n1 = Pt_n1./Pt_n1(3,:);
        pt_n1 = K*pt_n1;
        pt_n1(1:2,:) = pt_n1(1:2,:)+normrnd(0,sigma,2,4);
        pt_n1 = K\pt_n1;

        gt_rot = [];
        Ra = rotx(40*rand-20)*roty(40*rand-20)*rotz(40*rand-20);
        Ra = eye(3);
        quat1 = rotm2quat(Ra);
        gt_rot = [quat1(1); quat1(2); quat1(3); quat1(4)];
        Ta = rand(3,1);
        Pt2 = Ra*Pt1+Ta;

        for m = 1:4
            Pt_n2(:,m) = R_cali(:,:,m)*Pt2(:,m)+T_cali(:,m);
        end

        pt_n2 = Pt_n2./Pt_n2(3,:);
        pt_n2 = K*pt_n2;
        pt_n2(1:2,:) = pt_n2(1:2,:)+normrnd(0,sigma,2,4);
        pt_n2 = K\pt_n2;


        Rb = rotx(40*rand-20)*roty(40*rand-20)*rotz(40*rand-20);
        quat2 = rotm2quat(Rb);

        Tb = rand(3,1);
        gt_rot = [gt_rot; quat2(1); quat2(2); quat2(3); quat2(4); Ta; Tb];
        Pt3 = Rb*Pt1+Tb;

        for m = 1:4
            Pt_n3(:,m) = R_cali(:,:,m)*Pt3(:,m)+T_cali(:,m);
        end
        pt_n3 = Pt_n3./Pt_n3(3,:);
        pt_n3 = K*pt_n3;
        pt_n3(1:2,:) = pt_n3(1:2,:)+normrnd(0,sigma,2,4);
        pt_n3 = K\pt_n3;

        for m = 1:4
            t_cali(:,m) = -R_cali(:,:,m)'*T_cali(:,m);
        end

        s1 = t_cali(:,1)-t_cali(:,2);
        s2 = t_cali(:,1)-t_cali(:,3);
        s3 = t_cali(:,1)-t_cali(:,4);

        gt_rot = [gt_rot; t_cali(:,1)];

        for m = 1:4
            data1(:,m) = R_cali(:,:,m)'*pt_n1(:,m);
            data2(:,m) = R_cali(:,:,m)'*pt_n2(:,m);
            data3(:,m) = R_cali(:,:,m)'*pt_n3(:,m);
        end

        data = [];
        data = [data1(:); data2(:); data3(:); s1; s2; s3];
        %% data for test
        data = [data zeros(45,1)];

        %% ground truth
        depth = [Pt_n1(3,:) Pt_n2(3,:) Pt_n3(3,:)]';

        m_= num2str(i-1+1000*(ss-1),'%06d');
        n_=strcat('/path/target_params/','target_params_',m_,'.txt');
        dlmwrite(n_,data,'Delimiter','\t','precision',20)

        gt_=strcat('/path/ground_truth/','gt_',m_,'.txt');
        dlmwrite(gt_,depth,'Delimiter','\t','precision',20)

        rot_gt_=strcat('/path/rot_gt/','gt_',m_,'.txt');
        dlmwrite(rot_gt_,gt_rot,'Delimiter','\t','precision',20)



    end
end