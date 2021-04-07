% The code is written by Jie Wen
% If you have any questions to the code, please send email to
% jiewen_pr@126.com
% Please cite the following papers if you use the code:
% Jie Wen, Zheng Zhang, Zhao Zhang, Lei Zhu, Lunke Fei, Bob Zhang, Yong Xu, 
% Unified Tensor Framework for Incomplete Multi-view Clustering and Missing-view Inferring, 
% AAAI Conference on Artificial Intelligence, 2021.
function [Z,E,B,obj] = IMVTSC(Z_ini,X,Ne,W,lambda1,lambda2,lambda3,miu,rho,max_iter)
% figure;
Z = Z_ini;
P = Z;
clear Z_ini
for iv = 1:length(X)
    E{iv} = zeros(size(X{iv},1),Ne(iv));
    B{iv} = zeros(size(X{iv}));
    A{iv} = zeros(size(Z{iv}));
    C{iv} = zeros(size(X{iv}));
end

nv = length(X);
Nsamp = size(X{1},2);

for iter = 1:max_iter
    Z_pre = Z;
    E_pre = E;
    B_pre = B;
    P_pre = P;
    % --------- update Z ------%
    SumZ = 0;
    for iv = 1:nv
        SumZ = SumZ+Z{iv};
    end
    for iv = 1:nv
        XWEW = X{iv}+E{iv}*W{iv}';
        linshi1 = (2*lambda3*((nv-1)/nv)^2+miu)*eye(Nsamp)+miu*(XWEW'*XWEW);
        linshi2 = 2*lambda3*(nv-1)/(nv*nv)*(SumZ-Z{iv})+miu*P{iv}-A{iv}+XWEW'*(miu*(XWEW-B{iv})+C{iv});
        linshi = linshi1\linshi2;
        Z1 = zeros(size(linshi));
        for is = 1:size(linshi,1)
           ind_c = 1:size(linshi,1);
           ind_c(is) = [];
           Z1(is,ind_c) = EProjSimplex_new(linshi(is,ind_c));
        end
        Z{iv} = Z1;
    end
    clear XWEW linshi1 linshi2 linshi
    % ----------------- P --------------%
    Z_tensor = cat(3, Z{:,:});
    A_tensor = cat(3, A{:,:});
    Zv = Z_tensor(:);
    Av = A_tensor(:);
    [Pv, objV] = wshrinkObj(Zv + 1/miu*Av,1/miu,[Nsamp,Nsamp,nv],0,1);
    P_tensor = reshape(Pv, [Nsamp,Nsamp,nv]);
    for iv = 1:nv
        P{iv} = P_tensor(:,:,iv);
        % -------- E{iv} B{iv} ------- %
        WWZ = W{iv}'-W{iv}'*Z{iv};
        E{iv} = ((miu*(X{iv}*Z{iv}+B{iv}-X{iv})-C{iv})*WWZ')/(2*lambda1*eye(size(W{iv},2))+miu*WWZ*WWZ');
        linshi1 = X{iv}+E{iv}*W{iv}';
        temp1 = linshi1-linshi1*Z{iv}+1/miu*C{iv};
        temp2 = lambda2/miu;
        B{iv} = max(0,temp1-temp2) + min(0,temp1+temp2);
        % -------- A{iv} C{iv} ------%
        A{iv} = A{iv}+miu*(Z{iv}-P{iv});
        C{iv} = C{iv}+miu*(linshi1-linshi1*Z{iv}-B{iv});
        clear temp1 temp2 linshi1 WWZ
    end
    clear Z_tensor A_tensor Zv Av Pv P_tensor
    
    miu = min(miu*rho, 1e10);
    diff_Z = 0;
    diff_E = 0;
    diff_B = 0;
    diff_P = 0;
    leqm1  = 0;
    %% check convergence
    for iv = 1:nv
        linshi1 = X{iv}+E{iv}*W{iv}';
        Rec_error = linshi1-linshi1*Z{iv}-B{iv};
        leqm1 = max(leqm1,max(abs(Rec_error(:))));
        leq{iv} = Z{iv}-P{iv};
        diff_Z = max(diff_Z,max(abs(Z{iv}(:)-Z_pre{iv}(:))));
        diff_E = max(diff_E,max(abs(E{iv}(:)-E_pre{iv}(:))));
        diff_B = max(diff_B,max(abs(B{iv}(:)-B_pre{iv}(:))));
        diff_P = max(diff_P,max(abs(P{iv}(:)-P_pre{iv}(:))));        
    end
    leqm = cat(3, leq{:,:});
    leqm2 = max(abs(leqm(:)));
    clear leq leqm Rec_error_tensor Rec_error

%     err = max([leqm1,leqm2,diff_Z,diff_E,diff_B,diff_P]);
    err = max([leqm1,leqm2]);
    fprintf('iter = %d, miu = %.3f, difZ = %.d, err = %.8f,Rec_error=%d\n'...
            , iter,miu,diff_Z,err,leqm1+leqm2);
    obj(iter) = err;  
    
%     Rec_sample = reshape(E{2}(:,1),50,40);
%     imshow(Rec_sample,[]);title('E')
    if err < 1e-6
        iter
        break;
    end
    
    
end
end
