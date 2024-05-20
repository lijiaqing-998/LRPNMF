function [Us,V,obj] = LRPNMF(X,lambda1,lambda2,num_cluster,k,options)
%% X: a cell containing all views for the data, each view of size (n_samples X n_features)
%% lambda1：control centralization;
%% lambda2: control sparseness;
%% cluster number
% options: added new options 
%   submaxIter -> the max no. iterations for subproblems, if do not set, mean run to converge; 
%   L0 -> a guess of Lipschitz contant
%% Output: U(each view); V(shared by different views); obj: objective function
eta_d = 2;
eta_u = 2;
Max_iter = options.Max_iter;
Lt = options.L0; 
L0 = options.L0;  % initize Lipschitz contant
viewNum = length(X);  % the number of views
[m,n] = size(X{1,1});  % the sample number and dimensionality of a view; 
New_X = cell(1, viewNum);
 for i = 1:viewNum
     X{1,i} = double(full(X{1,i}));
     temp = mapminmax(X{1,i},0,1);
     New_X{1,i} = temp;   
 end
%      New_X{1,1} = X{1};
%      New_X{1,9} = X{9};
 maxM = 0;
for i = 1:length(New_X)
    [m,n] = size(New_X{i});
    if m>maxM
        maxM = m;
    end
end
% New_gnd = gnd(num,:);  % the label is used to cluster (real label);
%% 
% initialize basis and encoding matrices
disp('initialize basis and encoding matrices...')
  % store Us;
  Us = cell(1,viewNum);
  Vs = cell(1,viewNum);
  Ws = cell(1,viewNum);
  Ds = cell(1,viewNum);  % similar matrix of each view
for ii = 1:viewNum
    Xi = New_X{1,ii};  % mxn
    options = [];
    options.NeighborMode = 'KNN';
    options.k = k;
    options.WeightMode = 'Binary';
    W = constructW(Xi',options);
    Ws{1,ii} = W;
    Ds{1,ii} = diag(sum(W));
   [Us{1,ii},Vs{1,ii}] = KMeansdata(Xi, num_cluster);
   Us{1,ii} = abs(Us{1,ii});
   Vs{1,ii} = abs(Vs{1,ii});
end

for i = 1:length(New_X)
    [m,~] = size(New_X{i});
    if m<maxM
      New_X{i} = padarray(New_X{i},maxM-m,0,'post');
      Us{i} = padarray(Us{i},maxM-m,0,'post');
    end
end
%% 初始化V，取所有视角的平均
Vin = zeros(size(Vs{1,1},1),size(Vs{1,1},2));
Vinit = zeros(size(Vs{1,1},1),size(Vs{1,1},2));
for j = 1:viewNum
    Vin = Vinit+Vs{1,j};
end
Vinit = (1/viewNum)*Vin;  % 初始化的V
clear Vin;
disp('Initialzing end...')


%% 开始迭代
for iter = 1:Max_iter
    %% Update U
    % 更新U逐个更新每一个视角,每一次更新的Us{i}都会参与下一次的运算     
    for i = 1:viewNum
        L = Lt;
        U_U = zeros(size(Us{1,1},1),size(Us{1,1},2));
        for j = 1:viewNum
            U_U = U_U+Us{1,j};  % 所有的视角U的和         
        end
        U_ii = (1/viewNum)*(U_U); 
        U_i = U_ii-(1/viewNum)*Us{1,i};   % U_{-v} in Eq.(7)
        eps = ((viewNum-1)/viewNum);
        for i_iter = 1:500            
            Gradi = Us{1,i}*(Vinit*Ds{1,i}*Vinit'+lambda1*eps*eps*eye(num_cluster))-New_X{1,i}*Ws{1,i}*Vinit'-lambda1*eps*U_i;
            b = Us{1,i}-(1/L)*Gradi;
            TLU = zeros(size(b,1),size(b,2));
            zeroconst = zeros(size(b,1),1);
               for k = 1:size(b,2)
                     bk = b(:,k);
                     posb = bk(bk>0);
                  if (isempty(posb))
                      TLU(:,k) = zeroconst;
                  else
                      theta = computeTheta(posb,lambda2/L);   %关键是计算theta
                      TLU(:,k) = bk - max(bk - theta,0);
                   end
               end
            TLU(b < 0) = 0;
            fobj = trace(New_X{1,i}*Ds{1,i}*New_X{1,i}'-2*New_X{1,i}*Ws{1,i}*Vinit'*TLU'+TLU*Vinit*Ds{1,i}*Vinit'*TLU')+lambda1*norm(((eps-1)/eps)*TLU-U_i,'fro')^2;
            fobj_t = trace(New_X{1,i}*Ds{1,i}*New_X{1,i}'-2*New_X{1,i}*Ws{1,i}*Vinit'*Us{1,i}'+Us{1,i}*Vinit*Ds{1,i}*Vinit'*Us{1,i}')+lambda1*norm(((eps-1)/eps)*Us{1,i}-U_i,'fro')^2;
            tmpdiff = TLU - Us{1,i}-(1/L)*Gradi; 
            gradnorm = sum(sum(Gradi .* Gradi));
               if (fobj - fobj_t - sum(sum(tmpdiff .* tmpdiff))*L*0.5 + gradnorm*0.5/L) <= 0
                   break;
               else
                    L = L*eta_u;
               end
             Us{1,i} = TLU;
             Lt = max(L0, L /eta_d); 
             obj(i_iter) = 0.5*trace(New_X{1,i}*Ds{1,i}*New_X{1,i}'-2*New_X{1,i}*Ws{1,i}*Vinit'*TLU'+TLU*Vinit*Ds{1,i}*Vinit'*TLU')+lambda1*norm(((eps-1)/eps)*TLU-U_i,'fro')^2+lambda2*sum(max(TLU));  % eqn.(9)
              if i_iter >2 
                  if  abs(obj(i_iter)-obj(i_iter-1)) < 1e-6
                 break;
                  end
              end
        end
    end
        %% Update V
        
        [~,ep2] = size(Us{1,1});           
        UXW = zeros(ep2,size(Ws{1,1},2));
        UUD = zeros(ep2,size(Ws{1,1},2));
        for ij = 1:viewNum
            UXW = UXW+Us{1,ij}'*New_X{1,ij}*Ws{1,ij};
            UUD = UUD+Us{1,ij}'*Us{1,ij}*Vinit*Ds{1,ij};              
        end
            V = Vinit.*((UXW)./max(UUD,eps));
            Vinit = V;
            
       %%  迭代结束
       temp = 0;
       TLUL = zeros(size(Us{1,1},1),size(Us{1,1},2));
       for k1 = 1:viewNum
            TLUL = TLUL+Us{1,k1};
        end
        for k1 = 1:viewNum
            temp = temp+trace(New_X{1,k1}*Ds{1,k1}*New_X{1,k1}'-2*New_X{1,k1}*Ws{1,k1}*V'*Us{1,k1}'+Us{1,k1}*V*Ds{1,k1}*V'*Us{1,k1}')+lambda1*norm(Us{1,k1}-(1/viewNum)*(TLUL-Us{1,k1}),'fro')^2+lambda2*sum(max(Us{1,k1}));
        end
        
      obj(iter) = temp;
      if iter>2
          if abs(obj(iter)-obj(iter-1))<1e-6
               break
          end
      end

end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
