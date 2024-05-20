function newU = CompGradientMap( X, U, V, alpha, options )
%COMPGRADIENTMAP Summary of this function goes here
%   Detailed explanation goes here
eta_d = 2;
eta_u = 2;

differror = options.error;
maxIter = options.submaxIter; %   submaxIter -> the max no. iterations for subproblems, if do not set, mean run to converge; 
meanFitRatio = options.meanFitRatio;  
Lt = options.L0;   %   L0 -> a guess of Lipschitz contant
debug = options.debug;   %   debug -> whether debug or not

[mFea,K]=size(U);

%Pre-compute XVt and VVt
XVt = X * V';
VVt = V * V';

objhistory = calcObjSub(XVt, VVt, U, alpha);    %paper(11)计算初始的目标函数值
if debug == 1
    disp(['  comp_grad_map: start optimization. Sub-objective is ',num2str(objhistory)]);
end
if isempty(maxIter)
    meanFit = objhistory*10;
end

maxErr = 1;
TLU = zeros(mFea,K); % used for storing temporary U^{t+1}
zeroconst = zeros(mFea,1);
t = 0;
while maxErr > differror
    flag = 0;
    Grad = U*VVt - XVt;   % calculate value (13)
    L = Lt;
    fobj_t = calcf(XVt,VVt,U);   %calculate value the first term of (11)
    gradnorm = sum(sum(Grad .* Grad));   % the norm of gradient
    fobj = 0;
    while flag == 0
        % optimize m_L(U^t;U)   % optimization (12)
        B = U - (Grad/L);  % calculate (U^t-(1/L)*gradient)
        % First set TLU = 0 where B <= 0
        %%%TLU(B <= 0) = 0;
        % Then compute the part for which B > 0
%         if size(B,1) ~= size(TLU,1)
%             disp('bug');
%         end
        %disp(['size of B is ',num2str(size(B)),'size of TLU is ',num2str(size(TLU))]);
        if alpha == 0
            TLU = B;
        else
            for i = 1:K
                %disp(['here! i=',num2str(i)]);
                b = B(:,i);
                %disp(['size of b is ',num2str(size(b))]);
                posb = b(b>0);
                if (isempty(posb))
                    TLU(:,i) = zeroconst;
                else
                    theta = computeTheta(posb,alpha/L);   %关键是计算theta
                    TLU(:,i) = b - max(b - theta,0);
                end
                %disp(['here! i=',num2str(i)]);
            end
        end
        TLU(B < 0) = 0;
        
        % compare obj(TLU) and m_L(TLU)
        fobj = calcf(XVt,VVt,TLU);  % 
        tmpdiff = TLU - B;
        if (fobj - fobj_t - sum(sum(tmpdiff .* tmpdiff))*L*0.5 + gradnorm*0.5/L) <= 0
            flag = 1;
        else
            L = L*eta_u;
        end
    end             
    U = TLU;
    Lt = max(options.L0, L /eta_d);
    
    newobj = fobj + alpha * sum(max(U));
    objhistory = [objhistory newobj]; %#ok<AGROW>
    
    t = t+1;
    if debug == 1
        if newobj - objhistory(end-1) <= 0
            disp(['  comp_grad_map: ',num2str(t),'-th iteration completed. Sub-objective is ',num2str(newobj),'. Diff with last iteration: ',num2str(newobj - objhistory(end-1))]);
        else
            warning(['  comp_grad_map: ',num2str(t),'-th iteration completed. Sub-objective is ',num2str(newobj),'. Diff with last iteration: ',num2str(newobj - objhistory(end-1))]);
        end
    end
    
    if isempty(maxIter)
        meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
        maxErr = (meanFit-newobj)/meanFit;
    else
        maxErr = 1;
        if t >= maxIter
            maxErr = 0;
        end
    end
end

newU = U;





function obj = calcObjSub(XVt, VVt, U, alpha)  %计算（11）的值
obj = calcf(XVt, VVt, U);
obj = obj + alpha * sum(max(U));


function o = calcf(XVt, VVt, U)  % calucate the first term of (11)
% compute 1/2( Tr[UVV^{T}U^{T}] - 2Tr[UVX^{T}] ) which is more efficient
% than computing the least square
% 
o = 0.5*sum(sum((U*VVt) .* U));
o = o - sum(sum(U .* XVt));
