clc,clear;
close all;
clear memory;
currentFolder = pwd;
addpath(genpath(currentFolder));
load HW.mat
lambda1 = 0.01;
lambda2 = 0.01;
k = 21;%HW,ALOI-100.BRCA,BDGP
%k = 5;%Yale
truth = Y;
class_num = length(unique(truth)); % 类别个数
options.Max_iter = 500;
Max_iter = options.Max_iter;
options.L0 = 5;
replic = 10;
AC_ = zeros(1, replic);
NMI_ = zeros(1, replic);
purity_ = zeros(1, replic);
for i = 1:replic
[Us,V,obj] = LRPNMF(X,lambda1,lambda2,class_num,k,options);
V = V';
pre_label = litekmeans(V,class_num,'Replicates',20);
 result = EvaluationMetrics(truth, pre_label);
 AC_(i) = result(1);
 NMI_(i) = result(2);
 purity_(i) = result(3);
end
AC(1) = mean(AC_); AC(2) = std(AC_);
NMI(1) = mean(NMI_); NMI(2) = std(NMI_);
purity(1) = mean(purity_); purity(2) = std(purity_);


