%% -- Test --
clear
clc
close all

%% Initialization
% Models
load model_SVM
load model_SVM_gamma
load model_ridgereg
load model_RF
load model_logreg
load model_logreg_l1
load model_logreg_lambda
load model_linreg
load model_linreg_ker
load forest_results

% Testing data
load first-order/test_class
load first-order/test_reg
num_tst = length(y_tst);
num_tst_reg = zeros(5, 1);
for i_heur = 1:5
    num_tst_reg(i_heur) = length(y_tst_reg{i_heur});
end

%% Classification
% H1 - SVM with rbf kernel
% 0.4912
y1_tst_class = svmPredict(model_SVM_gamma{1}, x_tst);
err1_tst_class = sum(abs(y_tst(:, 1) - y1_tst_class)/2)/num_tst; 

% H2 - SVM with rbf kenel and regularizer
% 0.4552
y2_tst_class = svmPredict(model_SVM_gamma_C{2}, x_tst);
err2_tst_class = sum(abs(y_tst(:, 2) - y2_tst_class)/2)/num_tst; 

% H3 - SVM with rbf kenel and regularizer
% 0.4526
y3_tst_class = svmPredict(model_SVM_gamma_C{3}, x_tst);
err3_tst_class = sum(abs(y_tst(:, 3) - y3_tst_class)/2)/num_tst; 

% H4 - SVM with rbf kenel and regularizer
% 0.5163
y4_tst_class = svmPredict(model_SVM_gamma_C{4}, x_tst);
err4_tst_class = sum(abs(y_tst(:, 4) - y4_tst_class)/2)/num_tst; 

% H5 - SVM with rbf kenel and regularizer
% 0.4474
y5_tst_class = svmPredict(model_SVM_gamma_C{5}, x_tst);
err5_tst_class = sum(abs(y_tst(:, 5) - y5_tst_class)/2)/num_tst; 

%% Regression
% H1 - linear regression with polynomial kernel
% 163.1017
y1_tst_reg = linregPredict(model_linreg_ker{1}, x_tst_reg{1}*PCA_coef);
err1_tst_reg = norm(y_tst_reg{1} - y1_tst_reg)^2 / num_tst_reg(1);

% H2 - linear regression with rbf kernel
% 299.1728
y2_tst_reg = linregPredict(model_linreg_ker{2}, x_tst_reg{2});
err2_tst_reg = norm(y_tst_reg{2} - y2_tst_reg)^2 / num_tst_reg(2);

% H3 - linear regression with rbf kernel
% 194.9344
y3_tst_reg = linregPredict(model_linreg_ker{3}, x_tst_reg{3});
err3_tst_reg = norm(y_tst_reg{3} - y3_tst_reg)^2 / num_tst_reg(3);

% H4 - linear regression with rbf kernel
% 166.3987
y4_tst_reg = linregPredict(model_linreg_ker{4}, x_tst_reg{4});
err4_tst_reg = norm(y_tst_reg{4} - y4_tst_reg)^2 / num_tst_reg(4);

% H5 - linear regression with rbf kernel
% 154.0387
y5_tst_reg = linregPredict(model_linreg_ker{5}, x_tst_reg{5});
err5_tst_reg = norm(y_tst_reg{5} - y5_tst_reg)^2 / num_tst_reg(5);


