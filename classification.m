% -- Classification --
% Classify whether the theorem can be proved using one of the five
% heuristics
close all
clear
clc
%% Initialization
load first-order/train_class
load first-order/validation_class
num_trn = length(y_trn);
num_val = length(y_val);
%% Logistic regression - simple attempt
tic
% Train the model
model_logreg = logregFit(x_trn, y_trn);

% Training error
y_trn_logreg_est = logregPredict(model_logreg, x_trn);
err_trn_logreg = sum(abs(y_trn - y_trn_logreg_est)/2)/num_trn; 
% 0.2756

% Validation error
y_val_logreg_est = logregPredict(model_logreg, x_val);
err_val_logreg = sum(abs(y_val - y_val_logreg_est)/2)/num_val; 
% 0.2727

toc

%% Logistic regression -  with l1 regularization
tic
% Train the model
model_logreg_l1 = logregFit(x_trn, y_trn, 'regType', 'L1');

% Training error
y_trn_logreg_est_l1 = logregPredict(model_logreg_l1, x_trn);
err_trn_logreg_l1 = sum(abs(y_trn - y_trn_logreg_est_l1)/2)/num_trn; 
% 0.2749

% Validation error
y_val_logreg_est_l1 = logregPredict(model_logreg_l1, x_val);
err_val_logreg_l1 = sum(abs(y_val - y_val_logreg_est_l1)/2)/num_val; 
% 0.2681

toc

% >> Sparsity does not help much 

%% Logistic regression - lambda fine-tuning
num_lambda = 21;

tic
lambda = linspace(0, 2, num_lambda);
err_trn_logreg_lambda = zeros(num_lambda, 1);
err_val_logreg_lambda = zeros(num_lambda, 1);
for i_lambda = 1:num_lambda
    % Train the model
    model_logreg_lambda = ...
        logregFit(x_trn, y_trn, 'lambda', lambda(i_lambda));
    
    % Training error
    y_trn_logreg_est_lambda = logregPredict(model_logreg_lambda, x_trn);
    err_trn_logreg_lambda(i_lambda) = ...
        sum(abs(y_trn - y_trn_logreg_est_lambda)/2)/num_trn;
    % 0.2756; 0.2736; 0.2733; 0.2733; 0.2743; 0.2743; 0.2733; 0.2730;
    % 0.2726; 0.2723; 0.2723; 0.2723; 0.2723;(0.2720) 0.2726; 0.2723;
    % 0.2726; 0.2730; 0.2726; 0.2726; 0.2730
    
    % Validation error
    y_val_logreg_est_lambda = logregPredict(model_logreg_lambda, x_val);
    err_val_logreg_lambda(i_lambda) = ...
        sum(abs(y_val - y_val_logreg_est_lambda)/2)/num_val; 
    % 0.2727; 0.2688;(0.2681) 0.2688; 0.2701; 0.2701; 0.2695; 0.2721; 
    % 0.2714; 0.2708; 0.2714; 0.2714; 0.2714; 0.2708; 0.2727; 0.2727;
    % 0.2727; 0.2721; 0.2727; 0.2734; 0.2727
end
[err_val_logreg_lambda_bst, i_lambda_bst] = min(err_val_logreg_lambda); 
% 0.2681

toc

% >> lambda does not help much

%% SVM - simple attempt
tic
% Train the model
model_SVM = svmFit(x_trn, y_trn);

% Training error
y_trn_SVM_est = svmPredict(model_SVM, x_trn);
err_trn_SVM = sum(abs(y_trn - y_trn_SVM_est)/2)/num_trn; 
% 0.2295

% Validation error
y_val_SVM_est = svmPredict(model_SVM, x_val);
err_val_SVM = sum(abs(y_val - y_val_SVM_est)/2)/num_val; 
% 0.2400
toc

% >> SVM performs slightly better than logistic regression

%% SVM - with rbf kernel
num_gamma = 21;

tic
gamma = linspace(0, 1, num_gamma);
err_trn_SVM_gamma = zeros(num_gamma, 1);
err_val_SVM_gamma = zeros(num_gamma, 1);
for i_gamma = 1:num_gamma
    % Train the model
    model_SVM_gamma = ...
        svmFit(x_trn, y_trn, 'kernel', 'rbf', 'kernelParam', gamma(i_gamma));
    
    % Training error
    y_trn_SVM_est_gamma = svmPredict(model_SVM_gamma, x_trn);
    err_trn_SVM_gamma(i_gamma) = ...
        sum(abs(y_trn - y_trn_SVM_est_gamma)/2)/num_trn;
    % 0.2295; 0.1870; 0.1559; 0.1399; 0.1314; 0.1268; 0.1236; 0.1187;
    % 0.1180; 0.1161; 0.1128; 0.1105; 0.1089; 0.1069; 0.1066; 0.1049;
    % 0.1040; 0.1026; 0.1004; 0.0977;(0.0968)
    
    % Validation error
    y_val_SVM_est_gamma = svmPredict(model_SVM_gamma, x_val);
    err_val_SVM_gamma(i_gamma) = ...
        sum(abs(y_val - y_val_SVM_est_gamma)/2)/num_val; 
    % 0.2400; 0.2080;(0.1949) 0.2041; 0.2230; 0.2368; 0.2420; 0.2453;
    % 0.2426; 0.2368; 0.2361; 0.2217; 0.2198; 0.2224; 0.2256; 0.2256;
    % 0.2263; 0.2276; 0.2283; 0.2302; 0.2309

end
[err_val_SVM_gamma_bst, i_gamma_bst] = min(err_val_SVM_gamma);
% 0.1949

toc

% >> The model starts to overfit the training data as gamma increases
% >> The kernel helps to reduce the validation error
