% -- Regression --
close all
clear
clc

%% Initialization
load first-order/train_reg
load first-order/validation_reg
num_trn = zeros(5, 1);
num_val = zeros(5, 1);
for i_heur = 1:5
    num_trn(i_heur) = length(y_trn_reg{i_heur});
    num_val(i_heur) = length(y_val_reg{i_heur});
end

%% Linear regression - MLE
model_linreg = cell(5, 1);
err_trn_linreg = zeros(1, 5);
err_val_linreg = zeros(1, 5);
for i_heur = 1:5
    % Train the model
    model_linreg{i_heur} = linregFit(x_trn_reg{i_heur}, y_trn_reg{i_heur});
    
    % Training error
    % 144.2230, 177.9913, 167.7130, 163.5290, 205.1392
    y_trn_linreg_est = ...
        linregPredict(model_linreg{i_heur}, x_trn_reg{i_heur});
    err_trn_linreg(i_heur) = ...
        norm(y_trn_reg{i_heur} - y_trn_linreg_est)^2 / num_trn(i_heur);
    
    % Validation error
    % 211.2098, 224.4528, 169.4371, 205.0343, 230.6904
    y_val_linreg_est = ...
        linregPredict(model_linreg{i_heur}, x_val_reg{i_heur});
    err_val_linreg(i_heur) = ...
        norm(y_val_reg{i_heur} - y_val_linreg_est)^2 / num_val(i_heur);
end

%% Ridge regression
num_lambda = 10;

lambda = linspace(1,100, num_lambda);
model_ridgereg = cell(5, 1);
err_trn_ridgereg = zeros(num_lambda, 5);
err_val_ridgereg = zeros(num_lambda, 5);
err_trn_ridgereg_bst = zeros(1, 5);
err_val_ridgereg_bst = zeros(1, 5);
for i_heur = 1:5
    for i_lambda = 1:num_lambda
        % Train the model
        model_tmp = ...
            linregFit(x_trn_reg{i_heur}, y_trn_reg{i_heur}, ...
            'regType', 'l2', 'lambda', lambda(i_lambda));

        % Training error
        y_trn_ridgereg_est = ...
            linregPredict(model_tmp, x_trn_reg{i_heur});
        err_trn_ridgereg(i_lambda, i_heur) = ...
            norm(y_trn_reg{i_heur} - y_trn_ridgereg_est)^2 ...
            / num_trn(i_heur);

        % Validation error
        y_val_ridgereg_est = ...
            linregPredict(model_tmp, x_val_reg{i_heur});
        err_val_ridgereg(i_lambda, i_heur) = ...
            norm(y_val_reg{i_heur} - y_val_ridgereg_est)^2 ...
            / num_val(i_heur);
    end
    
    % Best training error
    % 144.3040, 178.0148, 167.7379, 163.5870, 205.1668
    err_trn_ridgereg_bst(i_heur) = min(err_trn_ridgereg(:, i_heur));
    
    % Best validation error
    % 192.4215, 175.5279, 144.7758, 171.5473, 190.6853
    [err_val_ridgereg_bst(i_heur), i_lambda_bst] = ...
        min(err_val_ridgereg(:, i_heur));
    
    model_ridgereg{i_heur} = ...
        linregFit(x_trn_reg{i_heur}, y_trn_reg{i_heur}, ...
        'regType', 'l2', 'lambda', lambda(i_lambda_bst));

end

%% Linear regression - with rbf kernel
num_sigma = 10;

sigma = linspace(1, 10, num_sigma);
model_linreg_ker = cell(5, 1);
err_trn_linreg_ker = zeros(num_sigma, 5);
err_val_linreg_ker = zeros(num_sigma, 5);
err_trn_linreg_ker_bst = zeros(1, 5);
err_val_linreg_ker_bst = zeros(1, 5);
pp.standardizeX = true;
pp.addOnes = true;

for i_heur = 1:5
    tic
    [~, C] = kmeans(x_trn_reg{i_heur}, 3);
    C_stn = standardizeCols(C);
    for i_sigma = 1:num_sigma
        % Train the model
        pp.kernelFn = @(X1, X2)kernelRbfSigma(X1, C_stn, sigma(i_sigma));
        model_tmp = linregFit(x_trn_reg{i_heur}, y_trn_reg{i_heur}, ...
        'preproc', pp,'regType', 'l2','lambda',5);
    
        % Training error
        y_trn_linreg_ker_est = ...
            linregPredict(model_tmp, x_trn_reg{i_heur});
        err_trn_linreg_ker(i_sigma, i_heur) = ...
            norm(y_trn_reg{i_heur} - y_trn_linreg_ker_est)^2 ...
            / num_trn(i_heur);

        % Validation error
        y_val_linreg_ker_est = ...
            linregPredict(model_tmp, x_val_reg{i_heur});
        err_val_linreg_ker(i_sigma, i_heur) = ...
            norm(y_val_reg{i_heur} - y_val_linreg_ker_est)^2 ...
            / num_val(i_heur);
    end
    % Best training error
    % 167.4300, 219.2664, 177.2118, 187.9455, 217.0234
    err_trn_linreg_ker_bst(i_heur) = min(err_trn_linreg_ker(:, i_heur));
    
    % Best validation error
    % 175.3013, 152.2863, 139.8237, 149.3693, 185.7987
    [err_val_linreg_ker_bst(i_heur), i_sigma_bst] = ...
        min(err_val_linreg_ker(:, i_heur));
    
    pp.kernelFn = @(X1, X2)kernelRbfSigma(X1, C_stn, sigma(i_sigma_bst));
    model_linreg_ker{i_heur} = ...
        linregFit(x_trn_reg{i_heur}, y_trn_reg{i_heur}, ...
        'preproc', pp);  
    toc
end
clear pp
%% Linear regression - with polynomial
degree = 5:5:30;

num_deg = length(degree);
model_linreg_pol = cell(5, 1);
err_trn_linreg_pol = zeros(num_deg, 5);
err_val_linreg_pol = zeros(num_deg, 5);
err_trn_linreg_pol_bst = zeros(1, 5);
err_val_linreg_pol_bst = zeros(1, 5);
pp.rescaleX = true;
pp.addOnes = true;

for i_heur = 1:5
    tic
    PCA_coef = ppca_matlab(x_trn_reg{i_heur}, 1);
    
    for i_deg = 1:num_deg
        % Train the model
        pp.poly = degree(i_deg);
        model_tmp = ...
        linregFit(x_trn_reg{i_heur}*PCA_coef, y_trn_reg{i_heur}, 'preproc', pp);
    
        % Training error
        y_trn_linreg_pol_est = ...
            linregPredict(model_tmp, x_trn_reg{i_heur}*PCA_coef);
        err_trn_linreg_pol(i_deg, i_heur) = ...
            norm(y_trn_reg{i_heur} - y_trn_linreg_pol_est)^2 ...
            / num_trn(i_heur);

        % Validation error
        y_val_linreg_pol_est = ...
            linregPredict(model_tmp, x_val_reg{i_heur}*PCA_coef);
        err_val_linreg_pol(i_deg, i_heur) = ...
            norm(y_val_reg{i_heur} - y_val_linreg_pol_est)^2 ...
            / num_val(i_heur);
    end
    % Best training error
    % 163.1017, 204.1473, 174.8212, 178.9102, 211.4954
    err_trn_linreg_pol_bst(i_heur) = min(err_trn_linreg_pol(:, i_heur));
    
    % Best validation error
    % 174.9665, 156.7986, 139.8774, 149.6898, 186.9506
    [err_val_linreg_pol_bst(i_heur), i_deg_bst] = ...
        min(err_val_linreg_pol(:, i_heur));
    
    pp.poly = degree(i_deg);
    model_linreg_pol{i_heur} = ...
        linregFit(x_trn_reg{i_heur}*PCA_coef, y_trn_reg{i_heur}, ...
        'preproc', pp);  
    toc
end

%% Exponential model
tic
% options = optimset('MaxFunEvals', 1e6);
xx_trn = [x_trn_reg{1}, ones(size(x_trn_reg{1},1),1)];
xx_val = [x_val_reg{1}, ones(size(x_val_reg{1},1),1)];
% theta_est = fminsearch(@(theta) exp_model(xx_trn, y_trn_reg{1}, theta), 10*ones(size(xx_trn,2),1));
theta_est = EM_theta(xx_trn, y_trn_reg{1}, -20, 20);
y_trn_est = 100*exp(-(xx_trn*theta_est));
err_trn = sum((y_trn_reg{1} - y_trn_est).^2)/num_trn(1)
y_val_est = 100*exp(-(xx_val*theta_est));
err_val = sum((y_val_reg{1} - y_val_est).^2)/num_trn(1)
toc