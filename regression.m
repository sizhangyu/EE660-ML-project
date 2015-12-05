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
    y_trn_linreg_est = linregPredict(model_linreg{i_heur}, x_trn_reg{i_heur});
    err_trn_linreg(i_heur) = ...
        norm(y_trn_reg{i_heur} - y_trn_linreg_est)^2 / num_trn(i_heur);
    
    % Validation error
    % 211.2098, 224.4528, 169.4371, 205.0343, 230.6904
    y_val_linreg_est = linregPredict(model_linreg{i_heur}, x_val_reg{i_heur});
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
            norm(y_trn_reg{i_heur} - y_trn_ridgereg_est)^2 / num_trn(i_heur);

        % Validation error
        y_val_ridgereg_est = ...
            linregPredict(model_tmp, x_val_reg{i_heur});
        err_val_ridgereg(i_lambda, i_heur) = ...
            norm(y_val_reg{i_heur} - y_val_ridgereg_est)^2 / num_val(i_heur);
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

%% 


