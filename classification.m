% -- Classification --
% Classify whether the theorem can be proved using the five
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
model_logreg = cell(5, 1);
err_trn_logreg = zeros(1, 5);
err_val_logreg = zeros(1, 5);
for i_heur = 1:5
    tic
    % Train the model
    model_logreg{i_heur} = logregFit(x_trn, y_trn(:, i_heur));

    % Training error
    % 0.3325, 0.3191, 0.3024, 0.3330, 0.3158
    y_trn_logreg_est = logregPredict(model_logreg{i_heur}, x_trn);
    err_trn_logreg(i_heur) = ...
        sum(abs(y_trn(:, i_heur) - y_trn_logreg_est)/2)/num_trn; 
    
    % Validation error
    % 0.5190, 0.4859, 0.5033, 0.5082, 0.5092
    y_val_logreg_est = logregPredict(model_logreg{i_heur}, x_val);
    err_val_logreg(i_heur) = ...
        sum(abs(y_val(:, i_heur) - y_val_logreg_est)/2)/num_val; 
    toc
end

% >> The errors are too high

%% Logistic regression -  with l1 regularization
model_logreg_l1 = cell(5, 1);
err_trn_logreg_l1 = zeros(1, 5);
err_val_logreg_l1 = zeros(1, 5);
for i_heur = 1:5
    tic
    % Train the model
    model_logreg_l1{i_heur} = ...
        logregFit(x_trn, y_trn(:, i_heur), 'regType', 'L1');

    % Training error
    % 0.3334, 0.3192, 0.3024, 0.3330, 0.3158
    y_trn_logreg_est_l1 = logregPredict(model_logreg_l1{i_heur}, x_trn);
    err_trn_logreg_l1(i_heur) = ...
        sum(abs(y_trn(:, i_heur) - y_trn_logreg_est_l1)/2)/num_trn; 

    % Validation error
    % 0.5203, 0.4872, 0.5052, 0.5062, 0.5098
    y_val_logreg_est_l1 = logregPredict(model_logreg_l1{i_heur}, x_val);
    err_val_logreg_l1(i_heur) = ...
        sum(abs(y_val(:, i_heur) - y_val_logreg_est_l1)/2)/num_val; 
    % 
    toc
end

% >> Sparsity does not help much

%% Logistic regression - lambda fine-tuning
num_lambda = 11;
lambda = linspace(0, 2, num_lambda);
model_logreg_lambda = cell(5, 1);
err_trn_logreg_lambda = zeros(num_lambda, 5);
err_val_logreg_lambda = zeros(num_lambda, 5);
err_trn_logreg_lambda_bst = zeros(1, 5);
err_val_logreg_lambda_bst = zeros(1, 5);

for i_heur = 1:5
    tic
    for i_lambda = 1:num_lambda
        % Train the model
        model_tmp = ...
            logregFit(x_trn, y_trn(:, i_heur), 'lambda', lambda(i_lambda));

        % Training error
        y_trn_logreg_est_lambda = logregPredict(model_tmp, x_trn);
        err_trn_logreg_lambda(i_lambda, i_heur) = ...
            sum(abs(y_trn(:, i_heur) - y_trn_logreg_est_lambda)/2)/num_trn;

        % Validation error
        y_val_logreg_est_lambda = logregPredict(model_tmp, x_val);
        err_val_logreg_lambda(i_lambda, i_heur) = ...
            sum(abs(y_val(:, i_heur) - y_val_logreg_est_lambda)/2)/num_val; 
    end
    % Best training error
    % 0.3290, 0.3158, 0.3016, 0.3320, 0.3158
    err_trn_logreg_lambda_bst(i_heur) = min(err_trn_logreg_lambda(:, i_heur)); 
    
    % Best validation error
    % 0.5160, 0.4859, 0.5026, 0.5046, 0.5029
    [err_val_logreg_lambda_bst(i_heur), i_lambda_bst] = ...
        min(err_val_logreg_lambda(:, i_heur)); 
    model_logreg_lambda{i_heur} = ...
        logregFit(x_trn, y_trn(:, i_heur), 'lambda', lambda(i_lambda_bst));
    toc
end
% >> lambda does not help much

%% SVM - simple attempt
model_SVM = cell(5, 1);
err_trn_SVM = zeros(1, 5);
err_val_SVM = zeros(1, 5);

for i_heur = 1:5
    tic
    % Train the model
    model_SVM{i_heur} = svmFit(x_trn, y_trn(:,i_heur));

    % Training error
    % 0.2568, 0.2506, 0.2478, 0.2592, 0.2395
    y_trn_SVM_est = svmPredict(model_SVM{i_heur}, x_trn);
    err_trn_SVM(i_heur) = ...
        sum(abs(y_trn(:, i_heur) - y_trn_SVM_est)/2)/num_trn; 

    % Validation error
    % 0.5101, 0.4568, 0.5121, 0.5007, 0.4886
    y_val_SVM_est = svmPredict(model_SVM{i_heur}, x_val);
    err_val_SVM(i_heur) = ...
        sum(abs(y_val(:, i_heur) - y_val_SVM_est)/2)/num_val; 
    % 0.2400
    toc
end
% >> SVM performs slightly better than logistic regression

%% SVM - with rbf kernel
num_gamma = 21;
gamma = linspace(0, 1, num_gamma);
model_SVM_gamma = cell(5, 1);
err_trn_SVM_gamma = zeros(num_gamma, 5);
err_val_SVM_gamma = zeros(num_gamma, 5);
err_trn_SVM_gamma_bst = zeros(1, 5);
err_val_SVM_gamma_bst = zeros(1, 5);

for i_heur = 1:5
    tic
    for i_gamma = 1:num_gamma
        % Train the model
        model_tmp = svmFit(x_trn, y_trn(:, i_heur), ...
            'kernel', 'rbf', 'kernelParam', gamma(i_gamma));

        % Training error
        y_trn_SVM_est_gamma = svmPredict(model_tmp, x_trn);
        err_trn_SVM_gamma(i_gamma, i_heur) = ...
            sum(abs(y_trn(:, i_heur) - y_trn_SVM_est_gamma)/2)/num_trn;

        % Validation error
        y_val_SVM_est_gamma = svmPredict(model_tmp, x_val);
        err_val_SVM_gamma(i_gamma, i_heur) = ...
            sum(abs(y_val(:, i_heur) - y_val_SVM_est_gamma)/2)/num_val; 
    end
    
    % Best training error
    % 0.1058, 0.1084, 0.1002, 0.1069, 0.0994
    err_trn_SVM_gamma_bst(i_heur) = min(err_trn_SVM_gamma(:, i_heur));
    
    % Best validation error
    %  0.4853, 0.4516, 0.4794, 0.4424, 0.4395
    [err_val_SVM_gamma_bst(i_heur), i_gamma_bst] = ...
        min(err_val_SVM_gamma(:, i_heur));
    model_SVM_gamma{i_heur} = svmFit(x_trn, y_trn(:, i_heur), ...
        'kernel', 'rbf', 'kernelParam', gamma(i_gamma_bst));
    toc
end
% >> The model starts to overfit the training data as gamma increases
% >> The kernel helps to reduce the validation error

%% SVM - with rbf kernel and regularizer
num_gamma = 21;
num_C = 9;

gamma = linspace(0, 1, num_gamma);
C = linspace(1, 10, num_C);
model_SVM_gamma = cell(5, 1);
err_trn_SVM_gamma_C = zeros(num_gamma, num_C, 5);
err_val_SVM_gamma_C = zeros(num_gamma, num_C, 5);
err_trn_SVM_gamma_bst = zeros(1, 5);
err_val_SVM_gamma_bst = zeros(1, 5);

for i_heur = 1:5
    tic
    for i_gamma = 1:num_gamma
        for i_C = 1:num_C
            % Train the model
            model_tmp = svmFit(x_trn, y_trn(:, i_heur), 'C', C(i_C),...
                'kernel', 'rbf', 'kernelParam', gamma(i_gamma));

            % Training error
            y_trn_SVM_est_gamma = svmPredict(model_tmp, x_trn);
            err_trn_SVM_gamma(i_gamma, i_C, i_heur) = ...
                sum(abs(y_trn(:, i_heur) - y_trn_SVM_est_gamma)/2)/num_trn;

            % Validation error
            y_val_SVM_est_gamma = svmPredict(model_tmp, x_val);
            err_val_SVM_gamma(i_gamma, i_C, i_heur) = ...
                sum(abs(y_val(:, i_heur) - y_val_SVM_est_gamma)/2)/num_val;
        end
    end
    
    % Best training error
    err_trn_SVM_gamma_bst(i_heur) = ...
        min(min(err_trn_SVM_gamma(:, :, i_heur)));
    
    % Best validation error
    [err_val_SVM_gamma_bst(i_heur), i_bst] = ...
        min(min(err_val_SVM_gamma(:, :, i_heur)));
    
    [i_gamma_bst, i_C_bst] = ind2sub([num_gamma, num_C], i_bst);
    model_SVM_gamma{i_heur} = svmFit(x_trn, y_trn(:, i_heur), ...
        'C', C(i_C_bst), 'kernel', 'rbf', ...
        'kernelParam', gamma(i_gamma_bst));
    toc
end
% >> The model starts to overfit the training data as gamma increases
% >> The kernel helps to reduce the validation error


%% Random forest - simple attempt
model_RF = cell(5, 1);
err_trn_RF = zeros(1, 5);
err_val_RF = zeros(1, 5);

for i_heur = 1:5
    i_heur
    tic
    % Train the model
    model_RF{i_heur} = fitForest(x_trn, y_trn(:, i_heur), 'ntrees', 50, 'bagSize', 1/5);

    % Training error
    % 0.0889, 0.0950, 0.0917, 0.0971, 0.0976
    y_trn_RF_est = predictForest(model_RF{i_heur}, x_trn);
    err_trn_RF(i_heur) = ...
        sum(abs(y_trn(:, i_heur) - y_trn_RF_est)/2)/num_trn; 

    % Validation error
    % 0.4957, 0.4666, 0.4846, 0.4794, 0.4886
    y_val_RF_est = predictForest(model_RF{i_heur}, x_val);
    err_val_RF(i_heur) = ...
        sum(abs(y_val(:, i_heur) - y_val_RF_est)/2)/num_val;
    toc
end

% >> The training is overfitting

%% Random forest - avoid overfitting
num_tree = 2;

tree = [20, 80];
model_RF_tree = cell(5, 1);
err_trn_RF_tree = zeros(num_tree, 5);
err_val_RF_tree = zeros(num_tree, 5);
err_trn_RF_tree_bst = zeros(1, 5);
err_val_RF_tree_bst = zeros(1, 5);
for i_heur = 1:5
    i_heur
    for i_tree = 1:num_tree
        tic
        % Train the model
        model_tmp = ...
            fitForest(x_trn, y_trn(:, i_heur), 'ntrees', tree(i_tree));

        % Training error
        y_trn_RF_tree_est = predictForest(model_tmp, x_trn);
        err_trn_RF_tree(i_tree, i_heur) = ...
            sum(abs(y_trn(:, i_heur) - y_trn_RF_tree_est)/2)/num_trn;

        % Validation error
        y_val_RF_tree_est = predictForest(model_tmp, x_val);
        err_val_RF_tree(i_tree, i_heur) = ...
            sum(abs(y_val(:, i_heur) - y_val_RF_tree_est)/2)/num_val;
        toc
    end
    
    % Best training error
    % 0.0407, 0.0369, 0.0345, 0.0397, 0.0381
    err_trn_RF_tree_bst(i_heur) = min(err_trn_RF_tree(:, i_heur));
    
    % Best validation error
    % 0.4879, 0.4568, 0.4850, 0.4647, 0.4706
    [err_val_RF_tree_bst(i_heur), i_tree_bst] = ...
        min(err_val_RF_tree(:, i_heur));
    model_RF_tree{i_heur} = fitForest(x_trn, y_trn, 'ntrees', tree(i_tree_bst));
end

