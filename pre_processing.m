% -- Pre-processing --
% Label the data according to their usages (classification or regression).
close all
clc

%% Initialization
% Load provided datasets
load first-order/alldataraw
load first-order/train
load first-order/validation
load first-order/test

% The number of data as provided
num_trn = size(train, 1);
num_val = size(validation, 1);
num_tst = size(test, 1);

%% PCA
S = svd(train(:, 1:51));
semilogy(S)
PCA_coef = ppca_matlab(train(:, 1:51), 46);

%% Classification
% Training data
x_trn = train(:, 1:51) * PCA_coef;
y_trn = zeros(num_trn, 5);
y_trn(:, 1) = sign(alldataraw(1:num_trn, 54));
y_trn(:, 2) = sign(alldataraw(1:num_trn, 55));
y_trn(:, 3) = sign(alldataraw(1:num_trn, 56));
y_trn(:, 4) = sign(alldataraw(1:num_trn, 57));
y_trn(:, 5) = sign(alldataraw(1:num_trn, 58));

% Validation data
x_val = validation(:, 1:51) * PCA_coef;
y_val = zeros(num_val, 5);
y_val(:, 1) = sign(alldataraw(num_trn+1:num_trn+num_val, 54));
y_val(:, 2) = sign(alldataraw(num_trn+1:num_trn+num_val, 55));
y_val(:, 3) = sign(alldataraw(num_trn+1:num_trn+num_val, 56));
y_val(:, 4) = sign(alldataraw(num_trn+1:num_trn+num_val, 57));
y_val(:, 5) = sign(alldataraw(num_trn+1:num_trn+num_val, 58));

% Testing data
x_tst = test(:, 1:51) * PCA_coef;
y_tst = zeros(num_tst, 5);
y_tst(:, 1) = sign(alldataraw(num_trn+num_val+1:end, 54));
y_tst(:, 2) = sign(alldataraw(num_trn+num_val+1:end, 55));
y_tst(:, 3) = sign(alldataraw(num_trn+num_val+1:end, 56));
y_tst(:, 4) = sign(alldataraw(num_trn+num_val+1:end, 57));
y_tst(:, 5) = sign(alldataraw(num_trn+num_val+1:end, 58));

%% Regression
% Training data
idx_trn = alldataraw(1:num_trn, 54:58) > 0;
x_trn_reg = cell(5, 1);
y_trn_reg = cell(5, 1);
for i = 1:5
    x_trn_reg{i} = x_trn(idx_trn(:,i), :);
    y_trn_reg{i} = alldataraw(idx_trn(:,i), 53+i);
end

% Validation data
idx_val = alldataraw(num_trn+1:num_trn+num_val, 54:58) > 0;
x_val_reg = cell(5, 1);
y_val_reg = cell(5, 1);
for i = 1:5
    x_val_reg{i} = x_val(idx_val(:,i), :);
    y_val_reg{i} = ...
        alldataraw([false(num_trn, 1); idx_val(:,i)], 53+i);
end

% Testing data
idx_tst = alldataraw(num_trn+num_val+1:end, 54:58) > 0;
x_tst_reg = cell(5, 1);
y_tst_reg = cell(5, 1);
for i = 1:5
    x_tst_reg{i} = x_tst(idx_tst(:,i), :);
    y_tst_reg{i} = alldataraw...
        ([false(num_trn+num_val, 1); idx_tst(:,i)], 53+i);
end
