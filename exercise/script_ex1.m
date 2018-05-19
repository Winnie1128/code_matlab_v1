clear all;
close all;
clc;
addpath('../');


%-- parameters
maxIter = 100;   %-- maximum number of iterations


%-- mnist database location
url = 'https://www.creatis.insa-lyon.fr/~bernard/ge/';
local_data_path = 'E:\winnie\2017-2018\3GE-S2\CLANU\code_matlab_v1/data/';
local_param_path = 'E:\winnie\2017-2018\3GE-S2\CLANU\code_matlab_v1/param/';


%-- Downlad minst database
filename_db = 'mnist.mat';
if (~exist([local_data_path,filename_db],'file'))
     tools.download(filename_db,url,local_data_path);
end


%-- Load mnist database
load([local_data_path,filename_db]);
widthDigit = size(training.images,2);
heightDigit = size(training.images,1);


%-- Perform training
num_labels = 10;          %-- 10 labels, from 0 to 9


%-- Create X matrix
X = zeros(size(training.images,3),widthDigit*heightDigit+1);
for k=1:size(training.images,3)
    digit = training.images(:,:,k);
    X(k,:) = [1,digit(:)'];
end


%-- Create y vector
y = training.labels;
[m,n] = size(X);


%-- Load pre-learned parameters
filename_param = 'param_ex1_2.mat';
load([local_param_path,filename_param]);


%-- Initialization of energy value J
J = 0;

% ====================== YOUR CODE HERE =========================
% YOU SHOULD COMPUTE THE VALUE OF THE ENERGY FUNCTION J
% ===============================================================
y = (y == 1);
h = lrc.sigmoid(X*phi');
J = -(1/m) * ( y'*log(h) + (1-y)'*log(1-h) );


disp(J)
