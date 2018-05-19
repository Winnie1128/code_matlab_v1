clear all;
close all;
clc;
addpath('../');
i=0;
Tau =[0.01:0.249:2.5];
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

pred_conjugate_gradient=[];
pred_gradient_descent=[];
for i=1:11
   
   tau=Tau(i);
   
epsilon = 0.01;
maxIter = 40;
        




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
m = size(X,1);

%--  train logistic regression method
disp('\nTraining Logistic Regression...\n')
[all_theta] = lrc.train(X, y, num_labels, maxIter, epsilon, tau);
[all_theta2] = lrc.train2(X, y, num_labels, maxIter, epsilon, tau);

%-- Save learned parameters
filename_param = 'param_mnist.mat';
if (~exist(local_param_path,'dir'))
     mkdir(local_param_path);
end
save([local_param_path,filename_param],'all_theta');
%disp(['Parameters saved to ',[local_param_path,filename_param]])
save([local_param_path,filename_param],'all_theta2');

%-- Predict for One-Vs-All on the training dataset


%testing
filename_param = 'param_mnist.mat';
load([local_param_path,filename_param]);


%-- Create X matrix
X = zeros(size(test.images,3),widthDigit*heightDigit+1);
for k=1:size(test.images,3)
    digit = test.images(:,:,k);
    X(k,:) = [1,digit(:)'];
end
y = test.labels;
m = size(X,1);


%-- Randomly select 100 data points to display
%rand_indices = randperm(m);
%sel = X(rand_indices(1:100), :);
%visu.displayDatabase(sel,widthDigit,heightDigit);


%-- Evaluate the performance of the learned method from the full testing database
%-- Predict for One-Vs-All



pred = lrc.predict(all_theta, X);
disp(['Testing Set Accuracy with gradient_desc: ',num2str(mean(double(pred == y)) * 100)])
pred_gradient_descent(i)=(mean(double(pred == y)) * 100);

pred2 = lrc.predict(all_theta2, X);
disp(['Testing Set Accuracy with conjugate_gradient: ',num2str(mean(double(pred2 == y)) * 100)])
pred_conjugate_gradient(i)=(mean(double(pred2 == y)) * 100);
i=i+1;
end
figure;
hold on;
plot(Tau,pred_gradient_descent,'-.go');
plot(Tau,pred_conjugate_gradient,'-.bx');
legend("gradient descent","conjugate gradient");
hold off;