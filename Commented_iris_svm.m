load fisheriris

%Let's try to classify between virginica and versicolor. So find all
%species that are not 'setosa'
classKeep = ~strcmp(species,'setosa');
%%this line comppares all elements in species to setosa and puts the
%%equivalent ones in classkeep

%Get class data. %Dimensions = 100 x 4 (100 points, each 4 dimensional)
X = meas(classKeep,:); 
%%meas is just a floating point data type

%Get labels for each of the 100 points. Dimension = 100 x 1
y = species(classKeep); %In this case this is a cell array but it could just be an array of integers. For example, +1 for positive class and -1 for negative class
%%maps classkeep to species (removes non-kept values from species)

%Let us divide this data into training and test data. 
trainX = X([1:25 51:75],:); %%the X part of the training set from species (numeric data) (hard linked)
trainY = y([1:25 51:75]); %%the Y part of the training set from species (labels) (hard linked)

testX = X([26:50 76:100],:); %%the X part of the test set from species (hard linked)
testY = y([26:50 76:100]); %%the Y part of the test set from species (hard linked)
%%
% Train an SVM classifier using the data.  It is good practice to specify
% the order of the classes.

SVMModel = fitcsvm(trainX,trainY,'ClassNames',{'versicolor', 'virginica',}); %Or equivalently, if we use integers as labels, we can use fitcsvm(X,y,'ClassNames',[1, -1]). 

%Predict the label for test data. 

%predictY gives the predicted label and

%score is a two column matrix with the same numbers of row as testX. 
%Each column contains the scores of how likely test data belongs to each
%of the two classes. In this example, the first column will have score for
%class 'versicolor' and second column will have score for 'verginica'

[predictY, score] = predict(SVMModel, testX); %%outputs predicted labls(y) and (ROC?)"score"

%Number of correct classifications. Compare predicted label values (predictY) with actual label values (testY).
correctclass = sum(cellfun(@strcmp, predictY,testY)); %%self explanatory
fprintf('Linear Kernel: Correct classifications = %d / %d\n', correctclass, numel(testY))

%Score for belonging to the class 'versicolor'. This is important for
%obtaining ROC curves.
scorePositive = score(:, 1);



%%
%fitcsvm by default uses a "linear" kernel. Kernel can be specified using
%the 'KernelFunction' Name-Value pair

%Let us use the non-linear kernel called Radial Basis Function or "rbf" kernel
SVMModel = fitcsvm(trainX,trainY,'KernelFunction','rbf','ClassNames',{'versicolor', 'virginica',}); 
[predictY, score] = predict(SVMModel, testX);

correctclass = sum(cellfun(@strcmp, predictY,testY));
fprintf('RBF Kernel: Correct classifications = %d / %d\n', correctclass, numel(testY))

%%
%Both linear and non-linear kernels have a 'scale' parameter. The default value is 1 but that may not be optimal.
%fitcsvm can find the optimal value using the 'KernelScale' Name-Value pair

%Linear classifier
SVMModel = fitcsvm(trainX,trainY,'KernelFunction','linear','KernelScale','auto','ClassNames',{'versicolor', 'virginica',}); 
[predictY, score] = predict(SVMModel, testX);

correctclass = sum(cellfun(@strcmp, predictY,testY));
fprintf('\nLinear Kernel with automated scale calculation: Correct classifications = %d / %d\n', correctclass, numel(testY))

%Non-linear classifier
SVMModel = fitcsvm(trainX,trainY,'KernelFunction','rbf','KernelScale','auto','ClassNames',{'versicolor', 'virginica',}); 
[predictY, score] = predict(SVMModel, testX);

correctclass = sum(cellfun(@strcmp, predictY,testY));
fprintf('RBF Kernel with automated scale calculation: Correct classifications = %d / %d\n', correctclass, numel(testY))

%%
%Before classification, columns of predictor data (trainX) can be centered
%and scaled by the weighted columnmean and standard deviation, respectively. Usually, this improves classification results.

%This "standardization" of data is achieved by the 'Standardize' Name-value pair
SVMModel = fitcsvm(trainX,trainY,'KernelFunction','linear','Standardize',true,'ClassNames',{'versicolor', 'virginica',}); 
[predictY, score] = predict(SVMModel, testX);

correctclass = sum(cellfun(@strcmp, predictY,testY));
fprintf('\nLinear Kernel with data centering: Correct classifications = %d / %d\n', correctclass, numel(testY))

SVMModel = fitcsvm(trainX,trainY,'KernelFunction','rbf','Standardize',true,'ClassNames',{'versicolor', 'virginica',}); 
[predictY, score] = predict(SVMModel, testX);

correctclass = sum(cellfun(@strcmp, predictY,testY));
fprintf('RBF Kernel with data centering: Correct classifications = %d / %d\n', correctclass, numel(testY))