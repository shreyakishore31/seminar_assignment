% Imports both the text files as input by giving file pathway
Test_data = importdata('Assignment2-handtrainfile.txt');
Test_data2 = importdata('Assignment2-handtestfile.txt'); 

%Prediction of classes
[out,out2] = Classifier_main(Test_data,Test_data2); % Calls classifier function

% Write the outputs into two different text files
writetable(out,'Trainfile_output.txt','WriteVariableName',false,'Delimiter',' ')% Print the answer table in a text file
writetable(out2,'Testfile_output.txt','WriteVariableName',false,'Delimiter',' ')


function [out,out2] = Classifier_main(Test_data,Test_data2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Test_numbers = Test_data.data;                           %saves co-ordinates
Test_labels = Test_data.textdata;                        % saves the class names
csvwrite('Te_handtrain.csv', Test_data.data);            % write data into csv file

Test_numbers2 = Test_data2.data;
Test_labels2 = Test_data2.textdata;
csvwrite('Te_handtest.csv', Test_data2.data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Feature extraction function is called
[lf_Ver,lf_hor,rf_Ver,rf_hor,mf_Ver,mf_hor,if_Ver,if_hor,thumb_Ver,thumb_hor,palm_width] = feature_extraction(Test_numbers);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creating tables using class names and features
label_table=cell2table(Test_labels);
Classification_mat = [lf_Ver,lf_hor,rf_Ver,rf_hor,mf_Ver,mf_hor,if_Ver,if_hor,thumb_Ver,thumb_hor,palm_width
];
Class_table=array2table(Classification_mat);
Class_main= [label_table,Class_table];             %Table with classes and features

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ensemble classifier is called as a function
[trainedClassifier, validationAccuracy] = ensemble(Class_main) %Table is given as input data

yfit = trainedClassifier.predictFcn(Class_main); % predict classes using model trained in Classification learner app in Matlab
trainedClassifier.HowToPredict; 
B = cell2table(yfit); 
M = array2table(Test_numbers);                    % Create array of original co-ordinates
out = horzcat(B,M);                               % concatenate the predicted class table and co-ordinate table
writetable(out,'Trainfile_output.txt','WriteVariableName',false,'Delimiter',' ')       % Print the answer table in a text file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Feature extraction function is called again and same variables are reused.
[lf_Ver,lf_hor,rf_Ver,rf_hor,mf_Ver,mf_hor,if_Ver,if_hor,thumb_Ver,thumb_hor,palm_width] = feature_extraction(Test_numbers2);
label_table=cell2table(Test_labels2);
Classification_mat = [lf_Ver,lf_hor,rf_Ver,rf_hor,mf_Ver,mf_hor,if_Ver,if_hor,thumb_Ver,thumb_hor,palm_width
];
Class_table1=array2table(Classification_mat);
Class_main1= [label_table,Class_table1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prediction done for test set
yfit2 = trainedClassifier.predictFcn(Class_main1);
C = cell2table(yfit2);
N = array2table(Test_numbers2);
out2 = horzcat(C,N);
writetable(out2,'Testfile_output.txt','WriteVariableName',false,'Delimiter',' ')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function for feature extraction
function [lf_Ver,lf_hor,rf_Ver,rf_hor,mf_Ver,mf_hor,if_Ver,if_hor,thumb_Ver,thumb_hor,palm_width] = feature_extraction(Test_numbers)
% Feature extraction is done by using Euclidean distances between
% co-ordinates
little_finger_x = (Test_numbers(1:100,1) +Test_numbers(1:100,11))/2;     % get the midpoint of x-coordinate of little finger
little_finger_y = (Test_numbers(1:100,2) +Test_numbers(1:100,12))/2;     % get the midpoint of y-coordinate of little finger
lf_Ver= sqrt(((little_finger_x-Test_numbers(1:100,7)).^2) +((little_finger_y-Test_numbers(1:100,8)).^2));   %length
lf_hor= sqrt(((Test_numbers(1:100,5)-Test_numbers(1:100,9)).^2) +((Test_numbers(1:100,6)-Test_numbers(1:100,10)).^2)); %width

ring_finger_x = (Test_numbers(1:100,11) +Test_numbers(1:100,19))/2;  % get the midpoint of x-coordinate of ring finger
ring_finger_y = (Test_numbers(1:100,12) +Test_numbers(1:100,20))/2;  % get the midpoint of y-coordinate of ring finger
rf_Ver= sqrt(((ring_finger_x-Test_numbers(1:100,15)).^2) +((ring_finger_y-Test_numbers(1:100,16)).^2));
rf_hor= sqrt(((Test_numbers(1:100,13)-Test_numbers(1:100,17)).^2) +((Test_numbers(1:100,14)-Test_numbers(1:100,18)).^2));

middle_finger_x = (Test_numbers(1:100,19) +Test_numbers(1:100,27))/2;   % get the midpoint of x-coordinate of middle finger
middle_finger_y = (Test_numbers(1:100,20) +Test_numbers(1:100,28))/2;   % get the midpoint of y-coordinate of middle finger
mf_Ver= sqrt(((middle_finger_x-Test_numbers(1:100,23)).^2) +((middle_finger_y-Test_numbers(1:100,24)).^2));
mf_hor= sqrt(((Test_numbers(1:100,21)-Test_numbers(1:100,25)).^2) +((Test_numbers(1:100,22)-Test_numbers(1:100,26)).^2));

index_finger_x = (Test_numbers(1:100,27) +Test_numbers(1:100,35))/2;    % get the midpoint of x-coordinate of index finger
index_finger_y = (Test_numbers(1:100,28) +Test_numbers(1:100,36))/2;    % get the midpoint of y-coordinate of index finger
if_Ver= sqrt(((index_finger_x-Test_numbers(1:100,31)).^2) +((index_finger_y-Test_numbers(1:100,32)).^2));
if_hor= sqrt(((Test_numbers(1:100,29)-Test_numbers(1:100,33)).^2) +((Test_numbers(1:100,30)-Test_numbers(1:100,34)).^2));

thumb_x = (Test_numbers(1:100,37) +Test_numbers(1:100,45))/2;           % get the midpoint of x-coordinate of thumb
thumb_y = (Test_numbers(1:100,38) +Test_numbers(1:100,46))/2;           % get the midpoint of y-coordinate of thumb
thumb_Ver= sqrt(((thumb_x-Test_numbers(1:100,41)).^2) +((thumb_y-Test_numbers(1:100,42)).^2));
thumb_hor= sqrt(((Test_numbers(1:100,39)-Test_numbers(1:100,43)).^2) +((Test_numbers(1:100,40)-Test_numbers(1:100,44)).^2));

palm_width= sqrt(((Test_numbers(1:100,1)-Test_numbers(1:100,35)).^2) +((Test_numbers(1:100,2)-Test_numbers(1:100,36)).^2));

end
% Function for Ensemble Subspace Decrimnat
function [trainedClassifier, validationAccuracy] = ensemble(trainingData)
% [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% returns a trained classifier and its accuracy. This code recreates the
% classification model trained in Classification Learner app. Use the
% generated code to automate training the same model with new data, or to
% learn how to programmatically train models.
%
%  Input:
%      trainingData: a table containing the same predictor and response
%       columns as imported into the app.
%
%  Output:
%      trainedClassifier: a struct containing the trained classifier. The
%       struct contains various fields with information about the trained
%       classifier.
%
%      trainedClassifier.predictFcn: a function to make predictions on new
%       data.
%
%      validationAccuracy: a double containing the accuracy in percent. In
%       the app, the History list displays this overall accuracy score for
%       each model.
%
% Use the code to train the model with new data. To retrain your
% classifier, call the function from the command line with your original
% data or new data as the input argument trainingData.
%
% For example, to retrain a classifier trained with the original data set
% T, enter:
%   [trainedClassifier, validationAccuracy] = trainClassifier(T)
%
% To make predictions with the returned 'trainedClassifier' on new data T2,
% use
%   yfit = trainedClassifier.predictFcn(T2)
%
% T2 must be a table containing at least the same predictor columns as used
% during training. For details, enter:
%   trainedClassifier.HowToPredict

% Auto-generated by MATLAB on 20-Feb-2019 18:59:13


% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'Classification_mat1', 'Classification_mat2', 'Classification_mat3', 'Classification_mat4', 'Classification_mat5', 'Classification_mat6', 'Classification_mat7', 'Classification_mat8', 'Classification_mat9', 'Classification_mat10', 'Classification_mat11'};
predictors = inputTable(:, predictorNames);
response = inputTable.Test_labels;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false];

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
subspaceDimension = max(1, min(6, width(predictors) - 1));
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Subspace', ...
    'NumLearningCycles', 30, ...
    'Learners', 'discriminant', ...
    'NPredToSample', subspaceDimension, ...
    'ClassNames', {'bird'; 'cat'; 'chicken'; 'dog'; 'dolphin'; 'dove'; 'dragon'; 'duck'; 'elephant'; 'fly'; 'fox'; 'great_egret'; 'leopard'; 'lion'; 'pitbull'; 'salmon'; 'seahorse'; 'sheep'; 'tiger'; 'white_tiger'});

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'Classification_mat1', 'Classification_mat2', 'Classification_mat3', 'Classification_mat4', 'Classification_mat5', 'Classification_mat6', 'Classification_mat7', 'Classification_mat8', 'Classification_mat9', 'Classification_mat10', 'Classification_mat11'};
trainedClassifier.ClassificationEnsemble = classificationEnsemble;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2018a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'Classification_mat1', 'Classification_mat2', 'Classification_mat3', 'Classification_mat4', 'Classification_mat5', 'Classification_mat6', 'Classification_mat7', 'Classification_mat8', 'Classification_mat9', 'Classification_mat10', 'Classification_mat11'};
predictors = inputTable(:, predictorNames);
response = inputTable.Test_labels;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
end


end
