% Aslan Mehrabi  
% Document Classification of New York Times Articles. 
% Classification Methods: Neural Networks // SVM (Gaussian Kernel / Linear Kernal / Sigmoid Kernel) 




% v: 10 fold cross-validation
% d: 5 degree in kernal function
% c cost
% s svm type, 0: multi class classification
% t kernal type 2: radial basis function / 3 sigmoid
% q quite mode, no output

%kernel RBF       
%option = '-t 2 -c 100000 -v 4 -q -d 4 -s 0 ';
%c==1  26.5   cross validation accuracy
% c==0.01 26.5
% c=100  62.7  
% c= 10  53.4
% c=0.1  26.15
% c=1000  64.4
% c= 10000  63.6
% c=100000 63.6
% c=500 65.1    **  // bagging 64.65 / ncrs 66.2 


%option = '-t 2 -c 500 -q -d 4 -s 0 ';
% bedune cross validation, ba andaze giri validation
% c=500 66.995
% c=10 56.45
% c=1000 66.9
% c= 10000  65.3

%--
%kernel linear
%option = '-t 0 -c 100000 -v 4 -q -d 2 -s 0 ';
% taghire d farghi nadasht, 2,4,10 test shod
% c 0.001 d4  56.4  d2 56.4 cross 58.55
% c0.01 d4   65 d2 65
% c0.1  d4 63.8  d2 63.8  d10 63.8   cross 66.05 ** bagging 62.9 //ncr64.6
% c1 d4  63.6
% c10 d4 63.6  d2 63.6  cross 65.35
% c100 d4 63.6
% c1000 d4 63.6   cross 65.35

%behtar budane gheire cross => bishtar budane train data

% kernal sigmoid
%option = '-t 3 -c 0.1 -v 4 -q -d 2 -s 0 ';
% c 0.1 26.15  ncross 23.5
% c 10 46.2     ncross 50.4
% c 100 59.75  ncross 61.55
% c1000 65.1    ncorss 66.75  **  bagging => 42.8 // ncross bagging 44.3
% c10000 63.6  ncross 65.35

option = '-t 2 -c 500 -v 4  -q -d 4 -s 0 ';

classifierSVM = svmtrain (trainLabel , trainFeatures , option)
%fprintf('salam\n');
 [Output,accuracy,~] = svmpredict (validLabel , validFeatures ,classifierSVM);
 
 
 
 % ncross(valid data) tf_idf SVM:
 
 fprintf('RBF \n');
option = '-t 2 -c 500   -q -d 4 -s 0 ';
classifierSVM = svmtrain (trainLabel , tf_idf, option)
[Output,accuracy,~] = svmpredict (validLabel , validFeatures ,classifierSVM);

fprintf('linear \n')
option = '-t 0 -c 0.1   -q -d 4 -s 0 ';
classifierSVM = svmtrain (trainLabel , tf_idf, option)
[Output,accuracy,~] = svmpredict (validLabel , validFeatures ,classifierSVM);

fprintf('sig\n')
option = '-t 3 -c 1000   -q -d 4 -s 0 ';
classifierSVM = svmtrain (trainLabel , tf_idf, option)
[Output,accuracy,~] = svmpredict (validLabel , validFeatures ,classifierSVM);
 
 
 
 
 
 
 
 
 
 
 %  whiten() => tozih ke nemishe
 
 
     % trainFeature
 %[a,b,c]=svds(trainFeatures,500);
% rbf reduction => 500:56.05 / 100 : 52.8  / 20:50.45 / 10:45.1 / 5:42.2 / 1:31.1 
% sigmoid reduc => /500:56.15 / 100:53.05 / 10:45 / 5:42.25 / 1:
% linear reduc =>  500: 26.15 /  100:26.15 / 10: 26.15 /    => tojih ke chera bade


fprintf('RBF 500\n');
[a,b,c]=svds(trainFeatures,500);
option = '-t 2 -c 500 -v 4  -q -d 4 -s 0 ';
classifierSVM = svmtrain (trainLabel , a , option)

fprintf('linear 500\n')
option = '-t 0 -c 0.1 -v 4  -q -d 4 -s 0 ';
classifierSVM = svmtrain (trainLabel , a , option)

fprintf('linear 100\n')
[a,b,c]=svds(trainFeatures,1);
classifierSVM = svmtrain (trainLabel , a , option)



 [Output,accuracy,~] = svmpredict (validLabel , validFeatures ,classifierSVM);


    % using tf_idf  for dimention reduction
    % rbf     500:56.25 /100: 61.65 / 10:50.35 /5:47.55/1:28
    % linear  500:26.15 /100: 26.15  / 10:26.15 /5:26.15/1:25.15
    %sigmoid  500:55.6 /100: 61.7 / 10:50.25 / 47.55/1:28

    
[a,b,c]=svds(tf_idf,1);

fprintf('RBF\n');
option = '-t 2 -c 500 -v 4  -q -d 4 -s 0 ';
classifierSVM = svmtrain (trainLabel , a , option)
option = '-t 0 -c 0.1 -v 4  -q -d 4 -s 0 ';
fprintf('linear')
classifierSVM = svmtrain (trainLabel , a , option)
fprintf('sigmoid')
    
option = '-t 3 -c 1000 -v 4  -q -d 4 -s 0 ';
classifierSVM = svmtrain (trainLabel , a , option)




% ncorss(validation) reduction tf_idf
% rbf : 
% linear : 
% sigmois :



dim=500;
fprintf('dim: %d\n',dim);
[a,b,c]=svds([tf_idf;tf_idfValid],dim);
tf_idfRed500= a(1:2000,:);
tf_idfValidRed500 = a(2001:4000,:);

fprintf('dim: %d\n',dim);
[a,b,c]=svds([trainFeatures;validFeatures],dim);
trainFeatureRed500= a(1:2000,:);
validFeatureRed500 = a(2001:4000,:);




% ncorss(validation) reduction tf_idf
% rbf : 500:60.3 / 100:61.15 / 10:52.35 / 5:49.6 / 1: 30
% linear : 500:23.5 / 100:23.15 / 10:23.5 / 5:23.5 /1:23.5
% sigmois : 500 :61 / 100:61.2 / 10:52.3 / 5:49.65 /1:30




dim=1;
fprintf('dim: %d\n',dim);
[a,b,c]=svds([tf_idf;tf_idfValid],dim);
tf_idfRed= a(1:2000,:);
tf_idfValidRed = a(2001:4000,:);


fprintf('RBF\n');
option = '-t 2 -c 500  -q -d 4 -s 0 ';
classifierSVM = svmtrain (trainLabel , tf_idfRed , option)
[Output,accuracy,~] = svmpredict (validLabel , tf_idfValidRed,classifierSVM);
option = '-t 0 -c 0.1  -q -d 4 -s 0 ';
fprintf('linear\n')
classifierSVM = svmtrain (trainLabel , tf_idfRed , option)
[Output,accuracy,~] = svmpredict (validLabel , tf_idfValidRed,classifierSVM);
fprintf('sigmoid\n')
    
option = '-t 3 -c 1000  -q -d 4 -s 0 ';
classifierSVM = svmtrain (trainLabel , tf_idfRed , option)
[Output,accuracy,~] = svmpredict (validLabel , tf_idfValidRed,classifierSVM);



%%%%%%


% ncross (validation) reduction trainFeature

% rbf : 500:55.55 / 100:52.8/ 10:44.8/ 5:41.9 / 1:34
% linear : 500:23.5 / 100:23.5/ 10:23.5 / 5:23.5 /1:23.5
% sigmois : 500:56.3 :/ 100:52.95/ 10:44.9 / 5:41.9/1:34


dim=1;
fprintf('dim: %d\n',dim);
[a,b,c]=svds([trainFeatures;validFeatures],dim);
tf_idfRed= a(1:2000,:);
tf_idfValidRed = a(2001:4000,:);




fprintf('RBF\n');
option = '-t 2 -c 500  -q -d 4 -s 0 ';
classifierSVM = svmtrain (trainLabel , tf_idfRed , option)
[Output,accuracy,~] = svmpredict (validLabel , tf_idfValidRed,classifierSVM);
option = '-t 0 -c 0.1  -q -d 4 -s 0 ';
fprintf('linear\n')
classifierSVM = svmtrain (trainLabel , tf_idfRed , option)
[Output,accuracy,~] = svmpredict (validLabel , tf_idfValidRed,classifierSVM);
fprintf('sigmoid\n')
    
option = '-t 3 -c 1000  -q -d 4 -s 0 ';
classifierSVM = svmtrain (trainLabel , tf_idfRed , option)
[Output,accuracy,~] = svmpredict (validLabel , tf_idfValidRed,classifierSVM);






% ++++++++++++
%   1 vs All
% ++++++++++++

trainLabel1vsAll = zeros(4,numLine);
validLabel1vsAll = zeros(4,numLine);
for i = 0:3
   for j = 1:numLine
      trainLabel1vsAll(i+1,j) = trainLabel(j)==i;
      validLabel1vsAll(i+1,j) = validLabel(j)==i;
   end
end










%num Hiddent layer NN:
% 2:1454 / 5:1480 / 10:1448 / 20:1478 / 50:1510 / 100:1218 / 200:1262 /
% 500:1446 / 1000:1344 / 10000:1214
for i = [2 5 10 20 50 100 200 500 1000 10000]
    i
nn = NN_training(tf_idfRed100',trainLabel1vsAll(1,:),i);
[a p] = NN_evaluation(nn,tf_idfValidRed100',validLabel1vsAll(1,:));
tmp = sort(p);
pred = p > tmp(sum(validLabel1vsAll(1,:)==0));
sum(pred == validLabel1vsAll(1,:))
end






% validation result hidden tf_idf reduced: 20:971/40:46.6/ 50:991 / 100:41.6% /200:895 / 500:767 /1000:470/ 5000:512 / 

%hidden: 60   911

%hidden: 70 945

%hidden: 150  998


for h = [10 20 50 100 150 200 500 1000 5000]
    fprintf('hidden: %d',h);
    for i = 1:4
        % 100
        nn = NN_training(tf_idfRed100',trainLabel1vsAll(i,:),h);
        [a b] = NN_evaluation(nn,tf_idfValidRed100',validLabel1vsAll(i,:));    
        p(i,:) = b;
    end

    predLabel = zeros(numLine,1);
    for i= 1:numLine

       [val tmp ]= max(p(:,i));
       predLabel(i) = tmp;
    end

    accuracyNN = sum(validLabel == predLabel-1)
end




dim=100;
[a,b,c]=svds([trainFeatures;validFeatures],dim);
tf_idfRed100= a(1:2000,:);
tf_idfValidRed100 = a(2001:4000,:);

% validation result hidden trainFeature reduced: 10:964 / 20:965 / 

for h = [10 20 50 100 150 200 500 1000 5000]
    fprintf('hidden: %d',h);
    for i = 1:4
        % 100
        nn = NN_training(tf_idfRed100',trainLabel1vsAll(i,:),h);
        [a b] = NN_evaluation(nn,tf_idfValidRed100',validLabel1vsAll(i,:));    
        p(i,:) = b;
    end

    predLabel = zeros(numLine,1);
    for i= 1:numLine

       [val tmp ]= max(p(:,i));
       predLabel(i) = tmp;
    end

    accuracyNN = sum(validLabel == predLabel-1)
end


%ncross 1vsAll svm  tf_idf
% rbf 84.8 89.1 85.6 80.55  => 85.01
% linear 82.9 88.4 83.4 78.15 => 83.21
% sigmoid 85.1 88.65 85.45 80.85 => 85.01
% cross 
% rbf 83.65 90.25 82.3 78.3 => 83.62
% linear 82.3 89.3 80.6 76.75 => 82.23
% sigmoid 83.55 89.75 82.53 78.55 => 83.59



for i=1:4
        fprintf('RBF\n');
    option = '-t 2 -c 500 -v 4 -q -d 4 -s 0 ';
    classifierSVM = svmtrain (trainLabel1vsAll(i,:)' , tf_idf, option)
    %[Output,accuracy,~] = svmpredict (validLabel1vsAll(i,:)'  , tf_idfValid,classifierSVM);
    option = '-t 0 -c 0.1  -v 4 -q -d 4 -s 0 ';
    fprintf('linear\n')
    classifierSVM = svmtrain (trainLabel1vsAll(i,:)' , tf_idf, option)
    %[Output,accuracy,~] = svmpredict (validLabel1vsAll(i,:)'  , tf_idfValid,classifierSVM);

    fprintf('sigmoid\n')    
    option = '-t 3 -c 1000 -v 4 -q -d 4 -s 0 ';
    classifierSVM = svmtrain (trainLabel1vsAll(i,:)' , tf_idf, option)
    %[Output,accuracy,~] = svmpredict (validLabel1vsAll(i,:)' , tf_idfValid,classifierSVM);

end




% 1vsAll    cross bagging
% rbf  80.55 91.3 80.85 77.3    = 82.5
% linear  79.7  90.5 80.4 75.35  = 81.48
% sigmois 70.6 74.2 73.8 64.25  = 70.71

% ncross bagging
% rbf  81.75 90.6 84.65 77.35 =  83.58
% linear  81.7 89.6 83.4 76.05 = 82.68
% sigmois 70.5 74.9 75.4 65.6   = 71.6







for i=1:4
        fprintf('RBF\n');
    option = '-t 2 -c 500  -q -d 4 -s 0 ';
    classifierSVM = svmtrain (trainLabel1vsAll(i,:)' , trainFeatures, option)
    [Output,accuracy,~] = svmpredict (validLabel1vsAll(i,:)'  , validFeatures,classifierSVM);
    option = '-t 0 -c 0.1   -q -d 4 -s 0 ';
    fprintf('linear\n')
    classifierSVM = svmtrain (trainLabel1vsAll(i,:)' , trainFeatures, option)
    [Output,accuracy,~] = svmpredict (validLabel1vsAll(i,:)'  , validFeatures,classifierSVM);

    fprintf('sigmoid\n')    
    option = '-t 3 -c 1000  -q -d 4 -s 0 ';
    classifierSVM = svmtrain (trainLabel1vsAll(i,:)' , trainFeatures, option)
    [Output,accuracy,~] = svmpredict (validLabel1vsAll(i,:)' , validFeatures,classifierSVM);

end
