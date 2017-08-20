rootFolder = fullfile('~/Documents/NNdataset', 'dataset');
% imgSet_l1 = imageSet(fullfile(rootFolder, 'rain'));
% imgSet_l2 = imageSet(fullfile(rootFolder, 'norain'));
% imgSet_l3 = imageSet(fullfile(rootFolder, 'newnorain'));
imgSet_l4 = imageSet(fullfile(rootFolder, 'extra'));
% % % sampling ratio = 1:2.4
% %imgSet_l1 = partition(imgSet_l1, 0.82, 'randomize');
% %imgSet_l3 = partition(imgSet_l3, 0.6, 'randomize');
imgSet_l4 = partition(imgSet_l4, 0.1, 'randomize');
% % % sampling ratio = 1:1.9
% imgSet_l3 = partition(imgSet_l3, 0.55, 'randomize');
% imgSets = [imgSet_l1, imgSet_l2, imgSet_l3];
% clear imgSet_l1;
% clear imgSet_l2;
% clear imgSet_l3;
%[imgSet1, imgSet2, imgSet3, imgSet4, imgSet5, imgSet6, imgSet7] = partition(imgSets, [0.2,0.2,0.2,0.1,0.1,0.1,0.1], 'randomize');
trainingSet = [imgSet1, imgSet2, imgSet3, imgSet5, imgSet6];
validationSet = [imgSet7, imgSet_l4];
% clear imgSet_l4;
% % imgSets = [ imageSet(fullfile(rootFolder, 'norain')), ...
% %             imageSet(fullfile(rootFolder, 'rain'))  ];
% %        , ...
% %            imageSet(fullfile(rootFolder, 'laptop'))       
% %{imgSets.Description }; % display all labels on one line
% %[imgSets.Count]  ;       % show the corresponding count of images
% %minSetCount = min([imgSets.Count]);
% %imgSets = partition(imgSets, minSetCount, 'randomize');
% % k-fold cross-validation
% % k = 0.2;
% % [imgSet1, imgSet2, imgSet3, imgSet4, imgSet5] = partition(imgSets, [k,k,k,k,k], 'randomize');
% % validationSet = imgSet2;
% % trainingSet = [imgSet1, imgSet3, imgSet4, imgSet5];
% % trainingSet = imgSets;
% % [trainingSet, validationSet] = partition(imgSets, 0.7, 'randomize');
% 
% 
% determine HOG parameter and hog feature size
imgSize = [240 320]; % image resize
rect = [10, 10, 300, 220]; % crop image from center
cellSize = [8 8]; % the HOG cellsize should be varied with repeated classifier training by visualization
[hog_cellsize, vis] = extractHOGFeatures(imcrop(imresize(read(trainingSet(1),1),imgSize), rect),'CellSize',cellSize);
hogFeatureSize = length(hog_cellsize);
% 
% % hog feature extraction
disp('start computing HOG feature')
[ trainingFeatures, train_label ] = computeHOG( trainingSet, hogFeatureSize, cellSize, imgSize, rect );
rowrank = randperm(size(trainingFeatures, 1)); % generate a random sequence of training
trainingFeatures = trainingFeatures(rowrank, :);
train_label = train_label(rowrank);

% % SVM training process
% svm = svmtrain(trainingFeatures, train_label);
% % 
% % SVM testing process
[ validationFeatures, test_label ] = computeHOG( validationSet, hogFeatureSize, cellSize, imgSize, rect );
% % 
% scores = svmclassify(svm, validationFeatures);

% now our train and test sets have been made we need to write them to HDF5
% files. If the files exist, delete them.
disp('start writing hdf5 file')
delete('hog_train.hdf5')
delete('hog_test.hdf5')
% First write the train data
h5create('hog_train.hdf5','/data',[size(trainingFeatures,2), length(train_label)],'Datatype','single');
h5write('hog_train.hdf5','/data',trainingFeatures');
disp('writing training feature done')
h5create('hog_train.hdf5','/label',[1, length(train_label)],'Datatype','single');
h5write('hog_train.hdf5','/label',single(train_label));
disp('writing training label done')
% now write the test data
h5create('hog_test.hdf5','/data',[hogFeatureSize, length(test_label)],'Datatype','single');
h5write('hog_test.hdf5','/data',validationFeatures');
h5create('hog_test.hdf5','/label',[1, length(test_label)],'Datatype','single');
h5write('hog_test.hdf5','/label',single(test_label));

