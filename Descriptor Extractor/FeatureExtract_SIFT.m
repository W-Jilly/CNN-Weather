rootFolder = fullfile('~/Documents', 'NNdataset');
trainingSet = imageSet(fullfile(rootFolder, 'resNorain_new'));
% imgSet_l4 = partition(imgSet_l4, 0.1, 'randomize');
% imgSets = [ imageSet(fullfile(rootFolder, 'resRain')), ...
%             imageSet(fullfile(rootFolder, 'resNorain'))  ];
% %        , ...
% %            imageSet(fullfile(rootFolder, 'laptop'))       
% %{imgSets.Description }; % display all labels on one line
% %[imgSets.Count]  ;       % show the corresponding count of images
% %minSetCount = min([imgSets.Count]);
% %imgSets = partition(imgSets, minSetCount, 'randomize');
% %[imgSet1, imgSet2, imgSet3, imgSet4, imgSet5, imgSet6, imgSet7] = partition(imgSets, [0.2,0.2,0.2,0.1,0.1,0.1,0.1], 'randomize');
% trainingSet = [imgSet1, imgSet2, imgSet4];
% validationSet = [imgSet7, imgSet_l4];
% validationSet = partition(validationSet, 0.4, 'randomize');
% disp(sum([trainingSet.Count]));
% disp(sum([validationSet.Count]));
% 
descriptors = [];
numSets = numel(imgSets);
numImages = sum([imgSets.Count]);
counts = zeros(numImages, 1);
j = 1;
for categoryIndex=1:numSets
    imgSet = imgSets(categoryIndex);
    for i = 1:imgSet.Count
              img = read(imgSet,i);                     
              [tempDescriptors] = extractDescriptorsFromImage(img);
              counts(j) = size(tempDescriptors, 2);
              descriptors = [descriptors tempDescriptors]; 
              j = j + 1;
%               disp(imgSet.Description)
    end     
end
disp('feature extraction is done')

% dictionary of words
% iter = 230;
% codewords = 1000;
% vocabulary = learnCodebook(descriptors, codewords, iter);
% % clear descriptors % clean memory
% disp('dictionary is ready')

% encode images
% if_norm = 1;
% [ BoWvec, train_label ] = encodingImage( trainingSet, vocabulary, if_norm ); % hard assignment
% [ BoWvec, train_label ] = encodingImage_soft( trainingSet, vocabulary ); % hard assignment
% disp('training set encoding is done')
% disp('word encoding is done')
% 
% % training a SVM classifier
% % lambda = 0.01 ; % Regularization parameter
% % maxIter = 1000 ; % Maximum number of iterations
% rowrank = randperm(size(BoWvec, 2)); % generate a random sequence of training
% BoWvec = BoWvec(:, rowrank);
% train_label = train_label(rowrank);
% [w, b, info] = vl_svmtrain(BoWvec, label, lambda, 'MaxNumIterations', maxIter);
% disp('SVM training is done')

% test on SVM classifier
% [ BoWvec_test, test_label ] = encodingImage( validationSet, vocabulary, if_norm );
% disp('validation set encoding is done')
% 
% [~,~,~, scores] = vl_svmtrain(BoWvec, label, 0, 'model', w, 'bias', b, 'solver', 'none') ;
% vl_roc(label, scores) ;

%disp('test is done')

% % now our train and test sets have been made we need to write them to HDF5
% % files. If the files exist, delete them.
% disp('start writing hdf5 file')
% delete('sift_train.hdf5')
% delete('sift_test.hdf5')
% % First write the train data
% h5create('sift_train.hdf5','/data',[size(BoWvec,1), length(train_label)],'Datatype','single');
% h5write('sift_train.hdf5','/data',single(BoWvec));
% disp('writing training feature done')
% h5create('sift_train.hdf5','/label',[1, length(train_label)],'Datatype','single');
% h5write('sift_train.hdf5','/label',single(train_label));
% disp('writing training label done')
% % now write the test data
% h5create('sift_test.hdf5','/data',[size(BoWvec_val,1), length(test_label)],'Datatype','single');
% h5write('sift_test.hdf5','/data',single(BoWvec_val));
% h5create('sift_test.hdf5','/label',[1, length(test_label)],'Datatype','single');
% h5write('sift_test.hdf5','/label',single(test_label));
