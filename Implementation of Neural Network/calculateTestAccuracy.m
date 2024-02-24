TEST ACCURACY HESAPLANMASI : 
testData=imageDatastore('E:\TÜRK İŞARET DİLİ\Testing','IncludeSubfolders' ,true, 'LabelSource' , 'foldernames');

[YPred,scores] = classify(trainedNetwork_1,testData);
YValidation = testData.Labels;
accuracy = mean(YPred == YValidation);
