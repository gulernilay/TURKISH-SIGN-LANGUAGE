%% Read and resize all .jpg files in a specified folder

clear all; close all; clc;

% Define the folder path where the images are located
Folder = 'C:\Path\To\Your\Images'; 
% Create a file pattern to match all .jpg files in the folder
filePattern = fullfile(Folder, '*.jpg'); 
% Retrieve the list of all jpg files in the directory
theFiles = dir(filePattern); 
t = 1;

% Loop through each file found
for k = 1:length(theFiles) 
    baseFileName = theFiles(k).name; % Get the current file's name
    fullFilePath = fullfile(Folder, baseFileName); % Construct full file path
    imOriginal = imread(fullFilePath); % Read the image file

    % Define the new file name for the resized image
    file_name = sprintf('Resized%d.jpg', t); 
    % Resize the original image to 224x224 pixels
    I = imresize(imOriginal, [224 224]); 
    % Define the full file name where the resized image will be saved
    fullFileName = fullfile(Folder, file_name); 
    % Save the resized image to the same folder
    imwrite(I, fullFileName, 'jpg');
    % Pause for one second to manage processing load
    pause(1); 
    t = t + 1; % Increment the file name counter
end
