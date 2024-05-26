%% Batch Process for Image Segmentation

close all; clc; clear all;

% Specify the folder where images are stored
Folder = 'C:\Your\Path\To\Images'; 
% Define the file type to be processed
filePattern = fullfile(Folder, '*.jpg'); 
% List all files matching the file pattern
theFiles = dir(filePattern);
t = 1;

% Loop through each file found
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name; 
    fullFilePath = fullfile(Folder, baseFileName); % Create full path for reading the image
    I = imread(fullFilePath); % Read the image file
    file_name = sprintf('Processed%d.jpg', t); % Generate a file name for the processed image

    % SEGMENTATION PART:
    final_image = zeros(size(I,1), size(I,2)); % Initialize a segmentation matrix
    % Process each pixel in the image
    for i = 1:size(I,1)
        for j = 1:size(I,2)
            R = I(i,j,1); % Red channel value
            G = I(i,j,2); % Green channel value
            B = I(i,j,3); % Blue channel value
            % Define skin color detection logic
            if (R > 105 && G > 50 && B > 30)
                v = [R, G, B];
                if ((max(v) - min(v)) > 15 && abs(R-G) > 15 && R > G && R > B)
                    final_image(i,j) = 1; % Mark pixel as part of the segmentation
                end
            end
        end
    end

    fullFileName = fullfile(Folder, file_name); % Specify the path where the segmented image will be saved
    imwrite(final_image, fullFileName, 'jpg'); % Save the segmented image as JPG
    pause(1); % Pause for one second to manage processing load
    t = t + 1; % Increment the file name counter
end
