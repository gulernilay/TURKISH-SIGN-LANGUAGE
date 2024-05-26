
%% Scaling and Segmentation of Images in a Batch Process
close all; clc;
% Specify the folder containing images
Folder = 'C:\Your\Path\To\Images';

% Define file type to process (all jpg files in the folder)
filePattern = fullfile(Folder, '*.jpg');
% List all files matching the file pattern
theFiles = dir(filePattern);

t = 1; % Initialize counter for file naming

% Loop through each file in the folder
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name; % Get the current file name
    fullFilePath = fullfile(Folder, baseFileName); % Create full file path
    imOriginal = imread(fullFilePath); % Read the image

    % Scaling setup: apply 8 different scaling factors to each image
    s = 8;
    magnificationFactor = 0.1; % Start scaling factor at 0.1

    % Loop through each scaling factor
    for c = 1:s
        file_name = sprintf('ScaledImage%d.jpg', t); % Generate new file name
        magnificationFactor = magnificationFactor + 0.4; % Increase scaling factor by 0.4 each iteration

        % Scale the image using bicubic interpolation
        I = imresize(imOriginal, magnificationFactor, "bicubic");

        % Segmentation part: Identify skin based on RGB thresholds
        final_image = zeros(size(I, 1), size(I, 2)); % Initialize the segmentation result matrix

        for i = 1:size(I, 1)
            for j = 1:size(I, 2)
                R = I(i, j, 1);
                G = I(i, j, 2);
                B = I(i, j, 3);
                if (R > 105 && G > 50 && B > 30)
                    v = [R, G, B];
                    if ((max(v) - min(v)) > 15 && abs(R - G) > 15 && R > G && R > B)
                        final_image(i, j) = 1; % Mark pixel as skin
                    end
                end
            end
        end

        % Save the segmented image
        fullFileName = fullfile(Folder, file_name);
        imwrite(final_image, fullFileName, 'jpg');
        pause(1); % Pause for one second to manage processing load
        t = t + 1; % Increment file counter for unique naming
    end
end
