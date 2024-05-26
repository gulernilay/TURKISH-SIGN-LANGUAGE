%% Batch process images: Rotate and segment images from a folder

close all; clc;
% Define the folder containing images
Folder = 'C:\Your\Path\To\Images';
% Specify the file type to process; here, all jpg files
filePattern = fullfile(Folder, '*.jpg');
% List all files matching the file pattern in the specified folder
theFiles = dir(filePattern);

t = 1; % Initialize counter for file naming

% Loop through each file in the folder
for k = 1:length(theFiles)
    baseFileName = theFiles(k).name; % Get the file name
    fullFilePath = fullfile(Folder, baseFileName); % Create full file path
    imOriginal = imread(fullFilePath); % Read the image file

    s = 5; % Specify the number of rotations per image
    angle = 120; % Initial angle for rotation

    % Apply rotations and process each rotated image
    for c = 1:s
        file_name = sprintf('RotatedImage%d.jpg', t); % Generate new file name
        demo2 = imrotate(imOriginal, angle); % Rotate the original image
        angle = angle + 30; % Increment the angle for the next rotation

        I2 = imresize(demo2, [224 224]); % Resize the rotated image
        final_image = zeros(size(I2, 1), size(I2, 2)); % Initialize a matrix for segmentation

        % Segmentation based on color thresholds
        for i = 1:size(I2, 1)
            for j = 1:size(I2, 2)
                R = I2(i, j, 1);
                G = I2(i, j, 2);
                B = I2(i, j, 3);
                if (R > 105 && G > 50 && B > 30)
                    v = [R, G, B];
                    if ((max(v) - min(v)) > 15)
                        if (abs(R-G) > 15 && R > G && R > B)
                            final_image(i, j) = 1; % Mark pixel as skin
                        end
                    end
                end
            end
        end

        fullFileName = fullfile(Folder, file_name); % Create full path for the output file
        imwrite(final_image, fullFileName, 'jpg'); % Save the segmented image as JPG
        pause(1); % Pause for one second to manage processing load
        t = t + 1; % Increment file counter for unique naming
    end
end
