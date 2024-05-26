%% SCALING AND EROSION/DILATION SCRIPT
close all; clc;
Folder = 'C:\Your\Path\To\Images'; % Directory where the images are stored
filePattern = fullfile(Folder, '*.jpg'); % Process all .jpg files in the folder
theFiles = dir(filePattern); % List all the files in the directory
t = 1;

for k = 1 : length(theFiles) % Loop through each image in the folder
    baseFileName = theFiles(k).name; % Get the current file name
    I = imread(fullfile(Folder, baseFileName)); % Read the image
    file_name = sprintf('Morphological%d.jpg', t); % Generate a new file name for the output

    % SEGMENTATION PART:
    final_image = zeros(size(I,1), size(I,2)); % Initialize a matrix of zeros matching the size of I
    for i = 1:size(I,1) % Loop over all rows
        for j = 1:size(I,2) % Loop over all columns
            R = I(i,j,1); % Red channel
            G = I(i,j,2); % Green channel
            B = I(i,j,3); % Blue channel
            if (R > 105 && G > 50 && B > 30) % Threshold conditions
                v = [R, G, B];
                if ((max(v) - min(v)) > 15) % Color variance condition
                    if (abs(R-G) > 15 && R > G && R > B) % Skin color detection
                        final_image(i,j) = 1; % Mark pixel as part of the skin
                    end
                end
            end
        end
    end

    % MORPHOLOGICAL OPERATIONS:
    SE = strel("square", 3); % Structuring element
    dil = imdilate(final_image, SE); % Dilation operation
    er = imerode(dil, SE); % Erosion operation
    BW2 = imfill(er, 'holes'); % Fill holes in the binary image

    % Special handling for setting specific pixels (may need adjustment based on specific use-case)
    for i = 71:151
        BW2(i, 1) = 255; % Set specific range of pixels to 255
    end
    new = imfill(BW2, 'holes'); % Refill holes after manual adjustments

    fullFileName = fullfile(Folder, file_name); % Specify the full path for the output file
    imwrite(new, fullFileName, 'jpg'); % Save the processed image as JPEG
    pause(1); % Pause for one second to avoid overloading processing
    t = t + 1; % Increment file counter
end