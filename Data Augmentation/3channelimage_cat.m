%%
close all;clc;
Folder = 'C:\Your\Path\To\Images'; % Define the folder where the images are located
filePattern = fullfile(Folder, '*.jpg'); % Process all files ending with .jpg
theFiles = dir(filePattern); % List all the files and folders in the directory
t = 1;
for k = 1 : length(theFiles) % Loop through each image file in the folder
    baseFileName = theFiles(k).name; % Retrieve the name of the current file to be processed
    imOriginal = imread(fullfile(Folder, baseFileName)); % Read the image file
    file_name = sprintf('ChannelledImage%d.jpg', t); % Generate a sequential file name for the output, e.g., ChannelledImage1.jpg
    I = imresize(imOriginal, [227 227]); % Resize the image to 227x227 pixels
    % MAIN PART:
    final_image = cat(3, I, I, I); % Concatenate the image across the third dimension to replicate channels

    fullFileName = fullfile(Folder, file_name); % Specify the full path for the output file
    imwrite(final_image, fullFileName, 'jpg'); % Write the final image to the disk with JPG format
    pause(1); % Pause for one second to manage processing load
    t = t + 1; % Increment the counter for file naming
end
