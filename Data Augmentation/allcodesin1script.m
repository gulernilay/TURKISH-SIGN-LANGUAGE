%% ALL CODES IN ONE SCRIPT:
clear all; close all; clc;
% Load the image
I = imread('C:\Your\Path\To\Images');

%% SEGMENTATION:
% Initialize the segmentation result matrix to zero
final_image = zeros(size(I,1), size(I,2)); % Initialize a matrix of zeros matching the size of I
for i = 1:size(I,1) % Loop over rows
    for j = 1:size(I,2) % Loop over columns
        R = I(i,j,1); % Red channel value
        G = I(i,j,2); % Green channel value
        B = I(i,j,3); % Blue channel value
        % Skin detection conditions
        if (R > 105 && G > 50 && B > 30)
            v = [R, G, B];
            if ((max(v) - min(v)) > 15) % Check for sufficient color variance
                if (abs(R-G) > 15 && R > G && R > B)
                    final_image(i,j) = 1; % Identify as skin pixel
                end
            end
        end
    end
end

% Thresholding to create a binary image
image_thresholded = final_image;
image_thresholded(final_image >= 0.5) = 255;
image_thresholded(final_image < 0.5) = 0;

%% MORPHOLOGICAL OPERATIONS:
SE = strel('square',3); % Create a square structuring element
RT = imclose(image_thresholded, SE); % Perform morphological closing
RC = imfill(RT, 'holes'); % Fill holes

%% COMBINE CHANNELS AND CONVERT TO UINT8:
RGB = cat(3, RC, RC, RC); % Stack the binary mask into three channels
NEWIMAGE = uint8(RGB); % Convert to uint8 for further operations

%% BITWISE AND OPERATION:
final2Image = bitand(NEWIMAGE, I); % Apply bitwise AND with the original image

%% DISPLAY RESULTS:
figure();
subplot(1,4,1);
imshow(I, []);
subplot(1,4,2);
imshow(image_thresholded, []);
subplot(1,4,3);
imshow(RGB, []);
subplot(1,4,4);
imshow(final2Image, []);

%%
