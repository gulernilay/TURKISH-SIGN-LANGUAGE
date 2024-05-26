
% Convert an RGB image to a Grayscale image

% Read the original RGB image
i = imread('A_0_43.jpg');
% Display the original image
imshow(i);
title('Input Image'); % Title for the input image display

% Convert the RGB image to Grayscale
I1 = rgb2gray(i);
% Display the grayscale image
figure, imshow(I1);
title('Gray-Scale Image'); % Title for the grayscale image display
