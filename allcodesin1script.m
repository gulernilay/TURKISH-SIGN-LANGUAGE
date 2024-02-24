%% ALL CODES IN 1 SCRIPT:
clear all;close all;clc;
I= imread('E:\TÜRK İŞARET DİLİ\NEW DATABASE\ANA RESİMLER\B\ANA RESİMLER\Image7.png');

%SEGMENTATION: 
final_image = zeros(size(I,1), size(I,2)); % 200*396 lık matrik
    for i = 1:size(I,1) %200
        for j = 1:size(I,2) %396 
            R = I(i,j,1); %115
            G = I(i,j,2); % 116
            B = I(i,j,3); %110
            if(R > 105 && G > 50  && B > 30)
                v = [R,G,B];
                if((max(v) - min(v)) > 15) % 116-110 =6
                    if(abs(R-G) > 15 && R > G && R > B)
                        %it is a skin
                        final_image(i,j) = 1; % BEYAZ PİKSEL 
                    end
                end
            end
        end
    end
image_thresholded = final_image;
image_thresholded(final_image>=0.5) = 255;
image_thresholded(final_image<0.5) = 0;
%MORPHOLOGICAL OPERATIONS:
SE = strel('square',3); %Structuring element
RT= imclose(image_thresholded,SE);
RC= imfill(RT,"holes");
%CAT OPERATION AND UINT
RGB = cat(3,RC,RC,RC);
NEWIMAGE=uint8(RGB);
%BITAND OPERATION:
final2Image= bitand(NEWIMAGE,I);
figure()
subplot(1,4,1)
imshow(I,[]) 

subplot(1,4,2)
imshow(image_thresholded,[])

subplot(1,4,3)
imshow(RGB,[])

subplot(1,4,4)
imshow(final2Image,[]);


%%
