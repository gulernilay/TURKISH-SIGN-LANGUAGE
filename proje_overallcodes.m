%% İŞARET DİLİ DATASET PROCESS CODES:
%% Read all files and apply segmentation and store processed files.
close all;clc;
Folder = 'C:\Users\nilay\Desktop\DENEME' ;% % Resimler hangi dosyadan alınacak 
filePattern = fullfile(Folder, '*.jpg'); % İşleme alınan dosya tipi farketmeksizin son halini jpg olarak kaydet 
theFiles = dir(filePattern);% dosyaları ve klasörlerı  lıstelenir / Ynaı bır dosyada kac tane resım fıle var ıse o kadar ıslem yapacak 
t=1;
for k = 1 : length(theFiles) % Dosyada bulunan resim sayısı kadar döngü döndürülür 
    baseFileName = theFiles(k).name; % işlenecek dosyanın ismi alınır  
    I = imread(baseFileName);% işlenecek dosyanın okunur 
    file_name = sprintf('Processed%d.png', t);

  % SEGMENTATION PART :
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
    fullFileName = fullfile(Folder, file_name); % dosyaya yenı dosya açıyor ve ona jpg yazdırıyor
    imwrite(final_image,file_name,'jpg');
    pause(1); % pause for one seconD
    t=t+1;
end 
%%

clear all;close all;clc;
% Gray Level Scale Image 
i=imread('A_0_43.jpg');
imshow(i) ;
title('Input Image'); 
 I1=rgb2gray(i);
 figure,imshow(I1) ;
 title('Gray-Scale Image');


%% Read a file  and resize them  
        I = imread('A_0_36.jpg'); %Read built-in image
        figure, imshow(I); %Display image
        K = imresize(I,[240 240]); %Create an output image with 100 rows and 150 columns
        figure, imshow(K); %Display 100 x 150 image

%%  Read all files and resize them. 
clear all;close all;clc;
Folder = 'C:\Users\nilay\Desktop\DENEME' ;% % Resimler hangi dosyadan alınacak 
filePattern = fullfile(Folder, '*.jpg'); % İşleme alınan dosya tipi farketmeksizin son halini jpg olarak kaydet 
theFiles = dir(filePattern);% dosyaları ve klasörlerı  lıstelenir / Ynaı bır dosyada kac tane resım fıle var ıse o kadar ıslem yapacak 
t=1;
for k = 1 : length(theFiles) % Dosyada bulunan resim sayısı kadar döngü döndürülür 
    baseFileName = theFiles(k).name; % işlenecek dosyanın ismi alınır  
    imOriginal = imread(baseFileName);% işlenecek dosyanın okunur 

    file_name = sprintf('Resized%d.png', t);   % name Image with a sequence of number, ex Image1.png , Image2.png....
    I = imresize(imOriginal,[224 224]); % scaling part 
    % bicubic veya bilinear en iyi sonuç veren metotlardan biriydi 
    fullFileName = fullfile(Folder, file_name); % dosyaya yenı dosya açıyor ve ona jpg yazdırıyor
    imwrite(I,file_name,'jpg');
    pause(1); % pause for one second
    t=t+1;
end

%% ÖNCE SCALING SONRA SEGMENTATION
close all;clc;
Folder = 'C:\Users\nilay\Desktop\DENEME' ;% % Resimler hangi dosyadan alınacak 

filePattern = fullfile(Folder, '*.jpg'); % İşleme alınan dosya tipi farketmeksizin son halini jpg olarak kaydet 
theFiles = dir(filePattern);% dosyaları ve klasörlerı  lıstelenir / Ynaı bır dosyada kac tane resım fıle var ıse o kadar ıslem yapacak 
t=1;
for k = 1 : length(theFiles) % Dosyada bulunan resim sayısı kadar döngü döndürülür 
    baseFileName = theFiles(k).name; % işlenecek dosyanın ismi alınır  
    imOriginal = imread(baseFileName);% işlenecek dosyanın okunur 
    
    % SCALING PART :
    s= 8;  % 1 resime 8 farklı scaling faktörü uygulanır 
    magnificationFactor = 0.1;  % Piksel sayısı azaltılır veya coğaltılır 
    for c = 1:s 
    file_name = sprintf('ScaledImage%d.png', t);   % name Image with a sequence of number, ex Image1.png , Image2.png....
    magnificationFactor = magnificationFactor + 0.4 ;  % Uygulanan scalıng : 0.1,0.5,0.9,1.3,1.7,2.1,2.5,2.9
    I = imresize(imOriginal, magnificationFactor,"bicubic"); % scaling part 
  % bicubic veya bilinear en iyi sonuç veren metotlardan biriydi 
    

  % SEGMENTATION PART :
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
    fullFileName = fullfile(Folder, file_name); % dosyaya yenı dosya açıyor ve ona jpg yazdırıyor
    imwrite(final_image,file_name,'jpg');
    pause(1); % pause for one second
    t=t+1;
    end
end
%%



close all; clc;
Folder = 'C:\Users\nilay\Desktop\DENEME' ;% Resimler hangi dosyadan alınacak 
filePattern = fullfile(Folder, '*.jpg'); % İşleme alınan dosya tipi farketmeksizin son halini jpg olarak kaydet 
theFiles = dir(filePattern); % dosyaları ve klasörlerı  lıstelenir / Ynaı bır dosyada kac tane resım fıle var ıse o kadar ıslem yapacak 

t=1;
for k = 1 : length(theFiles)  % Dosyada bulunan resim sayısı kadar döngü döndürülür 
    baseFileName = theFiles(k).name; % işlenecek dosyanın ismi alınır 
    imOriginal = imread(baseFileName); % işlenecek dosya okunur 

    s= 10; % Her bir resime 5 farklı rotation uygulaması uygulanır ( for loop is applied 5 times)
    angle=30; % 120-150-180-210-240-270 derece döndürme işlemleri uygulanır  

     for c = 1:s
     
     file_name = sprintf('RotatedImage%d.png', t);  
     % name Image with a sequence of number, ex Image1.png , Image2.png..
        angle = angle +30;
        demo2 = imrotate(imOriginal,angle);  % rotation part 
        I2=demo2; 

        I = imresize(I2,[224 224]); % scaling part 
        final_image = zeros(size(I,1),size(I,2)); % segmentatıon part 
            for i = 1:size(I,1) %200
                for j = 1:size(I,2) %396 
                    R = I(i,j,1); %115
                    G = I(i,j,2); % 116
                    B = I(i,j,3); %110
                    if(R > 105 && G > 50  && B > 30)
                        v = [R,G,B];
                        if((max(v) - min(v)) > 15) % 116-110 =6
                            if(abs(R-G) > 15 && R > G && R > B)
%                                 it is a skin
                                final_image(i,j) = 1; % BEYAZ PİKSEL 
                            end
                        end
                    end
                end
            end
            fullFileName = fullfile(Folder, file_name); % Rotasyon ve segmentasyon uygulanmıs resımler adlandırılarak hedef dosyaya kaydedılır
            imwrite(final_image,file_name,'jpg'); % JPG dosyası olarak yazdırılır
            pause(1); % Algorıtma 1 sn aralıklarla calısır 
            t=t+1;
            
     end
end

%%
close all;clc;
Folder = 'C:\Users\nilay\Desktop\DENEME' ;% % Resimler hangi dosyadan alınacak 
filePattern = fullfile(Folder, '*.jpg'); % İşleme alınan dosya tipi farketmeksizin son halini jpg olarak kaydet 
theFiles = dir(filePattern);% dosyaları ve klasörlerı  lıstelenir / Ynaı bır dosyada kac tane resım fıle var ıse o kadar ıslem yapacak 
t=1;
for k = 1 : length(theFiles) % Dosyada bulunan resim sayısı kadar döngü döndürülür 
    baseFileName = theFiles(k).name; % işlenecek dosyanın ismi alınır  
    imOriginal = imread(baseFileName);% işlenecek dosyanın okunur 
    file_name = sprintf('ChannelledImage%d.jpg', t);   % name Image with a sequence of number, ex Image1.png , Image2.png....
    I = imresize(imOriginal, [227 227]);  
      % MAIN PART :
      final_image = cat(3,I,I,I); 

    fullFileName = fullfile(Folder, file_name); % dosyaya yenı dosya açıyor ve ona jpg yazdırıyor
    imwrite(final_image,file_name,'png');
    pause(1); % pause for one second
    t=t+1;
end 


%% SCALING AND EROSION/DILUTION
close all;clc;
Folder = 'C:\Users\nilay\Desktop\DENEME' ;% % Resimler hangi dosyadan alınacak 
filePattern = fullfile(Folder, '*.jpg'); % İşleme alınan dosya tipi farketmeksizin son halini jpg olarak kaydet 
theFiles = dir(filePattern);% dosyaları ve klasörlerı  lıstelenir / Ynaı bır dosyada kac tane resım fıle var ıse o kadar ıslem yapacak 
t=1;
for k = 1 : length(theFiles) % Dosyada bulunan resim sayısı kadar döngü döndürülür 
    baseFileName = theFiles(k).name; % işlenecek dosyanın ismi alınır  
    I = imread(baseFileName);% işlenecek dosyanın okunur 
    file_name = sprintf('Morphological%d.png', t);

  % SEGMENTATION PART :
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

    SE= strel("square",3);
    im = double(final_image);
    dil=imdilate(im,SE);
    er=imerode(dil,SE);
    BW2 = imfill(er,'holes');
    
    fullFileName = fullfile(Folder, file_name); % dosyaya yenı dosya açıyor ve ona jpg yazdırıyor
    imwrite(BW2 ,file_name,'jpg');
    pause(1); % pause for one seconD
    t=t+1;
end 


%% EROSION/DILUTION 
close all;clc;
im = imread('A_0_36.jpg');
imshow(im);
final_image = double(im);
% MORPHOLOGICAL MODIFICATIONS : ( AFTER SEGMENTATION, SHAPE İS REBUİLT
SE= strel("square",3);
erosion=imerode(final_image,SE);
figure;imshow(erosion);title("Erosion");
dilution=imdilate(final_image,SE);
figure;imshow(dilution);title("Dilution");
for i=71:151
    BW2(i,1)=255;
    i=i+1;
end 
new = imfill(BW2,'holes');
% for i2=209:224


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












