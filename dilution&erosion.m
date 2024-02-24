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

    for i=71:151
        BW2(i,1)=255;
        i=i+1;
    end 
    new = imfill(BW2,'holes');

    
    fullFileName = fullfile(Folder, file_name); % dosyaya yenı dosya açıyor ve ona jpg yazdırıyor
    imwrite(BW2 ,file_name,'jpg');
    pause(1); % pause for one second
    t=t+1;
end 