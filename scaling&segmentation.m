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