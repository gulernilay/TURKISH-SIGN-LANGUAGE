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