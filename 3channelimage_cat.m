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
