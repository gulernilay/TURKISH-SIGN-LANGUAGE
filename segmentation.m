% Read all files and apply segmentation and store processed files.

close all;clc;clear all;

Folder = 'C:\Users\nilay\Desktop\DENEME' ; 
filePattern = fullfile(Folder, '*.jpg'); 
theFiles = dir(filePattern);
t=1;
for k = 1 : length(theFiles) 
    baseFileName = theFiles(k).name; 
    I = imread(baseFileName);
    file_name = sprintf('Processed%d.png', t);

  % SEGMENTATION PART :
final_image = zeros(size(I,1), size(I,2)); 
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
