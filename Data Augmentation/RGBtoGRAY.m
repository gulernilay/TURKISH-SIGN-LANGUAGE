
clear all;close all;clc;

% Gray Level Scale Image i

i=imread('A_0_43.jpg');
imshow(i) ;
title('Input Image'); 
  I1=rgb2gray(i);
  figure,imshow(I1) ;
  title('Gray-Scale Image');