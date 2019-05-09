cartoon_filter is the code of my project.
I have put some demo source image in demo_image folder for testing. If you want to test your own photo, please put it
that folder and named it with number.

After opening the code with whatever IDE you want,  you will see that there is a num
variance which indicate the name of the picture you want to input from the demo_image folder.
This is a while loop generating program, please change both the num variance and the name of the last image you want 
to input after the "while". Then you can batch generate the results and the results will be stored in output folder.

I provided two sky image and three style imageshere, please use whatever you want and don't  forget to change 
the sky and style images' file names in my code. Notice that style1 and style2 can lead to a bright and colorful output 
and style3's output's color will be much more similar to original image.

I recommand you to run the code under Anaconda. All you need to do is installing a OpenCV package. If you use other
environment to run the code, please install following package first:

cv2
matplotlib
imageio
numpy
scipy
skimage

Thanks

Ziyu

