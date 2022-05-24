# Reading-text-image_OpenCV
Problem Statement : Reading and cleaning text image
Solution summary :
we started with an image having text in it. We then blurred the image to remove the noise. Then, thresholding is done to convert the pixels, 
either to '0' or '255'. This has made the text sharp but has also infused some noise in the form of dots. Then we tried to remove those dots 
by performing erosion and dilation operations. The problem here is that these operations work better if the size of the noise/dots is much 
smaller than the text. Doing more morphological transformation for more number of loops or increasing the size of the structuring element 
will also adversely affect the shape of the text. So, we identified the dots by finding the contours and removed the ones whose area is less 
than a threshold. Finally, we had a clean image which can be used as input to OCR.
