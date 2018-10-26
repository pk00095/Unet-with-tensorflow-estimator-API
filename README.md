# Unet-with-tensorflow-estimator-API


# ground truth mask

use the [pixel_Annotation_tool](https://github.com/abreheret/PixelAnnotationTool) to Annotate the image then create binary mask by thresholding .

example of creating binary mask from the masks created by pixel Annotation Tool:

    img = cv2.imread(i,0)   # read the image as grayscale
    e= img>100              # threshold for pixel value above 100
    ss = e.astype(np.uint8) 
    ss = ss*255
    cv2.imwrite('name.jpg',ss)
    
# creating and parsing tfrecords

* [unet_utils.py](https://github.com/pk00095/Unet-with-tensorflow-estimator-API/blob/master/unet_utils.py) contains functions necessary to convert raw images and masks into [tfrecord](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564) format.

* use the script tfecord_creator.py to create tfrecords from training and testing images. open it and set the path to point to the directories containing training and testing images. 


# training unet
[unet_estimator.py](https://github.com/pk00095/Unet-with-tensorflow-estimator-API/blob/master/unet_estimator.py) contains the necessary function to create the unet, build the estimator, set the training schedule, perform training, store checkpoints, write events for tensorboard vizualization.

