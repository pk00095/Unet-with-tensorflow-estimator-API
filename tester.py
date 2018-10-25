from unet_utils import Recorder
import tensorflow as tf
rec = Recorder()

image_path='/home/pratik/Documents/segmind/face_data/inputs'
mask_path='/home/pratik/Documents/segmind/face_data/targets_face_only'
outpath='./train.tfrecords'



rec.convert(image_path,mask_path,outpath)

#a, b = rec.imgs_input_fn(outpath,height=128,width=128,shuffle=True,repeat_count=-1)

#with tf.Session() as sess:
#    img,mask = sess.run([a,b])
#    print img.shape,mask.shape
