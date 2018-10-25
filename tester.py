from unet_utils import Recorder
import tensorflow as tf
rec = Recorder()

train_image_path='/home/pratik/Documents/face_data/inputs_train'
train_mask_path='/home/pratik/Documents/face_data/targets_face_only_train'
train_outpath='./train.tfrecords'
# converts the training data to tfrecords
rec.convert(train_image_path, train_mask_path, train_outpath)


test_image_path='/home/pratik/Documents/face_data/inputs_test'
test_mask_path='/home/pratik/Documents/face_data/targets_face_only_test'
test_outpath='./test.tfrecords'
# converts testing data to tfrecords
rec.convert(test_image_path, test_mask_path, test_outpath)
