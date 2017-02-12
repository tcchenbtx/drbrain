from keras.utils.visualize_util import plot
from keras.models import load_model
import os
import numpy as np
import cv2
import h5py
from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.utils.vggnet import VGG16
from vis.visualization import visualize_saliency
from utils import data_generator
import json

# path
base_path = os.path.abspath(os.path.dirname(__file__))
base_path = os.path.join(base_path, "..")
raw_data_path = os.path.join(base_path, "raw_data")
code_path = os.path.join(base_path, "code")
log_path = os.path.join(code_path, "log")
preprocess_data_path = os.path.join(base_path, "preprocess_data")

# path to the Makefile
make_file_path = os.path.join(code_path, "Makefile")

# path to where to save the h5
total_matrix_path = os.path.join(preprocess_data_path, "total_matrix_no_mci.h5")
total_tf_matrix_path = os.path.join(preprocess_data_path, "total_tf_matrix_no_mci.h5")
unique_matrix_path = os.path.join(preprocess_data_path, "unique_matrix_no_mci.h5")
unique_tf_matrix_path = os.path.join(preprocess_data_path, "unique_tf_matrix_no_mci.h5")

unique_tf_train_path = os.path.join(preprocess_data_path, "unique_tf_train_no_mci.h5")
unique_tf_valid_path = os.path.join(preprocess_data_path, "unique_tf_valid_no_mci.h5")
unique_tf_test_path = os.path.join(preprocess_data_path, "unique_tf_test_no_mci.h5")

# rorate image
train_rotate_path = os.path.join(preprocess_data_path, "train_rotate.h5")
valid_rotate_path = os.path.join(preprocess_data_path, "valid_rotate.h5")
test_rotate_path = os.path.join(preprocess_data_path, "test_rotate.h5")

# check size
with h5py.File(valid_rotate_path, 'r') as hf:
    X = hf.get('X')
    Y = hf.get('Y')
    print(X.shape)
    print(Y.shape)

with h5py.File(test_rotate_path, 'r') as ht:
    X = ht.get('X')
    Y = ht.get('Y')
    print(X.shape)
    print(Y.shape)



# model path
model_path = os.path.join(code_path, "models")
model_body = os.path.join(model_path, "rotate_model_new.h5")
model_weights = os.path.join(model_path, "rotate_weights_new.h5")
model_plot = os.path.join(model_path, "new_structural_new.png")

model = load_model(model_body)
plot(model, to_file=model_plot)



# Build the VGG16 network with ImageNet weights

print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)



# add name to preidcitons model the dense (4 ) one!!!



layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

# Images corresponding to tiger, penguin, dumbbell, speedboat, spider


with h5py.File(unique_tf_test_path, 'r') as hf:
    X = hf.get('X')
    Y = hf.get('Y')
    print(X.shape)
    print(Y.shape)
    target_image = np.array(X[0,:,:,:,:]).reshape((1,96,112,96,1))


print(model.predict(target_image))
pred_class = np.argmax(model.predict(target_image))
print(pred_class)
print(type(pred_class))



score = model.evaluate_generator(data_generator.generator_tf_no_mci(valid_rotate_path),
                                val_samples=1656, max_q_size=20, nb_worker=1, pickle_safe=False)


#y_score = model.evaluate_generator(data_generator.generator_tf(unique_tf_valid_path),
#                                val_samples=32, max_q_size=20, nb_worker=1, pickle_safe=False)








# score = model.evaluate(valid_X, valid_Y, batch_size=500, verbose=1)

# evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
# print score
# score = model.evaluate(valid_X_input, valid_Y_input, batch_size=3, verbose=1)
print('Valid score:', score[0])
print('Valid accuracy:', score[1]*100)


score = model.evaluate_generator(data_generator.generator_tf_no_mci(test_rotate_path),
                                val_samples=1656, max_q_size=20, nb_worker=1, pickle_safe=False)

print('Test score:', score[0])
print('Test accuracy:', score[1]*100)




# old test
score = model.evaluate_generator(data_generator.generator_tf_no_mci(unique_tf_test_path),
                                val_samples=215, max_q_size=20, nb_worker=1, pickle_safe=False)

print('Test score:', score[0])
print('Test accuracy:', score[1]*100)









# with 32 valid
#predict_y = model.predict_generator(data_generator.generator_tf_no_mci_order(unique_tf_test_path), 
#                                    val_samples=185, max_q_size=10, nb_worker=1, pickle_safe=False)


#with h5py.File(unique_tf_test_path, 'r') as hf:
#    X = hf.get('X')
#    Y = hf.get('Y')
#    real_y = np.array(Y[:,:])
#    print("unique tf test")
#    print(X.shape)
#    print(Y.shape)


# with rotate test 

predict_y = model.predict_generator(data_generator.generator_tf_no_mci_order(test_rotate_path),
                                    val_samples=1656, max_q_size=10, nb_worker=1, pickle_safe=False)


with h5py.File(test_rotate_path, 'r') as hf:
    X = hf.get('X')
    Y = hf.get('Y')
    real_y = np.array(Y[:,:])
    print("unique tf test")
    print(X.shape)
    print(Y.shape)


print(real_y.shape)
print(predict_y.shape)


real_y_list = []
predict_y_list = []

for i in range(real_y.shape[0]):
    real_y_list.append(np.argmax(real_y[i,:]))
    predict_y_list.append(np.argmax(predict_y[i, :]))




predict_8082 = {}
predict_8082['real_y'] = real_y_list
predict_8082['predict_y'] = predict_y_list


with open("predict_8082.json", "w") as uoutput:
    json.dump(predict_8082, uoutput, indent=4, sort_keys=True)




print(real_y_list)
print(predict_y_list)



from sklearn import metrics
myprecision = metrics.precision_score(real_y_list, predict_y_list, average='macro')
myrecall = metrics.recall_score(real_y_list, predict_y_list, average='micro')
print(myprecision)
print(myrecall)


from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(real_y_list, predict_y_list)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))





#target_img_original_shape = target_image.reshape((96, 112, 96))
#heatmap = visualize_saliency(model, layer_idx, [pred_class], target_img_original_shape, text="test")

#print(type(heatmap))



#
# image_paths = [
#
# ]
#
# heatmaps = []
# for path in image_paths:
#     # Predict the corresponding class for use in `visualize_saliency`.
#     seed_img = utils.load_img(path, target_size=(224, 224))
#     pred_class = np.argmax(model.predict(np.array([img_to_array(seed_img)])))
#
#     # Here we are asking it to show attention such that prob of `pred_class` is maximized.
#     heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img, text=utils.get_imagenet_label(pred_class))
#     heatmaps.append(heatmap)
#
# cv2.imshow("Saliency map", utils.stitch_images(heatmaps))
# cv2.waitKey(0)
