# Steps for this program are:
# 1- Importing frameworks
# 2- Preparing paths
# 3- Processing images
# 4- Defining placeholders
# 5- Defining the network structure
# 6- Defining the loss function and optimizer
# 7- Model training and prediction
# 8- Model prediction

# to download data from OBS run: wget https://hciaai.obs.cn-north-4.myhuaweicloud.com:443/flower_photos.zip
# then: unzip image.zip

from skimage import io,transform
import glob
import os
import numpy as np
import tensorflow as tf
import time

#step1: prepare path
path = './flower_photos/'
model_path = './model/'
model_name = 'CNN_model'
tb_dir = './tbdir'

#step2: prepare the standard image parameters
w = 100
h = 100
c = 3 # channels

#step3: perform data reading and standarization
def read_img(path):
	cate = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]
	print(cate)
	imgs = []
	labels = []
	for idx, folder in enumerate(cate):
		for im in glob.glob(folder+'/*.jpg'):
			print('reading the images: %s'%(im))
			img=io.imread(im)
			img=transform.resize(img,(w,h))
			imgs.append(img)
			labels.append(idx)
	return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

data,label=read_img(path)

#step4: data pre-processing

num_example = data.shape[0]
arr = np.arrange(num_example)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]

ratio = 0.8
s = np.int(num_example*ratio)
x_train = data[:s] 
y_train = label[:s]
x_val = data[s:]
y_val = label[s:]
print(len(x_train))
print(len(x_val))

#step5: placeholder defining
x = tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_ = tf.placeholder(tf.float32,shape=[None,],name='y_')

#step6: CNN construction
def inference(input_tensor, train, regularizer):
	#1st volume layer
	with tf.variable_scope('layer1-conv1'):
		conv1_weights = tf.get_variable("weight",[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable("bias",[32],initializer=tf.constant_initializer(0.0))
		conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
	#2nd hidden layer
	with tf.name_scope('layer2-pool1'):
		pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1], strides = [1,2,2,1], padding='VALID')
	#3rd hidden layer
	with tf.variable_scope('layer3-conv2'):
		conv2_weights = tf.get_variable("weight",[5,5,32,64], initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable("bias",[64],initializer=tf.constant_initializer(0.0))
		conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1],padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
	#4th hidden layer
	with tf.name_scope('layer4-pool2'):
		pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
		nodes = 25*25*64
		reshaped = tf.reshape(pool2, [-1,nodes])
	#5th hidden layer
	with tf.variable_scope('layer5-fc1'):
		fc1_weights = tf.get_variable("weight",[nodes,1024],initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses',regularizer(fc1_weights))
		fc1_biases = tf.get_variable("bias",[1024],initializer=tf.constant_initializer(0.1))
		fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
		if train: fc1 = tf.nn.dropout(fc1, 0.5)
	#6th hidden layer
	with tf.variable_scope('layer6-fc2'):
		fc2_weights = tf.get_variable("weight",[1024,5],initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses' regularizer(fc2_weights))
		fc2_biases = tf.get_variable("bias",[5],initializer=tf.constant_initializer(0.1))
		logit = tf.matmul(fc1,fc2_weights) + fc2_biases

	return logit
	
#define regular terms
regularizer = tf.contrib.layers.l2_regularizer(0.001)
#view the model
logits = model(x,False,regularizer)
print("shape of logits:", logits.shape)
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval')

#Step7: Loss function, Optimizer and verification indicator defining

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#Step8: Training and validation
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield inputs[excerpt], targets[excerpt]


n_epoch = 10 #prepare training parameters
batch_size = 64
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(n_epoch):
	print("epoch", epoch + 1) #define model operation on training and verification set
	start_time = time.time()

	#training
	train_loss, train_Acc, n_batch = 0, 0, 0
	for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
		_,err,ac = sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
		train_loss += err; train_acc += ac; n_batch += 1
	print("		train loss: %f" % (np.sum(train_loss)/n_batch))
	print("		train acc: %f" % (np.sum(train_acc)/n_batch))

	#validation
	val_loss, val_acc, n_batch = 0, 0, 0
	for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
	err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
	val_loss += err; val_acc += ac; n_batch += 1
	print("		validation loss: %f" % (np.sum(val_loss)/ n_batch))
	print("		validation acc: %f" % (np.sum(val_acc)/ n_batch))
	print(" 	epoch time: %f" % (time.time()-start_time))
	print('------------------------------------------------------------')

#Step9: Saving and session closing
saver.save(sess, model_path)
sess.close()

#-----------------SAMPLE IMAGE TEST ----------------------------

#Import Modules
ath1 = "flower_photos/daisy/5547758_eea9edfd54_n.jpg"
path2 = "flower_photos/dandelion/7355522_b66e5d3078_m.jpg"
path3 = "flower_photos/roses/394990940_7af082cf8d_n.jpg"
path4 = "flower_photos/sunflowers/6953297_8576bf4ea3.jpg"
path5 = "flower_photos/tulips/10791227_7168491604.jpg"

#Generate a type dictionary
flower_dict = {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}

#Define the image standarization function
w = 100
h = 100
c = 3

def read_one_image(path):
	img = io.imread(path)
	img = transform.resize(img,(w,h))
	return np.asarray(img)

#Standard the test data
with tf.Session() as sess:
	data = []
	data1 = read_one_image(path1)
	data2 = read_one_image(path2)
	data3 = read_one_image(path3)
	data4 = read_one_image(path4)
	data5 = read_one_image(path5)
	data.append(data1)
	data.append(data2)
	data.append(data3)
	data.append(data4)
	data.append(data5)

#Reload the model
saver = tf.train.import_meta_graph('Model/model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('Model'))

#Output model parameters
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")

#invoke the model
feed_dict = {x:data}
logits = graph.get_tensor_by_name("logits_eval:0")
classification_result = sess.run(logits, feed_dict)

print(classification_result)
print(tf.argmax(classification_result,1).eval())

#print test result
output = []
output = tf.argmax(classification_result, 1).eval()
for i in range(len(output)):
	print("flower", i+1, "prediction:" + flower_dict[output[i]])