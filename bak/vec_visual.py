import os
import tensorflow as tf
import numpy as np
import fasttext
from tensorboard.plugins import projector

# load model
word2vec = fasttext.load_model('../models/fasttext_model_0804_09_skipgram.bin')

# create a list of vectors
dim = len(word2vec.get_words())
embedding = np.empty((len(word2vec.get_words()), word2vec.get_dimension()), dtype=np.float32)
for i, word in enumerate(word2vec.get_words()):
    embedding[i] = word2vec.get_word_vector(word)

# setup a TensorFlow session
tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='browse_history_embedding')

place = tf.placeholder(tf.float32, shape=embedding.shape)
set_x = tf.assign(X, place, validate_shape=False)
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: embedding})

# write labels
with open(os.path.join('../log', 'metadata.tsv'), 'w') as f:
    for word in word2vec.get_words():
        f.write(word + '\n')

# create a TensorFlow summary writer
summary_writer = tf.summary.FileWriter('log', sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'browse_history_embedding:0'
embedding_conf.metadata_path = os.path.join('../log', 'metadata.tsv')
projector.visualize_embeddings(summary_writer, config)

# save the model
saver = tf.train.Saver()
saver.save(sess, os.path.join('../log', "model.ckpt"))