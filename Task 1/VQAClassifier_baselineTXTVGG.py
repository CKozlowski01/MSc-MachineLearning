#####################################################
# VQAClassifier_baseline.py
# 
# Program adapted and rewriten from the following sources:
# https://www.tensorflow.org/tutorials
# https://pypi.org/project/sentence-transformers/
#
# See file README.txt for further details.
# 
# Version: 1.0 
# Date: 01 March 2023
# Contact: hcuayahuitl@lincoln.ac.uk
#####################################################

from reprlib import Repr
import sys
import os
import time
import math
import einops
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from official.nlp import optimization
import matplotlib.pyplot as plt



epochs = 40
learning_rate = 0.001
model_arch_file = 'model_arch_vqa_baseline.png'
class_names = {'yes', 'no'}
num_classes = len(class_names)
classifier_model = None
history = None
classifier_model_name = 'VQA_Classifier'
AUTOTUNE = tf.data.AUTOTUNE
patch_size = 6
num_patches = (72 // patch_size) ** 2
projection_dim = 32
num_heads = 2
transformer_units = [
projection_dim * 2,
projection_dim,
]  # Size of the transformer layers
transformer_layers = 2
#mlp_head_units = [512, 128]  # Size of the dense layers of the final classifier

class VQA_DataLoader():
    BATCH_SIZE = 16
    IMAGE_SIZE = (72, 72)
    IMAGE_SHAPE = (72, 72, 3)
    SENTENCE_EMB_SHAPE = (384)
    IMAGES_PATH = r"C:\Users\tatam\Desktop\assessment ML\images\val2014-resised"
    train_data_file = IMAGES_PATH+"\\..\\vaq2.0.TrainImages.txt"
    dev_data_file = IMAGES_PATH+"\\..\\vaq2.0.DevImages.txt"
    test_data_file = IMAGES_PATH+"\\..\\vaq2.0.TestImages.txt"
    sentence_embeddings_file = "vaq2.cmp9137.sentence_transformers.txt"
    sentence_embeddings = {}
    train_ds = None
    val_ds = None
    test_ds = None

    def __init__(self):
        self.sentence_embeddings = self.load_sentence_embeddings()
        self.train_ds = self.load_classifier_data(self.train_data_file)
        self.val_ds = self.load_classifier_data(self.dev_data_file)
        self.test_ds = self.load_classifier_data(self.test_data_file)

    # Sentence embeddings are dense vectors representing text data, one vector per sentence. 
    # Sentences with similar vectors would mean sentences with equivalent meanning.  
	# They are useful here to provide text-based features of questions in the data.
    # Note: sentence embeddings don't include answers, they are solely based on the question.
    def load_sentence_embeddings(self):
        sentence_embeddings = {}
        print("READING sentence embeddings...")
        with open(self.sentence_embeddings_file, 'rb') as f:
            data = pickle.load(f)
            for sentence, dense_vector in data.items():
                sentence_embeddings[sentence] = dense_vector
        print("Done reading sentence_embeddings!")
        return sentence_embeddings

    # In contrast to text-data based on pre-trained features, image data does not use
    # any form of pre-training in this program. Instead, it makes use of raw pixels.
    # Notes: 
	#       (1) The question and answer are not used for training, only for testing.
    #       (2) Input features to the classifier are only pixels and sentence embeddings.
    def process_input(self, img_path, dense_vector, text, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.IMAGE_SIZE)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.cast(img, tf.float32) / 255
        features = {}
        features["image_input"] = img
        features["text_embedding"] = dense_vector
        features["question_answer"] = text
        features["file_name"] = img_path
        return features, label

    # This method load the multimodal data, which comes from the following sources:
    # (1) image files in IMAGES_PATH, and (2) files with pattern vaq2.0.*Images.txt
    # The data is stored in a tensorflow data structure to make it easy to use by
    # the tensorflow model during training, validation and test.
    # This method was carefully prepared to load the data rapidly, i.e., by loading
    # already creating sentence embeddings rather than creating them at runtime.
    def load_classifier_data(self, data_files):
        print("LOADING data from "+str(data_files))
        print("=========================================")
        image_data = []
        text_data = []
        embeddings_data = []
        label_data = []
		
        # get image, text, label of image_files
        with open(data_files) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip("\n")
                img_name, question_answer = line.split("\t")
                img_name = img_name.split("#")[0]
                img_name = os.path.join(self.IMAGES_PATH, img_name.strip())

                # get binary labels from yes/no answers
                label = [1, 0] if question_answer.endswith("? yes") else [0, 1]

				# get sentence embeddings WITHOUT answers
                text = question_answer.replace("? yes", "?")
                text = text.replace("? no", "?")
                text_sentence_embedding = self.sentence_embeddings[text]
                text_sentence_embedding = tf.constant(text_sentence_embedding)

                image_data.append(img_name)
                embeddings_data.append(text_sentence_embedding)
                text_data.append(question_answer)
                label_data.append(label)

        print("|image_data|="+str(len(image_data)))
        print("|text_data|="+str(len(text_data)))
        print("|label_data|="+str(len(label_data)))
		
        # prepare a tensorflow dataset using the lists generated above
        dataset = tf.data.Dataset.from_tensor_slices((image_data, embeddings_data, text_data, label_data))
        dataset = dataset.shuffle(self.BATCH_SIZE * 8)
        dataset = dataset.map(self.process_input, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(self.BATCH_SIZE).prefetch(AUTOTUNE)
        self.print_data_samples(dataset)
        return dataset

    def print_data_samples(self, dataset):
        print("PRINTING data samples...")
        print("-----------------------------------------")
        for features_batch, label_batch in dataset.take(1):
            for i in range(1):
                print(f'Image pixels: {features_batch["image_input"]}')
                print(f'Sentence embeddings: {features_batch["text_embedding"]}')
                print(f'Question-answer: {features_batch["question_answer"].numpy()}')
                label = label_batch.numpy()[i]
                print(f'Label : {label}')
        print("-----------------------------------------")

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# Build the diagonal attention mask
diag_attn_mask = 1 - tf.eye(num_patches)
diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)

class ShiftedPatchTokenization(layers.Layer):
    def __init__(
        self,
        image_size=72,
        patch_size=patch_size,
        num_patches=(72 // patch_size) ** 2,
        projection_dim=projection_dim,
        vanilla=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vanilla = vanilla  # Flag to swtich to vanilla patch extractor
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        self.flatten_patches = layers.Reshape((num_patches, -1))
        self.projection = layers.Dense(units=projection_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def crop_shift_pad(self, images, mode):
        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return shift_pad

    def call(self, images):
        if not self.vanilla:
            # Concat the shifted images with the original image
            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )
        # Patchify the images and flatten it
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        if not self.vanilla:
            # Layer normalize the flat patches and linearly project it
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            # Linearly project the flat patches
            tokens = self.projection(flat_patches)
        return (tokens, patches)

class PatchEncoder(layers.Layer):
    def __init__(
        self, num_patches=num_patches, projection_dim=projection_dim, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def call(self, encoded_patches):
        encoded_positions = self.position_embedding(self.positions)
        encoded_patches = encoded_patches + encoded_positions
        return encoded_patches

class MultiHeadAttentionLSA(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The trainable temperature term. The initial value is
        # the square root of the key dimension.
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class VQA_Classifier(VQA_DataLoader):
	
    def __init__(self):
        super().__init__()
        self.build_classifier_model()
        self.train_classifier_model()
        self.test_classifier_model()
        self.plot_learning_curves()

    def create_vision_encoder(self, num_projection_layers, projection_dims, dropout_rate):

        img_input = layers.Input(shape=self.IMAGE_SHAPE, name="image_input")

        cnn_layer = layers.Conv2D(16, 3, padding='same', activation='relu')(img_input)
        cnn_layer = layers.Conv2D(16, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Conv2D(24, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.Conv2D(24, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Conv2D(36, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.Conv2D(36, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.Conv2D(36, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Flatten()(cnn_layer)

        outputs = self.project_embeddings(cnn_layer, num_projection_layers, projection_dims, dropout_rate)
        return img_input, outputs

    def project_embeddings(self, embeddings, num_projection_layers, projection_dims, dropout_rate):
        projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
        for _ in range(num_projection_layers):
            x = tf.nn.gelu(projected_embeddings)
            x = layers.Dense(projection_dims, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Add()([projected_embeddings, x])
            projected_embeddings = layers.LayerNormalization()(x)
        return projected_embeddings

    def create_text_encoder(self, num_projection_layers, projection_dims, dropout_rate):
        text_input = keras.Input(shape=self.SENTENCE_EMB_SHAPE, name='text_embedding')
        outputs = self.project_embeddings(text_input, num_projection_layers, projection_dims, dropout_rate)
        return text_input, outputs

    def create_text_encoder2(self, num_projection_layers, projection_dims, dropout_rate):
        vocab_size = 20000  # Only consider the most frequent 20k words
        maxlen = 300  # Only consider the first 200 words of each movie review

        embed_dim = 128  # Embedding size for each token
        num_heads = 8  # Number of attention heads
        ff_dim = 128  # Hidden layer size in feed forward network inside transformer

        text_input = keras.Input(shape=self.SENTENCE_EMB_SHAPE, name='text_embedding')
        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        x = embedding_layer(text_input)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.2)(x)
        #outputs = layers.Dense(2, activation="softmax")(x)

        outputs = self.project_embeddings(x, num_projection_layers, projection_dims, dropout_rate)
        return text_input, outputs

    def build_classifier_model(self):
        print(f'BUILDING model')
        img_input, vision_net = self.create_vision_encoder (num_projection_layers=1, projection_dims=128, dropout_rate=0.1)
        text_input, text_net = self.create_text_encoder2(num_projection_layers=1, projection_dims=128, dropout_rate=0.1)
        net = tf.keras.layers.Concatenate(axis=1)([vision_net, text_net])
        net = tf.keras.layers.Dropout(0.2)(net)
        net = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name=classifier_model_name)(net)
        self.classifier_model = tf.keras.Model(inputs=[img_input, text_input], outputs=net)
        self.classifier_model.summary()
	
    def train_classifier_model(self):
        print(f'TRAINING model')
        steps_per_epoch = tf.data.experimental.cardinality(self.train_ds).numpy()
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.2*num_train_steps)

        loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
        metrics = tf.keras.metrics.BinaryAccuracy()
        optimizer = optimization.create_optimizer(init_lr=learning_rate,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')


        self.classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # the next two lines if you want the neural architecture to be generated in a file, which 
		# requires installing graphviz (from https://www.graphviz.org/download/) and pip install pydot       
        #tf.keras.utils.plot_model(self.classifier_model, to_file=self.model_arch_file, show_shapes=False, show_layer_names =True)
        #print("File %s created!" % (self.model_arch_file))

        # uncomment the next line if you wish to make use of early stopping during training
        #callbacks = [tf.keras.callbacks.EarlyStopping(patience=11, restore_best_weights=True)]

        self.history = self.classifier_model.fit(x=self.train_ds, validation_data=self.val_ds, epochs=epochs)#, callbacks=callbacks)
        print("model trained!")

    def test_classifier_model(self):
        print("TESTING classifier model (showing a sample of question-answer-predictions)...")
        num_classifications = 0
        num_correct_predictions = 0

        # read test data for VQA classification
        for features, groundtruth in self.test_ds:
            groundtruth = groundtruth.numpy()
            predictions = self.classifier_model(features)
            predictions = predictions.numpy()
            question_answers = features["question_answer"].numpy()
            file_names = features["file_name"].numpy()

            # read test data per batch
            for batch_index in range(0, len(groundtruth)):
                predicted_values = predictions[batch_index]
                probability_yes = predicted_values[0]
                probability_no = predicted_values[1]
                predicted_class = "[1 0]" if probability_yes > probability_no else "[0 1]"
                if str(groundtruth[batch_index]) == predicted_class: 
                    num_correct_predictions += 1
                num_classifications += 1

                # print a sample of predictions, e.g. 10% of all possible
                if random.random() < 0.1:
                    question_answer = question_answers[batch_index]
                    file_name = file_names[batch_index].decode("utf-8")
                    file_name_index = file_name.index("COCO") 
                    file_name = file_name[file_name_index:]
                    print("QA=%s PREDICTIONS: yes=%s, no=%s \t -> \t %s" % \
                             (question_answer, probability_yes, probability_no, file_name))

        # reveal test performance using our own calculations above
        accuracy = num_correct_predictions/num_classifications
        print("TEST accuracy=%4f" % (accuracy))

        # reveal test performance using Tensorflow calculations
        loss, accuracy = self.classifier_model.evaluate(self.test_ds)
        print(f'Tensorflow test method: Loss: {loss}; ACCURACY: {accuracy}')

    def plot_learning_curves(self):
        history_dict = self.history.history
        print("history_dict="+str(history_dict))
        acc = history_dict['binary_accuracy']
        val_acc = history_dict['val_binary_accuracy']

        epochs = range(1, len(acc) + 1)
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()

        plt.plot(epochs, acc, 'r', label='Training ACC', linestyle='solid')
        plt.plot(epochs, val_acc, 'b', label='Validation ACC', linestyle='dashdot')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend(loc='lower right')
        plt.legend(fontsize=14)
        plt.show()

d = VQA_Classifier()
