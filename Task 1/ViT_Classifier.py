##############################################
# Program extended from https://keras.io/examples/vision/image_classification_with_vision_transformer/
##############################################

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import os
import sys
import pathlib
import matplotlib.pyplot as plt

def read_image_data(data_path, verbose=False):
    count=1
    X = []
    Y = []
    data_dir = pathlib.Path(data_path)
    print("Reading folder="+str(data_dir))
    dataset = tf.keras.utils.image_dataset_from_directory(data_dir)
    for image_batch, labels_batch in dataset:
        labels = labels_batch.numpy()
        if (verbose): print("Batch labels -> "+str(labels))
        for i in range(0, len(image_batch)):
            image = image_batch[i]
            label = labels[i]
            X.append(image)
            Y.append([label])
            if (verbose): print("["+str(count)+"] image="+str(image.shape)+" label="+str(label))
            count += 1
      
    X = np.array(X)
    Y = np.array(Y)
    class_names = dataset.class_names
    
    print("X="+str(X.shape))
    print("Y="+str(Y.shape))
    print("class_names="+str(class_names))
    
    return X, Y, class_names

train_data_dir = "C:\\Users\\Computing\\Downloads\\flower_photos-resised\\flower_photos-resised_train"
test_data_dir = "C:\\Users\\Computing\\Downloads\\flower_photos-resised\\flower_photos-resised_test"
x_train, y_train, class_names = read_image_data(train_data_dir)
x_test, y_test, class_names = read_image_data(test_data_dir)

num_classes = len(class_names)
input_shape = (256,256,3)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 40
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
#mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
#mlp_head_units = [1024, 768]  # Size of the dense layers of the final classifier
mlp_head_units = [512, 128]  # Size of the dense layers of the final classifier

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
    
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
        
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
        
def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    model.summary()
    return model
    
def run_experiment(model, exec_mode, class_names):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    
    checkpoint_filepath = "tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    if exec_mode=='train':
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=0.1,
            callbacks=[checkpoint_callback],
        )
        
        return history

    else:
        model.load_weights(checkpoint_filepath)
        _, accuracy = model.evaluate(x_test, y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

        num_correct_predictions = 0
        num_total_predictions = 0
        labels = os.listdir(test_data_dir)
        for label in labels:
            filePath = test_data_dir+"/"+label
  
            files = os.listdir(filePath)
            for fileName in files:
  
                if fileName.endswith(".jpg") or fileName.endswith(".png"):
                    imagefile_path = filePath+"/"+fileName
                    img = tf.keras.utils.load_img(imagefile_path, target_size=(256, 256))
                    img_array = tf.keras.utils.img_to_array(img)
                    img_array = tf.expand_dims(img_array, 0)
                    predictions = model.predict(img_array)
                    scores = tf.nn.softmax(predictions[0])
                    predicted_class = tf.nn.softmax(scores)
                    predicted_class = class_names[np.argmax(scores)]
                    print(str(imagefile_path)+" class="+predicted_class+" prob.="+str(np.max(scores)))
                    
                    if predicted_class==label:
                        num_correct_predictions += 1
                    num_total_predictions += 1

        accuracy = num_correct_predictions/num_total_predictions
        print("Classification Accuracy="+str(accuracy))

        return None

if len(sys.argv)!=2:
    print("USAGE: python .\ViTClassifier.py [train|test]")
    exit(0)

else:
    exec_mode = sys.argv[1]
    print("exec_mode="+str(exec_mode))

    vit_classifier = create_vit_classifier()
    history = run_experiment(vit_classifier, exec_mode, class_names)