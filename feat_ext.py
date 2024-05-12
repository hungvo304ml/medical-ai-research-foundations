import tensorflow_hub as hub
import tensorflow as tf


def preprocess_image(image_path):
    """ Loads image from path and preprocesses to make it model ready
        Args:
        image_path: Path to the image file
    """
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
            
    hr_image = tf.image.resize(hr_image, [224, 224])
    hr_image = tf.cast(hr_image, tf.float32)
    hr_image = hr_image/255
    hr_image = tf.clip_by_value(hr_image, 0., 1.)    

    return tf.expand_dims(hr_image, 0)



module = hub.load("physionet.org/files/medical-ai-research-foundation/1.0.0/path-50x1-remedis-m/")


# Pathology: The image is of shape (<BATCH_SIZE>, 224, 224, 3)
# Chest X-Ray: The image is of shape (<BATCH_SIZE>, 448, 448, 3)
# image = <LOAD_IMAGE_HERE>
image = preprocess_image("/home/hqvo2/Projects/Spatial_Transcriptomics/data/Gene_Exp_Pred/Processed_data_JYA/VISDS000020/cropped_image_1.png")

embedding_of_image = module(image)

print(embedding_of_image.shape, type(embedding_of_image.numpy()))