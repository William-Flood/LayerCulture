import tensorflow as tf

def provide_dataset(dataset_file_name, batch_size):
    example_dataset = tf.data.TFRecordDataset(filenames=[dataset_file_name])
    def map_example(example):
        parsed_example = tf.io.parse_example(example, {
            "inputs": tf.io.FixedLenFeature(shape=(), dtype=tf.string),
            "outputs": tf.io.FixedLenFeature(shape=(), dtype=tf.string)
        })
        inputs = tf.io.parse_tensor(parsed_example["inputs"], tf.float32)
        outputs = tf.io.parse_tensor(parsed_example["outputs"], tf.float32)
        return {"inputs": inputs, "outputs":outputs}

    # dataset_iter = iter(strategy.experimental_distribute_dataset(
    #   example_dataset.map(map_example).repeat().shuffle(1000).batch(options["batch_size"])
    # ))

    dataset_iter = iter(example_dataset.repeat().shuffle(2000).map(map_example).batch(batch_size))
    return dataset_iter