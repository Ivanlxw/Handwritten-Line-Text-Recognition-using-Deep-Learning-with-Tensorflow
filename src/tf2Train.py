import os, sys
import time
import tensorflow as tf
from DataLoader import FilePaths, DataLoader
from tf2Custom import Model, DecoderType, ctc_loss

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=11998)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

if __name__ == "__main__":
    decoderType = DecoderType.BestPath
    loader = DataLoader(FilePaths.fnTrain, Model.batchSize,
                Model.imgSize, Model.maxTextLen, load_aug=True) 

    # tfData = loader.getTfDataSet().shuffle(100000).batch(Model.batchSize, drop_remainder=True)
    tfData = tf.data.Dataset.from_generator(loader.tf_gen, output_types=(tf.float64, tf.string)) 
    tfData = tfData.shuffle(100000)

    custom = Model(open(FilePaths.fnCharList).read(), decoderType)
    custom_model = custom.get_model()

    lr = 0.001  # learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    checkpoint_dir = os.path.join(os.getcwd(), "../train_scratch")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=custom_model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    EPOCHS = 400
    for epoch in range(EPOCHS):
        start = time.time()
        print("\nStart of epoch %d" % (epoch,))
        for step, (X_batch, y_batch) in enumerate(tfData):
            with tf.GradientTape() as tape:
                logits = custom_model(X_batch, training=True)

                ## loss calculation
                sparse = loader.tensorBatchtoSparse(y_batch.numpy())
                y_sparse = tf.sparse.SparseTensor(sparse[0], sparse[1], sparse[2])
                loss = ctc_loss(y_sparse, logits)
                
                grads = tape.gradient(loss, custom_model.trainable_weights)
                optimizer.apply_gradients(zip(grads, custom_model.trainable_weights))

                # Log every 50 batches.
                if step % 100 == 0:
                    print(f"Epoch: {epoch}, step: {step}")
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss))
                    )
                    if decoderType == DecoderType.BestPath: 
                        decoded = tf.nn.ctc_greedy_decoder(
                            inputs=logits, sequence_length=[Model.maxTextLen]*Model.batchSize)
                    else: 
                        decoded = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=[Model.maxTextLen]*Model.batchSize, beam_width=50, merge_repeated=True)
                    print("Decoded sample: ", custom.decoderToText(decoded))

        ## save every 40 epoch
        if EPOCHS % 50 == 0:
            sys.stdout.flush()
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            # checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=custom_model)
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f"Time taken: {time.time() - start}")
