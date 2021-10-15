import tensorflow.keras as keras
import tensorflow as tf



class Train_Loss_LR(keras.Model):
    '''
    Training and test steps are modified in such a way only loss is computed on LR images.
    '''

    def train_step(self, data):
        # Unpack data
        x, y = data

        lr_size = tf.shape(x)[1:3]

        # Get model's output inside GradientTape
        with tf.GradientTape() as tp:
            
            # Get superresolved image
            sr = self(x, training=True)
            # Downsample
            downscaled = tf.image.resize(sr, lr_size, method='bicubic', antialias=True)
            # Compute loss on low resolution images
            loss_lr = self.compiled_loss(x, downscaled)

        # Compute gradients
        gradients = tp.gradient(loss_lr, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute metrics on high resolution image
        self.compiled_metrics.update_state(y, sr)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss_lr
        return results

    def test_step(self, data):
        # Unpack data
        x, y = data

        lr_size = tf.shape(x)[1:3]

        # Get superresolved image
        sr = self(x, training=True)
        # Downsample
        downscaled = tf.image.resize(sr, lr_size, method='bicubic', antialias=True)
        # Compute loss over low resolution images
        loss_lr = self.compiled_loss(x, downscaled)

        # Compute metrics on low resolution image
        self.compiled_metrics.update_state(y, sr)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss_lr
        return results


class Train_Loss_LR_and_HR(keras.Model):

    def train_step(self, data):
        # Unpack data
        x, y = data

        lr_size = tf.shape(x)[1:3]

        # Get model's output inside GradientTape
        with tf.GradientTape() as tp:
            
            # Get superresolved image
            sr = self(x, training=True)
            # Downsample
            downscaled = tf.image.resize(sr, lr_size, method='bicubic', antialias=True)
            # Compute loss (MAE) over low resolution images
            loss_lr = self.compiled_loss(x, downscaled)
            # Compute loss (MAE) over high resolution images
            loss_hr = self.compiled_loss(y, sr)

            # Combine losses
            loss = loss_lr + loss_hr

        # Compute gradients
        gradients = tp.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute metrics on high resolution image
        self.compiled_metrics.update_state(y, sr)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        return results

    def test_step(self, data):
        # Unpack data
        x, y = data

        lr_size = tf.shape(x)[1:3]

        # Get superresolved image
        sr = self(x, training=True)
        # Downsample
        downscaled = tf.image.resize(sr, lr_size, method='bicubic', antialias=True)
        # Compute loss (MAE) over low resolution images
        loss_lr = self.compiled_loss(x, downscaled)
        # Compute loss (MAE) over high resolution images
        loss_hr = self.compiled_loss(y, sr)

        # Combine losses
        loss = loss_lr + loss_hr

        # Compute metrics on low resolution image
        self.compiled_metrics.update_state(y, sr)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        return results