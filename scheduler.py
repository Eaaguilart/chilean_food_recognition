import tensorflow as tf

class LearningRateScheduler(tf.keras.callbacks.Callback):
    """Planificador de Learning rate que define el learning rate deacuerdo a lo programado.

    Arguments:
        schedule: una funcion que toma el indice del epoch
            (entero, indexado desde 0) y el learning rate actual
            como entradas y regresa un nuevo learning rate como salida (float).
    """

    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Obtener el learning rate actua del optimizer del modelo.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Llamar la funcion schedule para obtener el learning rate programado.
        scheduled_lr = self.schedule(epoch, lr)
        # Definir el valor en el optimized antes de que la epoch comience
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print('\nEpoca %01d: Tasa de aprendizaje de %6.5f.' % (epoch, scheduled_lr))