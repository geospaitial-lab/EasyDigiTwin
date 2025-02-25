class Callback:

    def on_train_begin(self):
        # Gets called at the beginning of training
        pass

    def on_train_end(self):
        # Gets called at the end of training
        pass

    def on_epoch_begin(self, epoch):
        # Gets called at the beginning of an epoch
        pass

    def on_epoch_end(self, epoch):
        # Gets called at the end of an epoch
        pass

    def on_step_begin(self, global_step):
        # Gets called at the beginning of a step
        pass

    def on_step_end(self, global_step):
        # Gets called at the end of a step
        pass

    def on_render(self, render):
        # Gets called when scene is rendered
        pass

    def on_backward(self, global_step, grad_scale=1.0):
        # Gets called at the end of backward
        pass

    def on_pre_optimizers_step(self, global_step, optimizers):
        # Gets called at the beginning of an optimizer step
        pass

    def on_post_optimizers_step(self, global_step, optimizers):
        # Gets called at the end of an optimizer step
        pass


class CallbackList:
    def __init__(self):
        self.callbacks = []

    def add_callbacks(self, callbacks):
        """
        :param list[Callback] callbacks:
        :return:
        """

        self.callbacks += callbacks

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_step_begin(self, global_step):
        for callback in self.callbacks:
            callback.on_step_begin(global_step)

    def on_step_end(self, global_step):
        for callback in self.callbacks:
            callback.on_step_end(global_step)

    def on_render(self, render):
        for callback in self.callbacks:
            callback.on_render(render)

    def on_backward(self, global_step, grad_scale=1.0):
        for callback in self.callbacks:
            callback.on_backward(global_step, grad_scale=grad_scale)

    def on_pre_optimizers_step(self, global_step, optimizers):
        for callback in self.callbacks:
            callback.on_pre_optimizers_step(global_step, optimizers)

    def on_post_optimizers_step(self, global_step, optimizers):
        for callback in self.callbacks:
            callback.on_post_optimizers_step(global_step, optimizers)
