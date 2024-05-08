import matplotlib.pyplot as plt
import numpy as np


class TrainingDataCollector:
    def __init__(self):
        self.train_loss = np.full((0,), -1)
        self.train_acc = np.full((0,), -1)
        self.val_loss = np.full((0,), -1)
        self.val_acc = np.full((0,), -1)
        self.test_loss = None
        self.test_acc = None

    def get_statistics(self):
        return np.vstack((self.train_loss, self.train_acc,
                          self.val_loss, self.val_acc))

    def update_statistics(self, trl:float, tra:float, vl:float, va:float):
        self.train_loss = np.hstack((self.train_loss, np.array([trl])))
        self.train_acc = np.hstack((self.train_acc, np.array([tra])))
        self.val_loss = np.hstack((self.val_loss, np.array([vl])))
        self.val_acc = np.hstack((self.val_acc, np.array([va])))

    def add_test_statistics(self, tel:float, tea:float):
        self.test_loss = tel
        self.test_acc = tea


class TrainingDataPlotter:
    def __init__(self, datacollector:TrainingDataCollector):
        self.tdc = datacollector

    def plot_statistics(self):
        plt.style.use('ggplot')
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))
        num_epochs = len(self.tdc.train_loss)
        epochs = np.arange(1, num_epochs + 1)
        axs[0].plot(epochs, self.tdc.train_loss, 'r', label='training loss')
        axs[0].plot(epochs, self.tdc.val_loss, 'b', label='validation loss')
        axs[0].set_title('training and validation loss')
        axs[0].set_xlabel('epochs')
        axs[0].set_ylabel('loss')

        axs[1].set_ylim(0, 1)
        axs[1].plot(epochs, self.tdc.train_acc, 'r', label='training accuracy')
        axs[1].plot(epochs, self.tdc.val_acc, 'b', label='validation accuracy')
        axs[1].set_title('training and validation accuracy')
        axs[1].set_xlabel('epochs')
        axs[1].set_ylabel('accuracy')

        if self.tdc.test_loss is not None:
            axs[0].plot(num_epochs, self.tdc.test_loss, 'go', label='test loss')
        if self.tdc.test_acc is not None:
            axs[1].plot(num_epochs, self.tdc.test_acc, 'go', label='test accuracy')

        axs[0].legend()
        axs[1].legend()
        plt.show()

if __name__ == "__main__":
    tdc = TrainingDataCollector()
    for _ in range(10):
        trl, tra, vl, va = np.random.random(4)
        tdc.update_statistics(trl, tra, vl, va)
    tel, tea = np.random.random(2)
    tdc.add_test_statistics(tel, tea)
    tdp = TrainingDataPlotter(tdc)
    tdp.plot_statistics()
