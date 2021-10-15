import tensorflow as tf
import numpy as np
import os


class Save_Training_Evolution(tf.keras.callbacks.Callback):

    def __init__(self, filename, append_text=False):
        super().__init__()
        
        self.first_call = True
        self.filename = filename
        # Specify if evolution metric file has to be deleted (append_text=False) or keeped (append_text=True)
        self.append_text = append_text
    
    def on_epoch_end(self, epoch, logs=None):
        # On first call (end of first epoch), create a new text file, with headers (one per metric)
        # This file will have a csv file structure (comma separation)
        if self.first_call:
            self.first_call = False
            if not self.append_text:
                if os.path.exists(self.filename):
                    os.remove(self.filename)
                with open(self.filename, 'a') as f:
                    f.write(','.join(k for k in logs) + '\n')
        # Write on last line current metrics
        with open(self.filename, 'a') as f:
            f.write(','.join(list(map(str, [logs[k] for k in logs]))) + '\n')


'''
Function utilized to decrease learning rate by a given factor
'''
def lr_modifier(epoch, lr, num_steps, factor=0.9, verbose=True):
    # Erróneamente llamado num_steps y no num_epochs
    if epoch != 0 and epoch % num_steps == 0:
        new_lr = lr * factor
        if verbose:
            print('Epoch', str(epoch).zfill(4),'--- Learning rate reduced to:', new_lr)
        return new_lr
    return lr

def create_scheduler_function(num_steps, factor=0.9, verbose=True):

    def scheduler(epoch, learning_rate):
        return lr_modifier(epoch, learning_rate, num_steps, factor, verbose)

    return scheduler

def create_scheduler_function_dynamic_range(initial_num_steps, increment, factor, verbose=True):

    # This function will specify in when the learning_rate must be changed
    # f(n) = initial_num_steps + (n-1)*increment + f(n-2)

    # solving this equation, it will have this form:
    # f(n) = c0 + c1*n + c2*n^2

    # initial_num_steps = 20, increment = 5
    # n -- number of changes
    # n = 1 --> first change after 20 epochs (base statement)
    # n = 2 --> second change after 20 + 5 + 20 = 45 epochs
    # n = 3 --> 75 epochs
    # n = 4 --> 110 epochs
    # ...

    f = np.zeros(4, dtype=np.float32)
    f[0] = initial_num_steps

    # Get some real values (referring to n=2,3,4)
    for i in range(1,4):
        f[i] = initial_num_steps + i*increment + f[i-1]

    # { c0 + 2·c1 + 2^2·c2 = f(2) - in code --> f[1]
    # { c0 + 3·c1 + 3^2·c2 = f(3) ------------> f[2]
    # { c0 + 4·c1 + 4^2·c2 = f(4) ------------> f[3]
    #  ---------A---------   --B--

    # Solve system of linear equations using numpy.linalg.solve
    A = np.array([[1,2,4],[1,3,9],[1,4,16]])
    B = f[1:]

    coefficients = np.linalg.solve(A, B)
    coefficients = coefficients[::-1] # Reverse into: n^2, n^1, n^0

    # https://stackoverflow.com/questions/42179087/get-the-inverse-function-of-a-polyfit-in-numpy
    p = np.poly1d(coefficients)

    def num_of_changes_at_epoch(epoch):
        # Calculates the inverse of the quadratic equation
        return np.floor((p - epoch).roots[-1])

    def learning_rate_modifier_function(epoch, learning_rate):
        if epoch > 0 and num_of_changes_at_epoch(epoch) != num_of_changes_at_epoch(epoch-1):
            # If this has changed, it's time to reduce learning_rate again
            if verbose:
                print('Epoch', str(epoch).zfill(4),'--- Learning rate reduced to:', learning_rate * factor)
            return learning_rate * factor
        else:
            return learning_rate

    return learning_rate_modifier_function