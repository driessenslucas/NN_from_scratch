# %%
from pandas.io.formats.printing import PrettyDict
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X.values
y= y.astype(int).values

# %%
print(X.shape)
print(y.shape)

# %%
# Now we should normalize the pixel values
X = ((X / 255.) - .5) * 2

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()
for i in range(10):
    img = X[y == i][0].reshape(28,28)
    ax[i].imshow(img,cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.show()

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)

ax = ax.flatten()
for i in range(25):
    img = X[y == 7][0].reshape(28,28)
    ax[i].imshow(img,cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# plt.show()

# %%
# Train test split
from sklearn.model_selection import train_test_split

X_temp, X_test, y_temp, y_test = train_test_split(X,y, test_size=10000, random_state=123, stratify=y)

X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)
# %%
# import the neural network we just made
from neuralnet import NeuralNetMLP, int_to_onehot

model = NeuralNetMLP(num_features=28*28, num_hidden=50, num_classes=10)

# %%
# Training loop
import numpy as np
num_epochs = 50
minibatch_size = 100

def minibatch_generator(X,y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_indx in range(0, indices.shape[0] - minibatch_size + 1,
        minibatch_size):
            batch_idx = indices[start_indx:start_indx + minibatch_size]
            yield X[batch_idx], y[batch_idx]


# %%
# Lets test if the mini batches are getting generated!

#for i in range(num_epochs):
#    # iterate of minibatches
#    minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)

#    for X_train_mini, y_train_mini in minibatch_gen:
#        break
#    break

#print(X_train_mini.shape)
#print(y_train_mini)
# %%
def mse_loss(targets, probas, num_labels=10):
    one_hot_targets = int_to_onehot(
        targets, num_labels=num_labels
    )
    return np.mean((one_hot_targets - probas)**2)

def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)

# %%
# Lets test the preceiding functions

#_, probas = model.forward(X_Valid)
#mse = mse_loss(y_valid, probas)
#print(f"Inital validation MSE: {mse:.1f}")

#predicted_labels = np.argmax(probas, axis=1)
#acc = accuracy(y_valid, predicted_labels)
#print(f"Inital Accuracy: {acc * 100:.1f}")
# %%
def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0,0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    num_minibatches = 0
    for features, targets in minibatch_gen:
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(
            targets, num_labels=num_labels
        )
        loss = np.mean((probas - onehot_targets)**2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
        num_minibatches += 1

    mse = mse/num_minibatches
    acc = correct_pred/num_examples
    return mse,acc

mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f"Initial valid MSE: {mse:.1f}")
print(f"Initial valid accuracy: {acc*100:.1f}%")
# %%
# Training

def train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1):
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):
        # iterate over minibatches
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
        for X_train_mini, y_train_mini in minibatch_gen:
            ### computing outputs ###
            a_h, a_out = model.forward(X_train_mini)

            ### Compute Gradients ###
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = \
                model.backward(X_train_mini, a_h, a_out, y_train_mini)

            ### Update weights ###
            model.weight_hidden -= learning_rate * d_loss__d_w_h
            model.bias_hidden -= learning_rate * d_loss__d_b_h
            model.weight_output -= learning_rate * d_loss__d_w_out
            model.bias_output -= learning_rate * d_loss__d_b_out

        ### Epoch Logs ###
        train_mse, train_acc = compute_mse_and_acc(
            model, X_train, y_train
        )
        valid_mse, valid_acc = compute_mse_and_acc(
            model, X_valid, y_valid
        )
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)

        print(f"Epoch: {e+1:03d}/{num_epochs:03d}s "
            f"| Train MSE: {train_mse:.2f} "
            f"| Train Acc: {train_acc:0.2f}% "
            f"| Valid Acc: {valid_acc:0.2f}% "
        )

    return epoch_loss, epoch_train_acc, epoch_valid_acc

# %%
# Actually training

np.random.seed(123)
epoch_loss, epoch_train_acc, epoch_valid_acc = train(
    model, X_train, y_train, X_valid, y_valid, num_epochs=50, learning_rate=0.1
)


# %%

plt.plot(range(len(epoch_loss)), epoch_loss)
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.show()

# %%

plt.plot(range(len(epoch_train_acc)), epoch_train_acc, label="Training")
plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc, label="Validation")
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="lower right")
plt.show()

# %%

valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
print(f"Validation accuracy: {valid_acc*100:.1f}%")

# %%

X_valid_subset = X_valid[:1000, :]
y_valid_subset = y_valid[:1000]

_, probas = model.forward(X_valid_subset)
test_pred = np.argmax(probas,axis=1)

misclassified_images = X_valid_subset[y_valid_subset != test_pred][:25]
misclassified_labels = test_pred[y_valid_subset != test_pred][:25]

corrrect_labels = y_valid_subset[y_valid_subset != test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(8,8))

ax = ax.flatten()

for i in range(25):
    img = misclassified_images[i].reshape(28,28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title(f'{i+1}) '
                    f'True: {corrrect_labels[i]}\n'
                    f' Predicted: {misclassified_labels[i]}') 
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
