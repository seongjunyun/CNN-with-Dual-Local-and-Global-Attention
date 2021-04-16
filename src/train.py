import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data_loader
import model


def evaluation(target, cf_out):
    fpr, tpr, _ = metrics.roc_curve(target, cf_out)
    auc = metrics.auc(fpr, tpr)
    return auc


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


# Hyper Parameters
input_size = 10000
num_epochs = 10
batch_size = 32
learning_rate = 1e-4

print("Loading data...")

# Data Loader (Input Pipeline)


train_loader = data_loader.get_loader('train.csv', batch_size)

val_loader = data_loader.get_loader('validation.csv', batch_size, shuffle=False)
test_loader = data_loader.get_loader('test.csv', batch_size, shuffle=False)

print("train/val/test/: {:d}/{:d}/{:d}".format(len(train_loader), len(val_loader), len(test_loader)))
print("==================================================================================")

CNNDLGA = model.CNNDLGA(input_size)

if torch.cuda.is_available():
    CNNDLGA.cuda()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(CNNDLGA.parameters(), lr=learning_rate)

print("==================================================================================")
print("Training Start..")

batch_loss = 0
# Train the Model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (user, item, labels) in enumerate(train_loader):

        # Convert torch tensor to Variable
        batch_size = len(user)

        user = to_var(user)
        item = to_var(item)
        labels = to_var(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = CNNDLGA(user, item)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_loss += loss.data[0]
        if i % 10 == 0:
            # Print log info
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                  % (epoch, num_epochs, i, total_step,
                     batch_loss / 10, np.exp(loss.data[0])))
            batch_loss = 0

    # Save the Model
    torch.save(CNNDLGA.state_dict(), 'model_' + str(epoch) + '.pkl')

print("==================================================================================")
print("Training End..")

print("==================================================================================")
print("Testing Start..")

for i, (user, item, labels) in enumerate(val_loader):

    # Convert torch tensor to Variable
    batch_size = len(user)
    user = to_var(user)
    item = to_var(item)
    labels = to_var(labels)

    outputs = CNNDLGA(user, item)
    if i == 0:
        result = outputs.data.cpu().numpy()
    else:
        print
        len(result)
        result = np.append(result, outputs.data.cpu().numpy(), axis=0)

pickle.dump(result, open('result_val.pickle', 'wb'))

print("==================================================================================")
print("Testing End..")
