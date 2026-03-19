import os
import numpy as np

import ugrad
import ugrad.nn as nn
import ugrad.nn.functional as F

seed = 1717
np.random.seed(seed)
num_epochs = 3
batch_size = 128

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff')
X = X.reshape(-1, 1, 28, 28)
y = y.astype(np.int64)
X = X[:12000]
y = y[:12000]
y = np.eye(10)[y]
# Scale images to the [0, 1] range
X /= 255.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
print(f"{X_train.shape=} {X_test.shape=} {y_train.shape=} {y_test.shape=}")

print(f"{y_test[0]=}")

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_l1 = nn.Conv2d(1, 32, (3, 3), name='conv_l1')
        self.conv_l2 = nn.Conv2d(32, 64, (3, 3), name='conv_l2')
        self.max_pool = nn.MaxPool2d((2, 2), name='max_pool')
        self.l1 = nn.Linear(64 * 12 * 12, 128, name='l1')
        self.l2 = nn.Linear(128, 10, name='l2')

    def forward(self, x):
        z = F.relu(self.conv_l1(x))
        z = F.relu(self.conv_l2(z))
        z = self.max_pool(z)
        z = z.reshape((-1, 64 * 12 * 12))
        z = F.relu(self.l1(z))
        out = F.log_softmax(self.l2(z))
        return out

model = Model()
for p in model.parameters():
    p.data = p.data.to("cuda")
    p.grad = p.grad.to("cuda")

optimizer = ugrad.optim.Adam(model.parameters(), lr=1e-3)

num_batches = -(-X_train.shape[0] // batch_size)
num_batches_t = -(-X_test.shape[0] // batch_size)

for k in range(num_epochs):

    accuracy = 0
    train_loss = 0
    for batch in range(num_batches):
        inputs = ugrad.Tensor(X_train[batch * batch_size:(batch + 1) * batch_size]).to("cuda")
        labels = ugrad.Tensor(y_train[batch * batch_size:(batch + 1) * batch_size]).to("cuda")

        # Forward
        preds = model(inputs)
        loss = F.nll_loss(preds, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Update
        optimizer.step()

        accuracy += int((preds.data.argmax(-1) == labels.data.argmax(-1)).sum().to("cpu").item())
        train_loss += loss.data.to("cpu").item()

    accuracy /= X_train.shape[0]
    train_loss /= X_train.shape[0]
    
    accuracy_t = 0
    loss_t = 0
    with ugrad.no_grad():
        for batch in range(num_batches_t):
            inputs_t = ugrad.Tensor(X_test[batch * batch_size:(batch + 1) * batch_size]).to("cuda")
            labels_t = ugrad.Tensor(y_test[batch * batch_size:(batch + 1) * batch_size]).to("cuda")
            
            preds_t = model(inputs_t)
            loss_t += F.nll_loss(preds_t, labels_t).data.to("cpu").item()
            accuracy_t += int((preds_t.data.argmax(-1) == labels_t.data.argmax(-1)).sum().to("cpu").item())
    
    accuracy_t /= X_test.shape[0]
    loss_t /= X_test.shape[0]

    print(f"Epoch {k+1} loss {train_loss:.6f}, accuracy {accuracy * 100:.6f}%  test loss {loss_t:.6f}, test accuracy {accuracy_t * 100:.6f}% lr {optimizer.lr:.6f}")

