import os
os.environ['UGRAD_COMPUTE'] = 'cupy'
import cupy as np

import ugrad
import ugrad.nn as nn
import ugrad.nn.functional as F

def main():
    seed = 1717
    np.random.seed(seed)
    num_epochs = 50
    batch_size = 128

    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff')
    y = y.astype(np.int64)
    # Scale images to the [0, 1] range
    X /= 255.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    print(f"{X_train.shape=} {X_test.shape=} {y_train.shape=} {y_test.shape=}")

    inputs_t = ugrad.Tensor(X_test)
    labels_t = ugrad.Tensor(np.eye(10)[y_test])
    print(f"{labels_t.data[0]=}")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(784, 1568, name='l1')
            self.l2 = nn.Linear(1568, 392, name='l2')
            self.l3 = nn.Linear(392, 10, name='l3')

        def forward(self, x):
            z = F.relu(self.l1(x))
            z = F.dropout(z, p=0.5, training=self.training)
            z = F.relu(self.l2(z))
            z = F.dropout(z, p=0.5, training=self.training)
            out = F.log_softmax(self.l3(z))
            return out

    model = Model()
    optimizer = ugrad.optim.SGD(model.parameters(), lr=3e-2, momentum=0.9, nesterov=True)
    scheduler = ugrad.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.05, total_iters=num_epochs)

    num_batches = -(-X_train.shape[0] // batch_size)

    for k in range(num_epochs):

        accuracy = 0
        train_loss = 0
        model.train()
        for batch in range(num_batches):
            inputs = ugrad.Tensor(X_train[batch * batch_size:(batch + 1) * batch_size])
            labels = ugrad.Tensor(np.eye(10)[y_train[batch * batch_size:(batch + 1) * batch_size]])

            # Forward
            preds = model(inputs)
            loss = F.nll_loss(preds, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Update (SGD)
            optimizer.step()

            accuracy += int(np.count_nonzero(np.argmax(preds.data, axis=-1) == np.argmax(labels.data, axis=-1)))
            train_loss += loss.data.item()

        scheduler.step()
        accuracy /= X_train.shape[0]
        train_loss /= X_train.shape[0]
        
        model.eval()
        with ugrad.no_grad():
            preds_t = model(inputs_t)
            loss_t = F.nll_loss(preds_t, labels_t).data.item()

        accuracy_t = int(np.count_nonzero(np.argmax(preds_t.data, axis=-1) == np.argmax(labels_t.data, axis=-1))) / X_test.shape[0]

        print(f"Epoch {k+1} loss {train_loss:.6f}, accuracy {accuracy * 100:.6f}%  test loss {loss_t:.6f}, test accuracy {accuracy_t * 100:.6f}% lr {optimizer.lr:.6f}")


if __name__ == "__main__":
    main()
