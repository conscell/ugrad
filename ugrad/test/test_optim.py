import ugrad
import torch


def test_SGD():
    class Model(ugrad.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = ugrad.nn.Linear(3, 4, name='l1')
            self.l2 = ugrad.nn.Linear(4, 1, name='l2')

        def forward(self, x):
            z = self.l1(x).relu()
            out = self.l2(z).sigmoid()
            return out

    X = ugrad.Tensor([[0., 1., 2.], [10., 20., 30.]])
    y = ugrad.Tensor([[1.], [0.]])

    model = Model()

    class ModelT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(3, 4)
            self.l2 = torch.nn.Linear(4, 1)

        def forward(self, x):
            z = self.l1(x).relu()
            out = self.l2(z).sigmoid()
            return out

    modelT = ModelT().double()

    XT = torch.tensor([[0., 1., 2.], [10., 20., 30.]]).double()
    yT = torch.tensor([[1.], [0.]]).double()

    with torch.no_grad():
        modelT.l1.weight = torch.nn.Parameter(torch.tensor(model.l1.weight.data, dtype=torch.float64))
        modelT.l1.bias = torch.nn.Parameter(torch.tensor(model.l1.bias.data, dtype=torch.float64))

        modelT.l2.weight = torch.nn.Parameter(torch.tensor(model.l2.weight.data, dtype=torch.float64))
        modelT.l2.bias = torch.nn.Parameter(torch.tensor(model.l2.bias.data, dtype=torch.float64))

    optimizer = ugrad.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, dampening=0.9, weight_decay=1e-3)
    optimizerT = torch.optim.SGD(modelT.parameters(), lr=1e-2, momentum=0.9, dampening=0.9, weight_decay=1e-3)

    tol = 1e-6

    for i in range(1000):
        y_hat = model(X)
        y_hatT = modelT(XT)
        loss = ugrad.nn.functional.binary_cross_entropy(y_hat, y)
        lossT = torch.nn.functional.binary_cross_entropy(y_hatT, yT)
        assert abs(loss.data - lossT.data.item()) < tol
        assert optimizer.lr == optimizerT.param_groups[0]['lr']
        loss.backward()
        lossT.backward()
        optimizer.step()
        optimizerT.step()
        optimizer.zero_grad()
        optimizerT.zero_grad()

def test_SGD_scheduler():
    class Model(ugrad.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = ugrad.nn.Linear(3, 4, name='l1')
            self.l2 = ugrad.nn.Linear(4, 1, name='l2')

        def forward(self, x):
            z = self.l1(x).relu()
            out = self.l2(z).sigmoid()
            return out

    X = ugrad.Tensor([[0., 1., 2.], [10., 20., 30.]])
    y = ugrad.Tensor([[1.], [0.]])

    model = Model()

    class ModelT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(3, 4)
            self.l2 = torch.nn.Linear(4, 1)

        def forward(self, x):
            z = self.l1(x).relu()
            out = self.l2(z).sigmoid()
            return out

    modelT = ModelT().double()

    XT = torch.tensor([[0., 1., 2.], [10., 20., 30.]]).double()
    yT = torch.tensor([[1.], [0.]]).double()

    with torch.no_grad():
        modelT.l1.weight = torch.nn.Parameter(torch.tensor(model.l1.weight.data, dtype=torch.float64))
        modelT.l1.bias = torch.nn.Parameter(torch.tensor(model.l1.bias.data, dtype=torch.float64))

        modelT.l2.weight = torch.nn.Parameter(torch.tensor(model.l2.weight.data, dtype=torch.float64))
        modelT.l2.bias = torch.nn.Parameter(torch.tensor(model.l2.bias.data, dtype=torch.float64))

    optimizer = ugrad.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, dampening=0, weight_decay=1e-3, nesterov=True)
    scheduler = ugrad.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=100)

    optimizerT = torch.optim.SGD(modelT.parameters(), lr=1e-2, momentum=0.9, dampening=0, weight_decay=1e-3, nesterov=True)
    schedulerT = torch.optim.lr_scheduler.LinearLR(optimizerT, start_factor=1.0, end_factor=0.5, total_iters=100)

    tol = 1e-6

    for i in range(1000):
        y_hat = model(X)
        y_hatT = modelT(XT)
        loss = ugrad.nn.functional.binary_cross_entropy(y_hat, y)
        lossT = torch.nn.functional.binary_cross_entropy(y_hatT, yT)
        assert abs(loss.data - lossT.data.item()) < tol
        assert abs(optimizer.lr - optimizerT.param_groups[0]['lr']) < tol
        loss.backward()
        lossT.backward()
        optimizer.step()
        optimizerT.step()
        scheduler.step()
        schedulerT.step()
        optimizer.zero_grad()
        optimizerT.zero_grad()