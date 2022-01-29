import torch
import dgl


class UsefullNet:
    def __init__(self, original_model, params: dict):
        self.params = params
        self.model = original_model

    def __train(self, g, features, loss_func, train_mask, labels, optmizer):

        prediction = self.model(g, features)
        loss = loss_func(prediction[train_mask], labels[train_mask])

        optmizer.zero_grad()
        loss.backward()
        optmizer.step()

        # Write loss in Tensorboard

    def __validate(self, g=None, features=None, loss_func=None, val_mask=None, labels=None):

        prediction = self.model(g, features)
        val_loss = loss_func(prediction[val_mask], labels[val_mask])
        # Write loss in Tensorboard

    def fit(self, g):
        learning_rate = self.params['lr']
        optmizer = self.params['optim'](self.model.parameters(), lr=learning_rate)
        loss_func = self.params['loss_function']  # this is for regression mean squared loss

        features = g.ndata['x']
        labels = g.ndata['y']

        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']

        for epoch in (self.params['epcohs']):
            self.__train(g, features, loss_func, train_mask, labels, optmizer)

            self.__validate()

    def predict(self, g):
        features = g.ndata['x']
        prediction = self.model(g, features)

        return prediction
