import torch
import torch.nn as nn


def train(data_loader, model, optimizer, device):
    """
    This is main training function that trains the model
    for one epoch
    :param data_loader: this is the torch dataloader
    :param model: model (lstm model)
    :param optimizer: torch optimizer, e.g. adam, sgd etc.
    :param device: this can be "cuda" or "cpu"
    """

    # set model to training mode
    model.train()

    # go through batches of data in data loader
    for data in data_loader:
        # fetch review and target from dictionary
        review = data["review"]
        target = data["target"]

        reviews = review.to(device, dtype=torch.long)
        targets = target.to(device, dtype=torch.float)

        # clear the gradients
        optimizer.zero_grad()

        # make the predictions
        predictions = model(reviews)

        # calculate the loss
        loss = nn.BCEWithLogitsLoss()(predictions, targets.view(-1, 1))

        # compute gradient of loss w.r.t.
        # all parameters of the model that are trainable
        loss.backward()

        # single optimization step
        optimizer.step()


def evaluate(data_loader, model, device):
    # initialize empty lists to store predictions
    # and target's
    final_predictions = []
    final_targets = []

    # put the model in eval mode
    model.eval()

    # disable gradient calculation
    with torch.no_grad():
        for data in data_loader:
            reviews = data["review"]
            targets = data["target"]
            reviews = reviews.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            # make predictions
            predictions = model(reviews)

            # move predictions and targets to the list
            # we also need to move predictions and target to cpu
            predictions = predictions().cpu().numpy().tolist()
            targets = data["target"].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)

    # return final predictions and targets
    return final_predictions, final_targets
