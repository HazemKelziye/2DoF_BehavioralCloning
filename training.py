import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from parsedata import pipeline
from cnn import CNN2D_2headed
from torchmetrics import Precision, F1Score, Recall
import tqdm
import copy


os.environ['WANDB_API_KEY'] = None
wandb.login(relogin=True)

# setting the computation device and the random seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
np.random.seed(42)

ACTIONS_LABEL_ENCODING = { -1.0: 0, -0.6: 1, -0.2: 2, 0.2: 3, 0.6: 4, 1.0: 5}

X_train, X_test, y1_train, y1_test, y2_train, y2_test = pipeline("/home/vboxuser/PycharmProjects/episodes_v4_1.json")
run = wandb.init(project="BC-MultiHeaded-CNN")

model = CNN2D_2headed().float().to(device)

# hyperparameter selection
learning_rate = 0.003
n_epochs = 50
batch_size = 2048

hyperparameters = {
    "architecture": "MultiHeaded-CNN-11",
    "dataset": "episodes_v4_1.json",
    "learning_rate": learning_rate,
    "epochs": n_epochs,
    "batch_size": batch_size,
    "loss": "Cross-entropy",
    "optimizer": "Adam",
}

wandb.config.update(hyperparameters)

loss_fn = nn.CrossEntropyLoss()  # selecting the ce loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

precision_metric = Precision(num_classes=6, average='weighted', task='multiclass').to(device)
recall_metric = Recall(num_classes=6, average='weighted', task='multiclass').to(device)
f1_metric = F1Score(num_classes=6, average='weighted', task='multiclass').to(device)


# ensure at least one batch if dataset size is smaller than batch size
batches_per_epoch = max(len(X_train) // batch_size, 1)


best_acc = - np.inf   # init to negative infinity for best accuracy
best_weights = None
train_loss_hist = []
train_acc_hist = []
test_loss_hist = []
test_acc_hist = []

for epoch in range(n_epochs):
    epoch_loss1 = []
    epoch_loss2 = []
    epoch_loss_total = []
    epoch_acc1 = []
    epoch_acc2 = []

    model.train()
    with tqdm.trange(batches_per_epoch, unit="batch", mininterval=0) as bar:
        bar.set_description(f"Epoch {epoch}")
        for i in bar:
            # take a batch
            start = i * batch_size
            end = start + batch_size
            # Ensure not to go out of bounds
            end = min(end, len(X_train))
            X_batch = X_train[start:end].to(device)
            y1_batch = y1_train[start:end].long().to(device)
            y2_batch = y2_train[start:end].long().to(device)
            y1_pred, y2_pred = model(X_batch)
            loss1 = loss_fn(y1_pred, y1_batch)
            loss2 = loss_fn(y2_pred, y2_batch)
            optimizer.zero_grad()
            loss = loss1 + loss2  # adding the losses up
            loss.backward()
            optimizer.step()
            acc1 = (torch.argmax(y1_pred, 1) == y1_batch).float().mean()
            acc2 = (torch.argmax(y2_pred, 1) == y2_batch).float().mean()
            # log metrics to wandb
            wandb.log({"train/accuracy1": acc1,
                       "train/accuracy2": acc2,
                       "train/loss1": loss1,
                       "train/loss2": loss2,
                       "train/total_loss": loss,
                       })
            epoch_loss_total.append(loss.item())
            epoch_loss1.append(loss1.item())
            epoch_loss2.append(loss2.item())
            epoch_acc1.append(acc1.item())
            epoch_acc2.append(acc2.item())
            bar.set_postfix(loss=loss.item(), acc1=acc1.item(), acc2=acc2.item())

    # set model in evaluation mode and run through the test set
    model.eval()
    y1_test = y1_test.long().to(device)
    y2_test = y2_test.long().to(device)
    y1_pred, y2_pred = model(X_test.to(device))
    ce1 = loss_fn(y1_pred, y1_test)
    ce2 = loss_fn(y2_pred, y2_test)
    ce_total = ce1 + ce2
    acc1 = (torch.argmax(y1_pred, 1) == y1_test).float().mean()
    acc2 = (torch.argmax(y2_pred, 1) == y2_test).float().mean()
    acc_average = (acc1 + acc2)/2
    predictions1 = torch.argmax(y1_pred, 1)
    predictions2 = torch.argmax(y2_pred, 1)
    precision1 = precision_metric(predictions1, y1_test)
    precision2 = precision_metric(predictions2, y2_test)
    f1_1 = f1_metric(predictions1, y1_test)
    f1_2 = f1_metric(predictions2, y2_test)
    recall1 = recall_metric(predictions1, y1_test)
    recall2 = recall_metric(predictions2, y2_test)
    if acc_average > best_acc:
        best_acc = acc_average
        best_weights = copy.deepcopy(model.state_dict())

    # log metrics to wandb
    wandb.log({"test/accuracy1": acc1,
               "test/accuracy2": acc2,
               "test/loss1": ce1,
               "test/loss2": ce2,
               "test/total_loss": ce_total,
               "precision1": precision1,
               "recall1": recall1,
               "f1-score1": f1_1,
               "precision2": precision2,
               "recall2": recall2,
               "f1-score2": f1_2,
                })
    print(f"Epoch {epoch} validation: Cross-entropy1={ce1}, Cross-entropy2={ce2},\n Accuracy1={acc1}, Accuracy2={acc2}, Accuracy_avg={acc_average}")

model.load_state_dict(best_weights)

# save the trained model
model.eval()
# Assuming 'model' is your neural network model
torch.save(model.state_dict(), "multi_headed_model_5.pth")

# create artifacts to save the model.pth file
artifact = wandb.Artifact("multi_headed_model_5", type="model")
artifact.add_file("multi_headed_model_5.pth")
run.log_artifact(artifact)

