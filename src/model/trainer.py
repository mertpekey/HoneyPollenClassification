import torch
import torch.nn as nn

import model.config as config
import utils.other_utils as other_utils


class Trainer():

  def __init__(self, model, criterion = None, optimizer = None, device = "cpu"):
    
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.device = device
    

  def train_step(self, train_dataloader):
    
    # Set model to training mode
    self.model.train()

    epoch_loss, epoch_accuracy = 0, 0

    for batch, (X,y) in enumerate(train_dataloader):
      X = X.to(self.device)
      y = y.to(self.device)

      # Reset Gradients
      self.optimizer.zero_grad()

      # Prediction
      out = self.model(X)

      # Calculation Loss
      loss = self.criterion(out, y)

      # Calculating Gradients
      loss.backward()

      # Update Weights
      self.optimizer.step()

      # Calculating Performance Metrics
      epoch_loss += loss.detach().item() / X.shape[0]
      epoch_accuracy += (torch.argmax(out, dim=1) == y).sum() / X.shape[0]

    return epoch_loss, epoch_accuracy


  def eval_step(self, val_dataloader):
    
    # Set model to training mode
    self.model.eval()

    epoch_loss, epoch_accuracy = 0, 0
    y_trues, y_probs = [], []
    #epoch_accuracy = 0
    with torch.inference_mode():
      for batch, (X,y) in enumerate(val_dataloader):
        X = X.to(self.device)
        y = y.to(self.device)

        # Prediction
        out = self.model(X)
        # Calculation Loss
        loss = self.criterion(out, y)

        # Calculating Performance Metrics
        epoch_loss += loss.item() / X.shape[0]
        epoch_accuracy += (torch.argmax(out, dim=1) == y).sum() / X.shape[0]

    return epoch_loss, epoch_accuracy

  def predict_step(self, val_dataloader):

    self.model.eval()

    y_preds = []
    y_probs = []

    with torch.inference_mode():
      for batch, (X,y) in enumerate(val_dataloader):
        X = X.to(self.device)
        y = y.to(self.device)

        # Prediction
        out = self.model(X)

        # Calculating Performance Metrics
        y_prob = torch.softmax(out[:,0,:], dim=1)
        y_pred = torch.argmax(torch.softmax(out[:,0,:], dim=1), dim=1)

        y_probs = [y_prob[index, y_pred[index]].item() for index in range(len(y_pred))]
        y_preds.extend(y_pred)

    return y_probs, y_preds


  def train(self, train_dataloader, val_dataloader, num_epochs = 5):

    if config.LOAD_MODEL:
        other_utils.load_checkpoint(
            config.CHECKPOINT_MODEL,
            self.model,
            self.optimizer,
            config.LEARNING_RATE,
        )

    for epoch in range(num_epochs):

      train_loss, train_accuracy = self.train_step(train_dataloader)
      val_loss, val_accuracy = self.eval_step(val_dataloader)

      if config.SAVE_MODEL:
        other_utils.save_checkpoint(self.model, self.optimizer, filename=config.CHECKPOINT_MODEL)

      # Logging
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.5f}, "
          f"train_acc: {train_accuracy:.5f}, "
          f"val_loss: {val_loss:.5f}, "
          f"val_acc: {val_accuracy:.5f}"
      )