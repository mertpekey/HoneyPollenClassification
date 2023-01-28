import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import model.config as config
import utils.other_utils as other_utils
import utils.eval_utils as eval_utils


class Trainer():

  def __init__(self, model, criterion = None, optimizer = None, lr_scheduler = None, device = "cpu", model_name = None, experiment_name = None):
    
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.lr_scheduler = lr_scheduler
    self.device = device
    self.model_name = model_name
    self.experiment_name = experiment_name

    self.writer = other_utils.create_writer(self.experiment_name, self.model_name, None)
    

  def train_step(self, train_dataloader):
    
    # Set model to training mode
    self.model.train()

    epoch_loss, epoch_accuracy = 0, 0
    topk_acc_list = 0
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
      if self.lr_scheduler is not None:
        self.lr_scheduler.step()

      # Calculating Performance Metrics
      epoch_loss += loss.item()
      epoch_accuracy += eval_utils.accuracy_fn(y_true=y, y_pred=out.argmax(dim=1))

    epoch_loss /= len(train_dataloader)
    epoch_accuracy /= len(train_dataloader)

    return epoch_loss, epoch_accuracy


  def eval_step(self, val_dataloader):
    
    # Set model to training mode
    self.model.eval()

    epoch_loss, epoch_accuracy = 0, 0
    topk_acc_list = [0, 0, 0]
    y_trues, y_probs = [], []

    with torch.inference_mode():
      for batch, (X,y) in enumerate(val_dataloader):
        X = X.to(self.device)
        y = y.to(self.device)

        # Prediction
        out = self.model(X)
        # Calculation Loss
        loss = self.criterion(out, y)

        # Calculating Performance Metrics
        epoch_loss += loss.item()
        epoch_accuracy += eval_utils.accuracy_fn(y_true=y, y_pred=out.argmax(dim=1))

        topk_list = eval_utils.top_accuracy(out, y, topk=(1,3,5))
        topk_acc_list[0] += topk_list[0]
        topk_acc_list[1] += topk_list[1]
        topk_acc_list[2] += topk_list[2]

      topk_acc_list[0] /= len(val_dataloader)
      topk_acc_list[1] /= len(val_dataloader)
      topk_acc_list[2] /= len(val_dataloader)
      epoch_loss /= len(val_dataloader)
      epoch_accuracy /= len(val_dataloader)

    return epoch_loss, epoch_accuracy, topk_acc_list


  def predict_step(self, val_dataloader):

    self.model.eval()

    y_preds = []
    y_probs = []
    y_s = []

    with torch.inference_mode():
      for batch, (X,y) in enumerate(val_dataloader):
        X = X.to(self.device)
        y = y.to(self.device)

        # Prediction
        out = self.model(X)

        # Calculating Performance Metrics
        y_prob = torch.softmax(out, dim=1)
        y_pred = out.argmax(dim=1)

        y_probs = [y_prob[index, y_pred[index]].item() for index in range(len(y_pred))]
        y_preds.extend(y_pred)
        y_s.extend(y)

    return y_probs, y_preds, y_s


  def train(self, train_dataloader, val_dataloader, num_epochs = 5, patience = 5):

    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
    }

    if config.LOAD_MODEL:
        other_utils.load_checkpoint(
            config.CHECKPOINT_MODEL,
            self.model,
            self.optimizer,
            config.LEARNING_RATE,
        )

    best_val_loss = np.inf

    for epoch in tqdm(range(num_epochs)):

      train_loss, train_accuracy = self.train_step(train_dataloader)
      val_loss, val_accuracy, topk_list_val = self.eval_step(val_dataloader)

      # Early stopping
      if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_model = self.model
          _patience = patience
      else:
          _patience -= 1
      if not _patience:
          print("Stopping early!")
          break


      if config.SAVE_MODEL:
        other_utils.save_checkpoint(self.model, self.optimizer, filename=config.CHECKPOINT_MODEL)

      # Logging
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.5f}, "
          f"train_acc: {train_accuracy:.5f}, "
          f"val_loss: {val_loss:.5f}, "
          f"val_acc: {val_accuracy:.5f}, "
          f"top1_val acc: {topk_list_val[0]:.5f}, "
          f"top3_val acc: {topk_list_val[1]:.5f}, "
          f"top5_val acc: {topk_list_val[2]:.5f}"
      )

      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_accuracy)
      results["val_loss"].append(val_loss)
      results["val_acc"].append(val_accuracy)

      # Add loss results to SummaryWriter
      self.writer.add_scalars(main_tag="Loss", 
                          tag_scalar_dict={"train_loss": train_loss,
                                          "val_loss": val_loss},
                          global_step=epoch)

      # Add accuracy results to SummaryWriter
      self.writer.add_scalars(main_tag="Accuracy", 
                          tag_scalar_dict={"train_acc": train_accuracy,
                                          "val_acc": val_accuracy}, 
                          global_step=epoch)
      
      # Track the PyTorch model architecture
      self.writer.add_graph(model=self.model, 
                        input_to_model=torch.randn(config.BATCH_SIZE, 3, config.IMG_SIZE[0], config.IMG_SIZE[1]).to(config.DEVICE))
    
      # Close the writer
    self.writer.close()

    # Assign best model to the last model
    return best_model