import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Eval():
    def __init__(self, model, test_dataloader: DataLoader, device, frame_count):
        self.model = model
        self.test_dataloader = test_dataloader
        self.device = device
        self.frame_count = frame_count
        self._preds = np.empty((0))
        self._gts = np.empty((0))

    def eval(self):
        size = len(self.test_dataloader.dataset)
        correct = 0

        self.model.eval()
        with torch.no_grad():
            #Iteration test data for evaluation
            for _, (X, y) in enumerate(self.test_dataloader):
                X = X.to(self.device)
                frame_size = X.shape[1]

                y = y.type(torch.LongTensor).to(self.device)
                ground_truth = y

                pred = self.model(X)
                pred_target = pred.argmax(1)

                for i in range(y.shape[1]):
                    correct += (pred_target == ground_truth[:, i]).sum().item()
                    self._gts = np.append(self._gts, ground_truth[:, i].cpu().numpy())

                    self._preds = np.append(self._preds, pred_target.cpu().numpy())


            gts = self._gts
            preds = self._preds

            #all ground truth and prediction
            plt.figure(figsize=(10, 5))
            plt.plot(gts, label='ground truth')
            plt.plot(preds, label='prediction')
            plt.legend()
            plt.show()
            #Accuracy
            acc = (correct / (size * frame_size)) * 100
            print(f"Accuracy completed: {acc}")
            
            tp = (self._gts * self._preds).sum()
            tn = ((1 - self._gts) * (1 - self._preds)).sum()
            fp = ((1 - self._gts) * self._preds).sum()
            fn = (self._gts * (1 - self._preds)).sum()

            epsilon = 1e-7

            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            #F1 score
            f1 = 2 * (precision * recall) / (precision + recall + epsilon)

            print(f"F1 score completed: {f1 * 100}")
            print(f"Precision score  completed: {precision * 100}")
            print(f"Recall score  completed: {recall * 100}")
            print()

        return acc, f1, precision, recall
