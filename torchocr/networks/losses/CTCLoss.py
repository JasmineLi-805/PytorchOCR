from torch import nn
import torch


class CTCLoss(nn.Module):

    def __init__(self, blank_idx, reduction='mean'):
        super().__init__()
        self.loss_func = torch.nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)

    def forward(self, pred, args):
        batch_size = pred.size(0)
        # print(f'batch size when calc loss: {batch_size}')
        label, label_length = args['targets'], args['targets_lengths']
        # print(f'labels = {label}')
        pred = pred.log_softmax(2)
        pred = pred.permute(1, 0, 2)
        # print(f'prediction = {pred}')
        preds_lengths = torch.tensor([pred.size(0)] * batch_size, dtype=torch.long)
        # print(f'pred_len = {preds_lengths}, label_len = {label_length}')
        loss = self.loss_func(pred, label, preds_lengths, label_length)
        # print(loss)
        return {'loss': loss}