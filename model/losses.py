import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class _Loss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

def _reduction(loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")


def nll_logistic_hazard(phi: Tensor, idx_durations: Tensor, events: Tensor,
                        reduction: str = 'mean', training: bool = True) -> Tensor:
    """
    References:
    [1] Håvard Kvamme and Ørnulf Borgan. Continuous and Discrete-Time Survival Prediction
        with Neural Networks. arXiv preprint arXiv:1910.06724, 2019.
        https://arxiv.org/pdf/1910.06724.pdf

    Inputs:
        phi - network output
        idx_durations - tensor, recording the time at which an event or censoring occured. [time_1, time_2, etc]
        events - tensor, indicating whether the event occured (1) or censoring happened (0). [1, 1, 0, 1, 0, etc]
    Output:
        loss - scalar, reducted tensor, of the BCE loss along the time-axis for each patient
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")

    # Change type of events if necessary
    if events.dtype is torch.bool:
        events = events.float()

    # Change views
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)

    # Creates a target for bce: initialise everything with 0, and setting events at idx_duration
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)

    # Add weighting
    pos_weight = torch.tensor([1]).to(device)

    # Compute BCE
    if training:
        bce = F.binary_cross_entropy_with_logits(phi, y_bce, pos_weight=pos_weight, reduction='none')
    else:
        bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')

    # Compute the loss, along the time axis, ie for each patient separately
    loss = bce.cumsum(1).gather(1, idx_durations).view(-1)

    # Take the mean or something of the loss in the end
    return _reduction(loss, reduction)


def nll_loss(out, events, reduction='mean'):

  """
  References:
  [1] Tang, Weijing, et al. "Soden: A scalable continuous-time survival model
    through ordinary differential equation networks." 
    arXiv preprint arXiv:2008.08637 (2020).
    https://arxiv.org/pdf/2008.08637
    
    Inputs:
        out - CompetingRisks network output
        events - tensor indicating what type of event occured at each given event time, 0 being no event.
    Outputs:
        loss - scalar, gives the NLL loss for each patient over the time axis.
        """


  total_h = out[0][:, :, -1]
  total_CHF = out[1][:,  :, -1]


  epsilon = 1e-8

  num_risks = total_h.shape[0]


  events = events.view(-1, 1)

  loss_per_risk = 0

  for i in range(num_risks):
    current_events = (events == i).int()

    h = total_h[i]
    CHF = total_CHF[i]

    loss_per_risk -= current_events*torch.log(h + epsilon) - CHF

  loss = torch.sum(loss_per_risk)

  return _reduction(loss, reduction)


class NLLLoss(_Loss):
    def forward(self, out, events) -> Tensor:
        return nll_loss(out, events, self.reduction)


class NLLLogistiHazardLoss(_Loss):
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        return nll_logistic_hazard(phi, idx_durations, events, self.reduction, self.training)

def ranking_loss(cif_total, t, e):
  loss = 0
  for k, cifk in enumerate(cif_total):
      for ci, ti in zip(cifk[e-1 == k], t[e-1 == k]):
          if torch.sum(t > ti - 1) > 0:
              loss += torch.mean(torch.sigmoid((cifk[t > ti - 1][torch.arange((t > ti - 1).sum()), ti - 1] - ci[ti - 1])))
              # selects patients that have survived longer than ti and gets their CIF
  return loss


class RankingLoss(_Loss):
    def forward(self, cif_total, t, e) -> Tensor:
        return ranking_loss(cif_total, t, e)

class Loss(nn.Module):
    def __init__(self, alpha: list):
        super().__init__()
        self.alpha = alpha
        self.loss_surv = NLLLogistiHazardLoss()
        self.loss_ae = nn.MSELoss()
        self.loss_competing = NLLLoss()

    def forward(self, decoded, phi, mu, logvar, competing, target_loghaz, target_ae):
        """
            Forward call of the Loss Module. Computes the DySurv model loss by combining three weighted losses.
                1. Survival loss: negative log likelihood logistic hazard or BCE loss over the predictions.
                2. AE loss: reconstruction or MSE loss
                3. KL-divergence: KL-divergence or pushing the model to have a latent space close to a normal distribution.
                4. Competing Risk Loss: negative log likelihood for each cause-specific estimated hazard.
        """
        # Unpack data
        idx_durations, events = target_loghaz

        # Unpack targets
        # Survival Loss
        loss_surv = self.loss_surv(phi, idx_durations, events)

        # AutoEncoder Loss
        loss_ae = self.loss_ae(decoded, target_ae)*2

        # KL-divergence Loss
        loss_kd = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

        #Competing risks Loss
        loss_competing = self.loss_competing(competing, events)/500

        return self.alpha[0] * (loss_surv) + self.alpha[1] * loss_ae + self.alpha[2] * loss_kd + loss_competing * self.alpha[3]