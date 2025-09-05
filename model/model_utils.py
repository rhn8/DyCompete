import torch # For building the networks
from torch import nn
from torch import Tensor
from torchdiffeq import odeint_adjoint as odeint


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RiskWrapper(nn.Module):
  def __init__(self, model, z):
    super().__init__()
    self.model = model
    self.z = z

    #take in t_eval, then multiply by each event time for rescaled time -
    # then just integrate from 0-1


  def forward(self, t, state):
    return self.model(t, state, self.z)
  

class TimeRescale(nn.Module):
        def __init__(self, model, t_eval):
          super().__init__()
          self.t_eval = t_eval
          self.risk_wrapper = model

        def forward(self, t1, state):
           t = t1*self.t_eval

           dH_dt = self.risk_wrapper(t, state)
           return dH_dt*self.t_eval


class CompetingRisk(nn.Module):
    def __init__(self, latent_dim, output_dim, dropout=0):
        super().__init__()
        input_dim = latent_dim + 1 + 1

        self.fc1 = nn.Linear(input_dim, 2*input_dim)
        self.fc2 = nn.Linear(2*input_dim, 1)
        self.softplus = nn.Softplus()



        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        #add softplus here

#want to put in latent space as input- then use constraints of ODE
    def forward(self, t, state, z):
        """"t - Time, h - event type, z - latent space input"""

        state = state[:,0:1]
        if len(t.size()) == 0:
          t = t.expand(z.shape[0],1)
        if state.shape[0] != z.shape[0]:
          state = torch.zeros(z.shape[0],1)
          t = torch.ones(z.shape[0],1).to(device)

        x = torch.cat([t,state, z],dim=1)
        x = self.relu(self.fc1(x))
        x = self.softplus(self.fc2(x))

        return x
    

class CompetingRisks(nn.Module):
    """Computes cause-specific hazard for each competing risk
        using a Neural-ODE system and a black-box ODE solver."""

    def __init__(self,latent_dim, output_dim, risks=1):
        super().__init__()


        self.risks = risks
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.cause_specific = nn.ModuleList([CompetingRisk(latent_dim,
                                                            output_dim)
                                              for risk in range(self.risks)])        

    def forward(self, z, t_eval):
      initial = torch.zeros(z.shape[0],1, device=z.device)

      t_points = Tensor([0., 1.]).to(device)

      total_estimated_CHF = []
      total_estimated_hazard = []

      for risk in self.cause_specific:
        wrapper = RiskWrapper(risk, z)
        rescaled_ode = TimeRescale(wrapper, t_eval)

        #apply odeint adjoint to each cause-specific N-ODE

        estimated_CHF = odeint(rescaled_ode, initial, t_points, method='dopri5', rtol=1e-3,
                               atol=1e-3, adjoint_options=dict(norm="seminorm")).to(device)

        t_reciprocal = torch.where(t_eval != 0, torch.reciprocal(t_eval), 0.0)

        estimated_hazard = t_reciprocal*rescaled_ode(t_points[-1], estimated_CHF[-1]).to(device)

        estimated_CHF = torch.transpose(estimated_CHF, 0 ,1)
        estimated_hazard = torch.transpose(estimated_hazard, 0 ,1)

        total_estimated_CHF.append(estimated_CHF)
        total_estimated_hazard.append(estimated_hazard)

      total_estimated_CHF = torch.stack(total_estimated_CHF,dim=0).to(device)
      total_estimated_hazard = torch.stack(total_estimated_hazard,dim=0).to(device)

      return (total_estimated_hazard, total_estimated_CHF)
    



class Decoder(nn.Module):
    def __init__(self, no_features, output_size):
        super().__init__()
        self.no_features = no_features
        self.hidden_size = no_features
        self.output_size = output_size

        self.fc1 = nn.Linear(self.hidden_size, 3*self.hidden_size)
        self.fc2 = nn.Linear(3*self.hidden_size, 5*self.hidden_size)
        self.fc3 = nn.Linear(5*self.hidden_size, 3*self.hidden_size)
        self.fc4 = nn.Linear(3*self.hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.fc4(x)
        return out
    

class DyCompete(nn.Module):
    def __init__(self, in_features, encoded_features, out_features, risks=1):
        super().__init__()

        self.fc11 = nn.Linear(in_features, 3*in_features)
        self.fc12 = nn.Linear(3*in_features, 5*in_features)
        self.fc13 = nn.Linear(5*in_features, 3*in_features)
        self.fc14 = nn.Linear(3*in_features, encoded_features)
        self.fc24 = nn.Linear(3*in_features, encoded_features)

        self.relu = nn.ReLU()


        self.competing_net = CompetingRisks(encoded_features, out_features, risks)

        self.surv_net = nn.Sequential(
            nn.Linear(encoded_features, 3*in_features), nn.ReLU(),
            nn.Linear(3*in_features, 5*in_features), nn.ReLU(),
            nn.Linear(5*in_features, 3*in_features), nn.ReLU(),
            nn.Linear(3*in_features, out_features),
        )


        self.decoder2 = Decoder(encoded_features, in_features)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        sample_z = eps.mul(std).add_(mu)

        return sample_z

    def encoder(self, x):
        x = self.relu(self.fc11(x))
        x = self.relu(self.fc12(x))
        x = self.relu(self.fc13(x))
        mu_z = self.fc14(x)
        logvar_z = self.fc24(x)

        return mu_z, logvar_z

    def forward(self, input, time):

        t_eval = time.view(-1,1).float()
        mu, logvar = self.encoder(input.float())
        z = self.reparameterize(mu, logvar)

        return self.decoder2(z), self.surv_net(z), mu, logvar, self.competing_net(z, t_eval)

    def predict(self, input):
        mu, logvar = self.encoder(input)
        encoded = self.reparameterize(mu, logvar)
        return self.surv_net(encoded)