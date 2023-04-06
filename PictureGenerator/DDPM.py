import torch
from tqdm import tqdm

class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, imsize=256, device='cuda'):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.imsize = imsize
        self.device = device

        self.beta = self.prepare_noise_scheduler().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_scheduler(self):
        return torch.linspace(self.beta_start, self.beta_end, self.T)

    def noise_image(self, x, t):
        sqrt_a_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_a_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn(x.shape).to(self.device)
        return sqrt_a_hat * x + sqrt_one_minus_a_hat * e, e

    def sample_timestamp(self, n):
        return torch.randint(low=1, high=self.T, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            xt = torch.randn((n, 3, self.imsize, self.imsize)).to(self.device)

            for i in tqdm(reversed(range(1, self.T)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(xt, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    z = torch.randn_like(xt)
                else:
                    z = torch.zeros_like(xt)

                xt = 1/torch.sqrt(alpha)*(xt - ((1 - alpha)/(torch.sqrt(1 - alpha_hat)))*predicted_noise) + torch.sqrt(beta) * z

        model.train()
        xt = (xt.clamp(-1, 1) + 1)/2
        xt = (xt * 255).type(torch.uint8)
        return xt