import numpy as np


class Schedule:
    def __init__(self, schedule, timesteps):
        self.timesteps = timesteps
        self.schedule = schedule

    def cosine_beta_schedule(self, s=0.001):
        timesteps = self.timesteps
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999).astype(np.float32)

    def linear_beta_schedule(self):
        timesteps = self.timesteps
        scale = 1000 / timesteps
        beta_start = 1e-6 * scale
        beta_end = 0.02 * scale
        return np.linspace(beta_start, beta_end, timesteps).astype(np.float32)

    def quadratic_beta_schedule(self):
        timesteps = self.timesteps
        scale = 1000 / timesteps
        beta_start = 1e-6 * scale
        beta_end = 0.02 * scale
        return (np.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2).astype(np.float32)

    def sigmoid_beta_schedule(self):
        timesteps = self.timesteps
        scale = 1000 / timesteps
        beta_start = 1e-6 * scale
        beta_end = 0.02 * scale
        betas = np.linspace(-6, 6, timesteps)
        return (1.0 / (1.0 + np.exp(-betas)) * (beta_end - beta_start) + beta_start).astype(np.float32)

    def get_betas(self):
        if self.schedule == "linear":
            return self.linear_beta_schedule()
        elif self.schedule == 'cosine':
            return self.cosine_beta_schedule()
        else:
            raise NotImplementedError


if __name__ == "__main__":
    schedule = Schedule(schedule="linear", timesteps=100)
    print(schedule.get_betas().shape)
    print(schedule.get_betas())