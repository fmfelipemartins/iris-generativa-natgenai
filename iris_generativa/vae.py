"""VAE condicional para demonstração e EDA; estado reiniciado por execução/fold."""

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import config
from .progress import progress


class ConditionalVAE(nn.Module):
    """Encoder e decoder com uma camada oculta e condição one-hot."""

    def __init__(self, features, tasks, latent, hidden):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(features + tasks, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent + tasks, hidden), nn.ReLU(), nn.Linear(hidden, features)
        )

    def forward(self, x, condition, generator):
        encoded = self.encoder(torch.cat((x, condition), dim=1))
        mu, logvar = self.mu(encoded), self.logvar(encoded).clamp(-12, 12)
        noise = torch.randn(mu.shape, generator=generator)
        latent = mu + torch.exp(0.5 * logvar) * noise
        return self.decoder(torch.cat((latent, condition), dim=1)), mu, logvar


class VAEGenerator:
    """CPU determinística; gerador Torch privado para lotes, latentes e amostragem."""

    def __init__(
        self,
        features,
        tasks,
        seed,
        latent=config.VAE_LATENT_DIM,
        hidden=config.VAE_HIDDEN_DIM,
        learning_rate=config.VAE_LEARNING_RATE,
        beta=config.VAE_BETA,
        batch_size=config.VAE_BATCH_SIZE,
        kl_warmup_steps=0,  # Atualizações do otimizador até alcançar beta integral.
    ):
        if (
            min(features, len(tasks), latent, hidden, batch_size) < 1
            or beta < 0
            or learning_rate <= 0
        ):
            raise ValueError("Hiperparâmetros inválidos para VAE.")
        if kl_warmup_steps < 0:
            raise ValueError("Warmup deve ser não negativo.")
        self.kl_warmup_steps = kl_warmup_steps  # Contagem contínua entre gerações.
        self.optimizer_steps = 0  # Orçamento efetivo de treinamento.
        self.tasks, self.latent = list(tasks), latent
        self.beta, self.batch_size = beta, batch_size
        # A inicialização não altera o RNG global do processo chamador.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = ConditionalVAE(features, len(tasks), latent, hidden)
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.training_history = []

    def fit(
        self, populations, epochs=config.VAE_EPOCHS, show_progress=False, generation=0
    ):
        if epochs < 1:
            raise ValueError("Treinamento requer ao menos uma época.")
        points = np.vstack([populations[t] for t in self.tasks]).astype(np.float32)
        if not np.isfinite(points).all():
            raise ValueError("VAE recebeu valores não finitos.")
        labels = np.concatenate(
            [np.full(len(populations[t]), i) for i, t in enumerate(self.tasks)]
        )
        x = torch.from_numpy(points)
        condition = torch.eye(len(self.tasks))[torch.from_numpy(labels)]
        self.model.train()
        bar = progress(range(epochs), "VAE: épocas", show_progress, leave=False)
        for epoch in bar:
            order = torch.randperm(len(x), generator=self.rng)
            totals = np.zeros(3)
            for indices in order.split(self.batch_size):
                reconstructed, mu, logvar = self.model(
                    x[indices], condition[indices], self.rng
                )
                reconstruction = ((reconstructed - x[indices]) ** 2).sum(dim=1).mean()
                kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
                effective_beta = self.beta * min(
                    1.0, (self.optimizer_steps + 1) / max(1, self.kl_warmup_steps)
                )  # Rampa linear opcional.
                loss = (
                    reconstruction + effective_beta * kl
                )  # Objetivo desta atualização.
                if not torch.isfinite(loss):
                    raise FloatingPointError("Perda não finita no VAE.")
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optimizer.step()
                self.optimizer_steps += (
                    1  # Mede custo sem depender do número de épocas.
                )
                totals += len(indices) * np.array(
                    [loss.item(), reconstruction.item(), kl.item()]
                )
            totals /= len(x)
            row = dict(
                generation=generation,
                epoch=epoch + 1,
                loss=totals[0],
                reconstruction=totals[1],
                kl=totals[2],
                effective_beta=effective_beta,  # Peso ao fim da época.
                optimizer_steps=self.optimizer_steps,  # Atualizações acumuladas do modelo.
            )
            self.training_history.append(row)
            bar.set_postfix(loss=f"{totals[0]:.3f}")
        return self

    def sample(self, task, size, rng=None):
        if task not in self.tasks or size < 0:
            raise ValueError("Tarefa ou tamanho inválido.")
        self.model.eval()
        with torch.no_grad():
            latent = torch.randn((size, self.latent), generator=self.rng)
            condition = torch.eye(len(self.tasks))[self.tasks.index(task)].repeat(
                size, 1
            )
            points = (
                self.model.decoder(torch.cat((latent, condition), dim=1))
                .numpy()
                .astype(float)
            )
        return points, [(task,)] * size

    def history(self):
        return pd.DataFrame(self.training_history)

    def diagnose(self, populations):
        """Sonda determinística do encoder/decoder; não modifica RNG de treinamento."""
        probe = torch.Generator(device="cpu").manual_seed(config.VAE_DIAGNOSTIC_SEED)
        latent = torch.randn(
            (config.VAE_DIAGNOSTIC_SAMPLES, self.latent), generator=probe
        )
        rows = []
        with torch.no_grad():
            decoded = []
            for index, task in enumerate(self.tasks):
                condition = torch.eye(len(self.tasks))[index].repeat(len(latent), 1)
                decoded.append(
                    self.model.decoder(torch.cat((latent, condition), dim=1))
                )
            for index, task in enumerate(self.tasks):
                x = torch.tensor(np.asarray(populations[task]), dtype=torch.float32)
                condition = torch.eye(len(self.tasks))[index].repeat(len(x), 1)
                encoded = self.model.encoder(torch.cat((x, condition), dim=1))
                mu = self.model.mu(encoded)
                logvar = self.model.logvar(encoded).clamp(-12, 12)
                kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
                active = (
                    mu.var(0, unbiased=False) > config.VAE_ACTIVE_UNIT_THRESHOLD
                ).sum()
                alternatives = [
                    ((decoded[index] - d) ** 2).sum(1).mean().item()
                    for i, d in enumerate(decoded)
                    if i != index
                ]
                rows.append(
                    dict(
                        task=task,
                        encoder_kl=float(kl),
                        active_units=int(active),
                        decoder_latent_variance=float(
                            decoded[index].var(0, unbiased=False).sum()
                        ),
                        decoder_condition_effect=float(np.mean(alternatives))
                        if alternatives
                        else 0.0,
                    )
                )
        return rows
