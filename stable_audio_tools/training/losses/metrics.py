import torch
import torchaudio

from torch.nn import functional as F
from torch import nn
from typing import Tuple

### Metrics are loss-like functions that do not backpropagate gradients.

class PESQMetric(nn.Module):
    def __init__(self, sample_rate: int):
        super().__init__()
        self.resampler = (
            torchaudio.transforms.Resample(sample_rate, 16000)
            if sample_rate != 16000 else None)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        from pypesq import pesq
        
        if self.resampler is not None:
            inputs = self.resampler(inputs)
            targets = self.resampler(targets)

        inputs_np = inputs.cpu().numpy().astype("float64")
        targets_np = targets.cpu().numpy().astype("float64")
        batch_size = targets.shape[0]

        # Compute average pesq across batch size.
        val_pesq = (1.0 / batch_size) * sum(
            pesq(targets_np[i].reshape(-1), inputs_np[i].reshape(-1), 16000)
            for i in range(batch_size))
        return val_pesq
    

class LogSpectralDistance(nn.Module):
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        input_stft = torch.stft(inputs, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=torch.hann_window(self.n_fft).to(inputs.device),
                                return_complex=True)
        target_stft = torch.stft(targets, n_fft=self.n_fft, hop_length=self.hop_length,
                                 window=torch.hann_window(self.n_fft).to(targets.device),
                                 return_complex=True)

        input_mag = torch.abs(input_stft)
        target_mag = torch.abs(target_stft)

        diff = 10 * (torch.log10(input_mag + 1e-8) - torch.log10(target_mag + 1e-8))
        lsd = torch.sqrt(torch.mean(diff ** 2, dim=(1, 2)))  # Mean over F and T
        return torch.mean(lsd)  # Average over batch

class LTASDistance(nn.Module):
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        input_stft = torch.stft(inputs, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=torch.hann_window(self.n_fft).to(inputs.device),
                                return_complex=True)
        target_stft = torch.stft(targets, n_fft=self.n_fft, hop_length=self.hop_length,
                                 window=torch.hann_window(self.n_fft).to(targets.device),
                                 return_complex=True)

        input_mag = torch.abs(input_stft)
        target_mag = torch.abs(target_stft)

        input_ltas = torch.mean(input_mag, dim=2)  # Mean over time
        target_ltas = torch.mean(target_mag, dim=2)

        ltas_dist = torch.mean(torch.abs(input_ltas - target_ltas) / (target_ltas + 1e-8), dim=1)
        return torch.mean(10 * torch.log10(ltas_dist + 1e-8))  # Average over batch

class SISDRMetric(nn.Module):
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets - torch.mean(targets, dim=-1, keepdim=True)
        inputs = inputs - torch.mean(inputs, dim=-1, keepdim=True)

        alpha = torch.sum(inputs * targets, dim=-1, keepdim=True) / (
            torch.sum(targets ** 2, dim=-1, keepdim=True) + 1e-8)
        s_target = alpha * targets
        e_noise = inputs - s_target

        sisdr = 10 * torch.log10(torch.sum(s_target ** 2, dim=-1) / (torch.sum(e_noise ** 2, dim=-1) + 1e-8))
        return torch.mean(sisdr)

class SNRMetric(nn.Module):
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        noise = inputs - targets
        signal_power = torch.sum(targets ** 2, dim=-1)
        noise_power = torch.sum(noise ** 2, dim=-1)
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        return torch.mean(snr)

class STFTDistance(nn.Module):
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        input_stft = torch.stft(inputs, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=torch.hann_window(self.n_fft).to(inputs.device),
                                return_complex=True)
        target_stft = torch.stft(targets, n_fft=self.n_fft, hop_length=self.hop_length,
                                 window=torch.hann_window(self.n_fft).to(targets.device),
                                 return_complex=True)

        dist = torch.abs(input_stft - target_stft)
        return torch.mean(torch.sqrt(torch.sum(dist ** 2, dim=(1, 2))))  # L2 norm then mean

class MelDistance(nn.Module):
    def __init__(self, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 80):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        input_mel = self.mel_transform(inputs)
        target_mel = self.mel_transform(targets)

        dist = torch.abs(input_mel - target_mel)
        return torch.mean(torch.sqrt(torch.sum(dist ** 2, dim=(1, 2))))