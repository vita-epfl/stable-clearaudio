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
        # Compute STFT
        input_stft = torch.stft(inputs, n_fft=self.n_fft, hop_length=self.hop_length, 
                              window=torch.hann_window(self.n_fft).to(inputs.device),
                              return_complex=True)
        target_stft = torch.stft(targets, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=torch.hann_window(self.n_fft).to(targets.device),
                                return_complex=True)
        
        # Compute magnitude spectrograms
        input_mag = torch.abs(input_stft)
        target_mag = torch.abs(target_stft)
        
        # Compute log spectral distance
        lsd = torch.mean(torch.sqrt(torch.mean((torch.log10(input_mag + 1e-8) - 
                                              torch.log10(target_mag + 1e-8))**2, dim=1)))
        return lsd

class LongTermAverageSpectrum(nn.Module):
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute STFT
        input_stft = torch.stft(inputs, n_fft=self.n_fft, hop_length=self.hop_length,
                              window=torch.hann_window(self.n_fft).to(inputs.device),
                              return_complex=True)
        target_stft = torch.stft(targets, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=torch.hann_window(self.n_fft).to(targets.device),
                                return_complex=True)
        
        # Compute magnitude spectrograms
        input_mag = torch.abs(input_stft)
        target_mag = torch.abs(target_stft)
        
        # Compute average spectrum
        input_ltas = torch.mean(input_mag, dim=2)
        target_ltas = torch.mean(target_mag, dim=2)
        
        return input_ltas, target_ltas

class SISDRMetric(nn.Module):
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Scale-invariant signal to distortion ratio
        alpha = (torch.sum(inputs * targets, dim=-1, keepdim=True) / 
                (torch.sum(targets * targets, dim=-1, keepdim=True) + 1e-8))
        e_target = alpha * targets
        e_res = inputs - e_target
        
        sisdr = 10 * torch.log10(torch.sum(e_target * e_target, dim=-1) / 
                                (torch.sum(e_res * e_res, dim=-1) + 1e-8))
        return torch.mean(sisdr)

class SNRMetric(nn.Module):
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Signal to noise ratio
        noise = inputs - targets
        snr = 10 * torch.log10(torch.sum(targets * targets, dim=-1) / 
                              (torch.sum(noise * noise, dim=-1) + 1e-8))
        return torch.mean(snr)

class STFTDistance(nn.Module):
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute STFT
        input_stft = torch.stft(inputs, n_fft=self.n_fft, hop_length=self.hop_length,
                              window=torch.hann_window(self.n_fft).to(inputs.device),
                              return_complex=True)
        target_stft = torch.stft(targets, n_fft=self.n_fft, hop_length=self.hop_length,
                                window=torch.hann_window(self.n_fft).to(targets.device),
                                return_complex=True)
        
        # Compute distance between complex spectrograms
        stft_dist = torch.mean(torch.abs(input_stft - target_stft))
        return stft_dist

class MelDistance(nn.Module):
    def __init__(self, sample_rate: int, n_fft: int = 2048, hop_length: int = 512, 
                 n_mels: int = 80):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute mel spectrograms
        input_mel = self.mel_transform(inputs)
        target_mel = self.mel_transform(targets)
        
        # Compute distance between mel spectrograms
        mel_dist = torch.mean(torch.abs(input_mel - target_mel))
        return mel_dist