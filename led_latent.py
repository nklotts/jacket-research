"""
Pretrained VAE-based LED pattern codec using Stable Diffusion VAE.

Instead of having SAC predict all N_LEDS*3 values directly, the agent
operates in a compressed latent space:

  encode:  action (N_LEDS*3,) → latent (LATENT_DIM,)   [not used during training]
  decode:  latent (LATENT_DIM,) → action (N_LEDS*3,)   [used every step]

Architecture:
  LED array: 16×16×38 = 9728 LEDs × 3 channels = 29 184 values
  Reshape  → image (3 × 96 × 104) = 29 952 values  (padded with zeros)
  SD VAE   → latent (4 × 12 × 13) = 624 values   ← SAC operates here

Compression ratio: 29 184 → 624  (~47×)

The SD VAE is loaded from HuggingFace (stabilityai/sd-vae-ft-mse) and
kept fully frozen — we only use its decoder during training.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants (kept here so they can be imported without instantiating the class)
# ---------------------------------------------------------------------------
N_LEDS     = 16 * 16 * 38       # 9 728
IMG_H      = 96                  # divisible by 8 → VAE requirement
IMG_W      = 104                 # divisible by 8 → VAE requirement
IMG_PIXELS = IMG_H * IMG_W
LATENT_SCALE_FACTOR = 8
LATENT_H   = IMG_H // LATENT_SCALE_FACTOR          # 12
LATENT_W   = IMG_W // LATENT_SCALE_FACTOR          # 13
LATENT_DIM = 4 * LATENT_H * LATENT_W   # 624

# Scale SAC's tanh output [-1, 1] → VAE latent range (empirically ~[-3, 3])
LATENT_SCALE = 3.0


class LEDPatternCodec:
    """
    Wraps a pretrained SD VAE to encode/decode LED patterns.

    Usage in training loop
    ----------------------
    codec   = LEDPatternCodec(device)
    agent   = SACAgent(state_dim, action_dim=LATENT_DIM, ...)

    latent  = agent.select_action(state)          # (LATENT_DIM,)  in [-1,1]
    pattern = codec.decode(latent)                # (N_LEDS*3,)    in [0,1]
    reward, done, info = env.step(pattern, ...)   # send real pattern to RPi
    agent.replay_buffer.push(state, latent, ...)  # store latent, not pattern
    """

    VAE_ID = 'stabilityai/sd-vae-ft-mse'

    def __init__(self, device: torch.device, latent_scale: float = LATENT_SCALE):
        self.device       = device
        self.latent_scale = latent_scale
        self.latent_dim   = LATENT_DIM

        print(f"[LEDPatternCodec] Loading pretrained VAE: {self.VAE_ID}")
        print( "[LEDPatternCodec] (first run downloads ~335 MB from HuggingFace)")
        try:
            from diffusers import AutoencoderKL
        except ImportError as e:
            raise ImportError(
                "diffusers is required: pip install diffusers"
            ) from e

        self.vae = AutoencoderKL.from_pretrained(self.VAE_ID).to(device)
        # Freeze all parameters — we never backprop through the VAE
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

        print(f"[LEDPatternCodec] VAE ready.")
        print(f"[LEDPatternCodec] LED pattern {N_LEDS}×3={N_LEDS*3} → "
              f"image {IMG_H}×{IMG_W}×3 → latent 4×{LATENT_H}×{LATENT_W} = {LATENT_DIM}")
        print(f"[LEDPatternCodec] Compression: {N_LEDS*3} → {LATENT_DIM}  "
              f"({N_LEDS*3/LATENT_DIM:.1f}× reduction)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _action_to_tensor(self, action: np.ndarray) -> torch.Tensor:
        """
        action : (N_LEDS*3,) float32 in [0, 1]
        returns: (1, 3, IMG_H, IMG_W)  float32 in [-1, 1]   (on self.device)
        """
        pad = IMG_PIXELS * 3 - N_LEDS * 3        # zero-pad tail
        padded = np.concatenate([action, np.zeros(pad, dtype=np.float32)])
        img = padded.reshape(IMG_H, IMG_W, 3)     # (H, W, 3)
        t   = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)
        t   = t * 2.0 - 1.0                       # [0,1] → [-1,1]
        return t.to(self.device)

    def _tensor_to_action(self, recon: torch.Tensor) -> np.ndarray:
        """
        recon  : (1, 3, IMG_H, IMG_W)  in [-1, 1]
        returns: (N_LEDS*3,) float32   in [0, 1]
        """
        img  = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()   # (H,W,3)
        flat = img.reshape(-1)                                    # (H*W*3,)
        arr  = (flat[:N_LEDS * 3] + 1.0) / 2.0                   # [-1,1]→[0,1]
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, action: np.ndarray) -> np.ndarray:
        """
        Encode a full LED pattern into a latent code.
        Rarely needed during RL training — provided for analysis / replay.

        action  : (N_LEDS*3,) in [0, 1]
        returns : (LATENT_DIM,) float32
        """
        img  = self._action_to_tensor(action)
        z    = self.vae.encode(img).latent_dist.mean     # (1,4,H,W)
        # Scale back into SAC's expected input range
        z    = z / self.latent_scale
        return z.squeeze(0).flatten().cpu().numpy()

    @torch.no_grad()
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Decode a SAC latent code into a full LED pattern.
        Called every training step before sending pattern to RPi.

        latent  : (LATENT_DIM,) float32  in [-1, 1]  (SAC tanh output)
        returns : (N_LEDS*3,)   float32  in [0, 1]
        """
        z     = torch.from_numpy(latent.astype(np.float32)).to(self.device)
        z     = z * self.latent_scale                          # [-1,1] → VAE range
        z     = z.reshape(1, 4, LATENT_H, LATENT_W)
        recon = self.vae.decode(z).sample                      # (1,3,H,W)
        return self._tensor_to_action(recon)

    @torch.no_grad()
    def decode_batch(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Batch decode — useful if you ever want to inspect stored patterns.

        latents : (B, LATENT_DIM)  float32  in [-1, 1]
        returns : (B, N_LEDS*3)    float32  in [0, 1]
        """
        B  = latents.shape[0]
        z  = latents * self.latent_scale
        z  = z.reshape(B, 4, LATENT_H, LATENT_W).to(self.device)
        r  = self.vae.decode(z).sample                         # (B,3,H,W)
        r  = (r.clamp(-1, 1) + 1.0) / 2.0                     # [0,1]
        # (B,3,H,W) → (B,H*W*3) → (B,N_LEDS*3)
        out = r.permute(0, 2, 3, 1).reshape(B, -1)
        return out[:, :N_LEDS * 3]
