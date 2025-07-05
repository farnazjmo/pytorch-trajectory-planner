from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

# With help of AI

class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

         # Input shape: (B, n_track, 2) x2 => (B, n_track * 4)
        self.model = nn.Sequential(
            nn.Linear(n_track * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2),  # Predict x, y for each waypoint
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate along the last dimension and flatten
        x = torch.cat([track_left, track_right], dim=-1)  # shape: (B, n_track, 4)
        x = x.view(x.size(0), -1)  # shape: (B, n_track * 4)
        x = self.model(x)  # shape: (B, n_waypoints * 2)
        return x.view(-1, self.n_waypoints, 2)  # shape: (B, n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 192,
        nhead: int = 4,
        num_layers: int = 4,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Input encoder: map (x, y) => d_model
        self.input_proj = nn.Linear(3, d_model)  # 2 coords + 1 position

        # Query embeddings for each waypoint
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Weight initialization for better attention convergence
        nn.init.xavier_uniform_(self.query_embed.weight)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,            
            batch_first=True,
            norm_first=True         
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers   
        )

        # Output layer to (x, y)
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        B = track_left.shape[0]

        # Concatenate track boundaries: (B, 2*n_track, 2)
        track = torch.cat([track_left, track_right], dim=1)  # (B, 2 * n_track, 2)

        # Positional encoding: evenly spaced values between -1 and 1
        pos = torch.linspace(-1, 1, steps=track.shape[1], device=track.device)
        pos = pos.unsqueeze(0).unsqueeze(-1).expand(track.shape[0], -1, 1)  # (B, 2n, 1)

        track = torch.cat([track, pos], dim=-1)  # (B, 2n, 3)
        
        
        # Normalize between -1 and 1 or z-score
        track_mean = track.mean(dim=1, keepdim=True)
        track_std = track.std(dim=1, keepdim=True) + 1e-6
        track = (track - track_mean) / track_std  # z-score normalization

        # Project 3D input into d_model
        memory = self.input_proj(track)
        

        # Query embeddings: (n_waypoints, d_model) -> (B, n_waypoints, d_model)
        query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Decode waypoints
        decoded = self.decoder(query_embed, memory)  # (B, n_waypoints, d_model)
        waypoints = self.output_proj(decoded)  # (B, n_waypoints, 2)

        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Simple CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),  # (96x128) -> (48x64)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # -> (24x32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # -> (12x16)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # -> (64, 1, 1)
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        
        # ensure image is float32 and on same device as model
        image = image.to(dtype=torch.float32, device=self.input_mean.device)

        # normalize
        mean = self.input_mean[None, :, None, None]
        std = self.input_std[None, :, None, None]
        x = (image - mean) / std

        x = self.backbone(x)  # (B, 64, 1, 1)
        x = self.mlp(x)  # (B, n_waypoints * 2)
        return x.view(-1, self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
