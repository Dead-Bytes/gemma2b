import torch
import torch.nn as nn
from typing import Optional, Tuple 

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size = 768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size =224,
        patch_size=16,
        layer_norm_eps = 1e-6,
        attention_dropout = 0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):

    def __init__(
            self,
            config : SiglipVisionConfig
    ): 
        super().__init__()

        self.config = config 
        self.embed_dim = config.hidden_size
        self.image_size  = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid" #this shows padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids", 
            torch.arange(self.num_positions).expand((1,-1)),
            persistent=False,
        )

    def forward(self, pixel_values : torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch_size, channels, hieght, width] 
        # Convolve the 'patch_size' kernel over the image , with no overlapping patches since stride = patch_size 
        # the output of the convolution will have the shape [Batch_size, Embed_dim, Num_patches_H, Num_patches_W]
        # where Num_Patches_H = height // patch_size and Num_patches_W = width // patch_size

        patch_embed  = self.patch_embedding(pixel_values)
        
        # [Batch_size, Embed_Dim, Num_Patches_H,Num_Patches_W] - > [Batch_size, Embed_Dim, Nium_patches]
        # where Num_Patches = Num_patches_H + Num_Patches_W

        embeddings  = patch_embed.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Sizem, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1,2)
        # Add postion Embedding to each patch 
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size , config.hidden_size)

    def forward(self, hidden_states : torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attention = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps = self.config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps = self.config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attention(hidden_states = hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states 

class SiglipVisionTransformer(nn.Module):

    def __init__(
            self,
            config : SiglipVisionConfig
    ):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)
        
    def forward(self, pixel_values : torch.Tensor) -> torch.Tensor:
        """ pixel_values : [Batch_Size, Channels,  Height, Widhth] -> [Batch_Size, Num_patches,Embed_Dim]"""
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.embeddings(input_embeds = hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state
    
class SiglipVisionModel(nn.Module):

    def __init__(self, config : SiglipVisionConfig):
        super().__init__()
        self.config  = config
        self.vision_model = SiglipVisionTransformer(config)

    def forwrd(self, pixel_values) -> Tuple:
        """[Batch_Size, Channels,  Height, Widhth] -> [Batch_Size, Num_patches,Embed_Dim]"""
        return self.vision_model(pixel_values = pixel_values)
    
