from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class PatchEmbedding(nn.Module):
        def __init__(self, img_size, patch_size, in_channels, embed_dim=None, add_cls_token=True,
                     add_pos_embedding=True, add_distil_token=False, projection="conv", mask_ratio=0):
                """
                Creates patches from the input image and projects them to embed_dim
                (B, C, H, W) -> (B, num_patches, embed_dim)
                :param img_size: Size of the input image
                :param patch_size: Size of the patches to be extracted from the image
                :param in_channels: Number of channels
                :param embed_dim: Embedding dimension
                :param add_cls_token: Whether to add a learnable class token to the patch embeddings. This token will be the first token in the sequence
                :param add_pos_embedding: Whether to add positional embeddings to the patch embeddings
                :param add_distil_token: Whether to add a learnable distillation token to the patch embeddings. This token will be the last token in the sequence
                :param projection: Type of projection layer to use. Options: "conv" or "Unfold". Default: "conv". Unfold uses unfold to make patches keeping the image as it is, without any learnable parameters.
                :param mask_ratio: Ratio of patches to be masked during training
                """
                super().__init__()

                if projection not in ["conv", "unfold"]:
                        raise ValueError("Projection must be either 'conv' or 'unfold'")
                if projection == "conv" and embed_dim is None:
                        raise ValueError("If projection is 'conv', embed_dim must be provided")
                if projection == "unfold" and embed_dim is not None:
                        raise ValueError(
                                "If projection is 'unfold', embed_dim must not be provided. Embed_dim will automatically be set to: in_channels * (patch_size ** 2)")

                self.num_patches = (img_size // patch_size) ** 2
                self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size,) if projection == "conv" \
                        else nn.Unfold(kernel_size=patch_size, stride=patch_size)
                self.embed_dim = embed_dim if embed_dim is not None else in_channels * (patch_size ** 2)

                self.cls_token = None
                self.distil_token = None
                self.pos_embedding = None
                self.mask_ratio = mask_ratio
                if add_cls_token:
                        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
                        self.num_patches += 1
                if add_distil_token:
                        self.distil_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
                        self.num_patches += 1
                if add_pos_embedding:
                        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.embed_dim))

        def forward(self, x) -> torch.Tensor:
                # If the input is a 3D tensor, add a channel dimension
                if x.dim() == 3:
                        x = x.unsqueeze(1)

                # Extract the dimensions of the input tensor
                B, C, H, W = x.shape

                # Project the input to the embedding dimension
                x = self.projection(x)  # Shape: (B, embed_dim, H/patch_size, W/patch_size)
                original_x = x.flatten(2).transpose(1, 2)  # Shape: (B, num_patches, embed_dim)

                if self.mask_ratio > 0:
                        # Calculate the number of non-masked tokens
                        num_non_masked = int(
                                (self.num_patches - 1) * (1 - self.mask_ratio)) if self.cls_token is not None else int(
                                self.num_patches * (1 - self.mask_ratio))

                        # Generate a random mask and find the indices of the non-masked tokens
                        non_mask = torch.rand(B, (self.num_patches - 1), device=x.device).topk(num_non_masked,
                                                                                               dim=1).indices

                        # Gather non-masked tokens
                        x = original_x.gather(1, non_mask.unsqueeze(-1).expand(-1, -1, original_x.size(-1)))

                        # Save the indices of non-masked tokens
                        non_masked_indices = non_mask

                        # If the cls token is present, add it to the beginning of the sequence
                        if self.cls_token is not None:
                                x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
                                non_masked_indices = torch.cat(
                                        [torch.zeros(B, 1, dtype=torch.long, device=x.device), non_masked_indices + 1],
                                        dim=1)

                        # If the distillation token is present, add it to the end of the sequence
                        if self.distil_token is not None:
                                x = torch.cat([x, self.distil_token.expand(x.size(0), -1, -1)], dim=1)
                                non_masked_indices = torch.cat(
                                        [non_masked_indices, torch.full((B, 1), self.num_patches - 1, dtype=torch.long, device=x.device)],
                                        dim=1)

                        # Add positional embeddings
                        if self.pos_embedding is not None:
                                pos_embed = self.pos_embedding.expand(B, -1, -1).gather(
                                        1,
                                        non_masked_indices.unsqueeze(-1).expand(B, -1, self.embed_dim))
                                x += pos_embed
                        else:
                                x = x + self.pos_embedding

                        return x, original_x, non_masked_indices  # Randomly masked patches, Original patches, Masked indices

                x = original_x
                # If the cls token is present, add it to the beginning of the sequence
                if self.cls_token is not None:
                        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)

                # If the distillation token is present, add it to the end of the sequence
                if self.distil_token is not None:
                        x = torch.cat([x, self.distil_token.expand(x.size(0), -1, -1)], dim=1)

                # Add positional embeddings
                if self.pos_embedding is not None:
                        x = x + self.pos_embedding

                return x  # Shape: (B, num_patches, embed_dim)


class MultiHeadGlobalAttention(nn.Module):
        def __init__(self, embed_dim, num_heads):
                """
                Multi-head self-attention mechanism
                :param embed_dim: Embedding dimension
                :param num_heads: Number of attention heads
                """
                super().__init__()
                self.num_heads = num_heads
                self.embed_dim = embed_dim
                self.head_dim = embed_dim // num_heads
                assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

                self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
                self.o_proj = nn.Linear(embed_dim, embed_dim)
                self.scale = self.head_dim ** -0.5

        def forward(self, x):
                # Shape (Batch_size, Sequence_len, Embedding_dim)
                if x.dim() != 3:
                        raise ValueError(
                                "Input tensor must have 3 dimensions (Batch_size, Sequence_len, Embedding_dim)")

                batch_size, seq_length, embed_dim = x.size()  # Shape: (batch_size, seq_length, embed_dim)

                qkv = self.qkv_proj(x)  # Shape: (batch_size, seq_length, embed_dim * 3)

                qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1)  # Shape: (num_heads, batch_size, 3 * head_dim, seq_length)
                q, k, v = qkv.chunk(3, dim=2)  # Shape: (num_heads, batch_size, head_dim, seq_length)

                q = q.permute(1, 0, 3, 2)  # Shape: (batch_size, num_heads, seq_length, head_dim)
                k = k.permute(1, 0, 3, 2)  # Shape: (batch_size, num_heads, seq_length, head_dim)
                v = v.permute(1, 0, 3, 2)  # Shape: (batch_size, num_heads, seq_length, head_dim)

                attn_weights = torch.matmul(q, k.transpose(-2,
                                                           -1)) * self.scale  # Shape: (batch_size, num_heads, seq_length, seq_length)
                attn_weights = F.softmax(attn_weights, dim=-1)
                attn_output = torch.matmul(attn_weights, v)  # Shape: (batch_size, num_heads, seq_length, head_dim)

                attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_length,
                                                                      embed_dim)  # Shape: (batch_size, seq_length, embed_dim)
                output = self.o_proj(attn_output)  # Shape: (batch_size, seq_length, embed_dim)
                return output  # Shape: (batch_size, seq_length, embed_dim)


class MultiHeadLocalAttention(nn.Module):
        def __init__(self, embed_dim, num_heads, window_size, dropout=0.1, cls_token_in_every_window=False):
                """
                Multi-head local attention mechanism
                :param embed_dim: Embedding dimension
                :param num_heads: Number of attention heads
                :param window_size: Size of the local window
                :param dropout: Dropout value
                :param cls_token_in_every_window: Assuming that cls token is the first token in the sequence, whether to add cls token to the window of every other token or not.
                        If False, will treat cls token as a normal token.
                """
                super().__init__()
                assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.head_dim = embed_dim // num_heads
                self.window_size = window_size
                self.padding = (window_size - 1) // 2  # Padding for same-length output
                self.cls_token_in_every_window = cls_token_in_every_window

                # Projection layers
                self.q_proj = nn.Linear(embed_dim, embed_dim)
                self.k_proj = nn.Linear(embed_dim, embed_dim)
                self.v_proj = nn.Linear(embed_dim, embed_dim)
                self.out_proj = nn.Linear(embed_dim, embed_dim)

                self.dropout = nn.Dropout(dropout)

        def forward(self, x):
                """
                Args:
                    x: Tensor of shape (batch_size, seq_len, embed_dim)
                Returns:
                    Tensor of shape (batch_size, seq_len, embed_dim)
                """

                batch_size, seq_len, _ = x.shape
                H, D = self.num_heads, self.head_dim

                head_dim = self.head_dim

                # Project inputs to Q, K, V
                Q = self.q_proj(x)  # (B, L, E)
                K = self.k_proj(x)  # (B, L, E)
                V = self.v_proj(x)  # (B, L, E)

                # Reshape for multi-head attention
                Q = Q.view(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, L, D)
                K = K.view(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, L, D)
                V = V.view(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, L, D)

                if self.cls_token_in_every_window:
                        # Process class token with global attention
                        cls_Q = Q[:, :, 0:1, :]  # (B, H, 1, D)
                        attn_scores_cls = torch.einsum('bhqd,bhkd->bhqk', cls_Q, K) / (D ** 0.5)
                        attn_weights_cls = F.softmax(attn_scores_cls, dim=-1)
                        attn_weights_cls = self.dropout(attn_weights_cls)
                        output_cls = torch.einsum('bhqk,bhkd->bhqd', attn_weights_cls, V)  # (B, H, 1, D)

                        if seq_len == 1:
                                output = output_cls
                        else:
                                # Process rest tokens with local attention, adding cls token to each window
                                Q_rest = Q[:, :, 1:, :]  # (B, H, L_rest, D)
                                L_rest = seq_len - 1

                                # Pad K and V for local windows
                                K_padded = F.pad(K, (0, 0, self.padding, self.padding), value=0)
                                V_padded = F.pad(V, (0, 0, self.padding, self.padding), value=0)

                                # Unfold K and V to get local windows
                                K_unfold = K_padded.unfold(2, self.window_size, 1).permute(0, 1, 2, 4,
                                                                                           3)  # (B, H, L, W, D)
                                V_unfold = V_padded.unfold(2, self.window_size, 1).permute(0, 1, 2, 4, 3)

                                # Extract windows for rest tokens (positions 1 to L-1)
                                K_unfold_rest = K_unfold[:, :, 1:, :, :]  # (B, H, L_rest, W, D)
                                V_unfold_rest = V_unfold[:, :, 1:, :, :]

                                # Get cls token's K and V and expand to match rest tokens
                                cls_K = K[:, :, 0:1, :]  # (B, H, 1, D)
                                cls_V = V[:, :, 0:1, :]
                                cls_K_expanded = cls_K.unsqueeze(3).repeat(1, 1, L_rest, 1, 1)  # (B, H, L_rest, 1, D)
                                cls_V_expanded = cls_V.unsqueeze(3).repeat(1, 1, L_rest, 1, 1)  # (B, H, L_rest, 1, D)
                                # cls_K_expanded = cls_K.expand(-1, -1, L_rest, -1, -1)  # (B, H, L_rest, 1, D)
                                # cls_V_expanded = cls_V.expand(-1, -1, L_rest, -1, -1)

                                # Append cls token's K and V to each window
                                K_rest_with_cls = torch.cat([K_unfold_rest, cls_K_expanded],
                                                            dim=3)  # (B, H, L_rest, W+1, D)
                                V_rest_with_cls = torch.cat([V_unfold_rest, cls_V_expanded], dim=3)

                                # Create mask for rest tokens including cls token
                                mask_rest = torch.ones((batch_size, H, L_rest), dtype=torch.bool, device=x.device)
                                mask_rest_padded = F.pad(mask_rest, (self.padding, self.padding), value=False)
                                mask_unfold_rest = mask_rest_padded.unfold(2, self.window_size, 1)  # (B, H, L_rest, W)
                                mask_rest_with_cls = torch.cat(
                                        [mask_unfold_rest,
                                         torch.ones((batch_size, H, L_rest, 1), dtype=torch.bool, device=x.device)],
                                        dim=-1
                                )

                                # Compute attention for rest tokens
                                attn_scores_rest = torch.einsum('bhld,bhlwd->bhlw', Q_rest, K_rest_with_cls) / (
                                        D ** 0.5)
                                attn_scores_rest = attn_scores_rest.masked_fill(~mask_rest_with_cls, float('-inf'))
                                attn_weights_rest = F.softmax(attn_scores_rest, dim=-1)
                                attn_weights_rest = self.dropout(attn_weights_rest)
                                output_rest = torch.einsum('bhlw,bhlwd->bhld', attn_weights_rest, V_rest_with_cls)

                                # Combine cls and rest outputs
                                output = torch.cat([output_cls, output_rest], dim=2)  # (B, H, L, D)

                        # Reshape and project
                        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
                        output = self.out_proj(output)

                else:
                        # Pad and unfold K and V
                        K_padded = F.pad(K, (0, 0, self.padding, self.padding), value=0)  # (B, H, L+pad*2, D)
                        V_padded = F.pad(V, (0, 0, self.padding, self.padding), value=0)  # (B, H, L+pad*2, D)

                        # Unfold into windows (B, H, L, W, D)
                        K_unfold = K_padded.unfold(2, self.window_size, 1).permute((0, 1, 2, 4, 3))
                        V_unfold = V_padded.unfold(2, self.window_size, 1).permute((0, 1, 2, 4, 3))

                        # Create attention mask
                        mask = torch.ones((batch_size, self.num_heads, seq_len), dtype=torch.bool, device=x.device)
                        mask_padded = F.pad(mask, (self.padding, self.padding), value=False)  # (B, H, L+pad*2)
                        mask_unfold = mask_padded.unfold(2, self.window_size, 1)  # (B, H, L, W)

                        # Compute attention scores (B, H, L, W)
                        attn_scores = torch.einsum('bhld,bhlwd->bhlw', Q, K_unfold) / (head_dim ** 0.5)

                        # Mask invalid positions
                        attn_scores = attn_scores.masked_fill(~mask_unfold, float('-inf'))

                        # Compute attention weights
                        attn_weights = F.softmax(attn_scores, dim=-1)
                        attn_weights = self.dropout(attn_weights)

                        # Apply attention to values
                        output = torch.einsum('bhlw,bhlwd->bhld', attn_weights, V_unfold)

                        # Reshape and project back
                        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)
                        output = self.out_proj(output)

                return output


class TransformerEncoderLayer(nn.Module):
        def __init__(self, embed_dim, num_heads, mlp_dim, dropout, window_size=None, cls_token_in_every_window=False):
                super().__init__()
                self.attention = MultiHeadLocalAttention(embed_dim, num_heads, window_size, dropout,
                                                         cls_token_in_every_window) if window_size is not None else MultiHeadGlobalAttention(
                        embed_dim, num_heads)
                self.norm1 = nn.LayerNorm(embed_dim)
                self.mlp = nn.Sequential(
                        nn.Linear(embed_dim, mlp_dim),
                        nn.GELU(),
                        nn.Linear(mlp_dim, embed_dim),
                )
                self.norm2 = nn.LayerNorm(embed_dim)
                self.dropout = nn.Dropout(dropout)

                self.window_size = window_size

        def forward(self, x):
                x = x + self.dropout(self.attention(self.norm1(x)))
                x = x + self.dropout(self.mlp(self.norm2(x)))
                return x


class TransformerEncoder(nn.Module):
        def __init__(self, embed_dim, num_heads, num_layers, mlp_dim, dropout, window_size=None,
                     cls_token_in_every_window=False, return_all_outputs=False):
                super().__init__()
                if window_size is not None:
                        assert len(window_size) == num_layers, "Window size must be provided for each layer"
                if window_size is None:
                        window_size = [None] * num_layers # Global attention for all layers
                self.return_all_outputs = return_all_outputs
                self.layers = nn.ModuleList([
                        TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout, window_size[_],
                                                cls_token_in_every_window)
                        for _ in range(num_layers)
                ])

        def forward(self, x):
                """
                Forward pass
                :param x: Input tensor (B, num_patches, embed_dim)
                :return: Output tensor (B, num_patches, embed_dim) or list of output tensors of all transformer encoder layers
                """
                if self.return_all_outputs:
                        outputs = []
                        for layer in self.layers:
                                x = layer(x)
                                outputs.append(x)
                        return outputs
                for layer in self.layers:
                        x = layer(x)
                return x


class VisionTransformer(nn.Module):
        def __init__(self, img_size: int, patch_size: int, in_channels: int, num_classes: int, embed_dim: int, num_heads: int, num_layers: int, mlp_dim: int,
        dropout: float, all_outputs: bool = False, final_norm: bool = True, final_head: bool = True, window_size: Optional[List[int]] = None,
        add_cls_token: bool = True, add_pos_embedding: bool = True, add_distil_token: bool = False,
        cls_token_in_every_window: bool = False, make_patch_embedding: bool = True):
                """
                Implementation of Vision Transformer
                :param img_size: size of image
                :param patch_size: size of patches to be extracted from the image
                :param in_channels: number of input channels
                :param num_classes: number of classes
                :param embed_dim: embedding dimension
                :param num_heads: number of attention heads
                :param num_layers: number of transformer encoder layers
                :param mlp_dim: dimension of the feedforward network (embed_dim -> mlp_dim -> embed_dim)
                :param dropout: dropout value ratio (range: 0-1)
                :param all_outputs: return the output of all transformer encoder layers
                :param final_norm: apply layer normalization to the final output or not
                :param final_head: apply a linear layer (to project cls_token to num_classes) to the final output or not
                :param window_size: window size for each transformer encoder layer. If None (for any layer), the attention will be global attention. For example: [3, 3, 5, 5, None, None] will use window size 3 for the first two layers, 5 for the next two layers, and no window size (global attention) for the last two layers
                :param add_cls_token: add a learnable class token to the patch embeddings
                :param add_pos_embedding: add positional embeddings to the patch embeddings
                :param add_distil_token: add a learnable distillation token to the patch embeddings
                :param cls_token_in_every_window: add the class token to every window in the patch embeddings
                :param make_patch_embedding: make a patch embedding layer or not. If False, the input tensor is expected to be a patch embedding tensor
                """
                super().__init__()
                self.all_outputs = all_outputs

                self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim,
                                                  add_cls_token=add_cls_token, add_pos_embedding=add_pos_embedding,
                                                  add_distil_token=add_distil_token) if make_patch_embedding else None
                self.embed_dim = patch_size * patch_size * in_channels if embed_dim is None else embed_dim

                self.dropout = nn.Dropout(dropout)

                assert len(
                        window_size) == num_layers if window_size is not None else True, "Window size must be provided for each layer"
                self.encoder = TransformerEncoder(self.embed_dim, num_heads, num_layers, mlp_dim, dropout, window_size,
                                                  cls_token_in_every_window, all_outputs)

                self.norm = nn.LayerNorm(self.embed_dim) if final_norm else None
                self.head = nn.Linear(self.embed_dim, num_classes) if final_head else None

        def forward(self, x):
                """
                Forward pass
                :param x: Shape (B, C, H, W)
                :return: Output of the transformer encoder (B, num_classes) or list of outputs of all transformer encoder layers
                """
                x = self.patch_embed(x) if self.patch_embed is not None else x
                x = self.dropout(x)

                x = self.encoder(x)
                if self.all_outputs:
                        return x

                x = self.norm(x) if self.norm is not None else x
                cls_token_final = x[:, 0]  # Extract the [CLS] token

                return self.head(cls_token_final) if self.head is not None else cls_token_final


if __name__ == '__main__':
        # Global Attention
        model = VisionTransformer(img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, num_heads=12, num_layers=12, mlp_dim=3072, dropout=0.1)
        model = model.to('cuda')
        x = torch.randn(1, 3, 224, 224).to('cuda')
        torchsummary.summary(model, (3, 224, 224))
        print(model(x).shape)

        # Local Attention
        model = VisionTransformer(img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, num_heads=12, num_layers=12, mlp_dim=3072, dropout=0.1, window_size=[3, 3, 5, 5, 7, 7, 9, 9, None, None, None, None])
        model = model.to('cuda')
        x = torch.randn(1, 3, 224, 224).to('cuda')
        torchsummary.summary(model, (3, 224, 224))
        print(model(x).shape)