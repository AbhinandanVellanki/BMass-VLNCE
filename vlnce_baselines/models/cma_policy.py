from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net
from torch import Tensor

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.policy import ILPolicy


@baseline_registry.register_policy
class CMAPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ) -> None:
        super().__init__(
            CMANet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class CMANet(Net):
    """An implementation of the cross-modal attention (CMA) network in
    https://arxiv.org/abs/2004.02857
    """

    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ) -> None:
        super().__init__()
        self.model_config = model_config
        model_config.defrost()
        model_config.INSTRUCTION_ENCODER.final_state_only = False
        model_config.freeze()

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(
            model_config.INSTRUCTION_ENCODER
        )

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in ["VlnResnetDepthEncoder"]

        # Vision encoder training strategy
        vision_train_mode = getattr(model_config, "VISION_ENCODER_TRAINING", "full")
        if vision_train_mode == "full":
            rgb_trainable = True
            depth_trainable = True
        elif vision_train_mode == "freeze":
            rgb_trainable = False
            depth_trainable = False
        elif vision_train_mode == "partial":
            rgb_trainable = False
            depth_trainable = False
        elif vision_train_mode == "lora":
            rgb_trainable = False  # Placeholder, set up LoRA adapters here if implemented
            depth_trainable = False
        else:
            rgb_trainable = True
            depth_trainable = True

        self.depth_encoder = getattr(
            resnet_encoders, model_config.DEPTH_ENCODER.cnn_type
        )(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            trainable=depth_trainable,
            spatial_output=True,
        )

        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet18",
            "TorchVisionResNet50",
        ]
        self.rgb_encoder = getattr(
            resnet_encoders, model_config.RGB_ENCODER.cnn_type
        )(
            model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=model_config.normalize_rgb,
            trainable=rgb_trainable,
            spatial_output=True,
        )

        # If partial, unfreeze last layer(s) of encoder by index for modularity
        if vision_train_mode == "partial":
            # Freeze all params first
            for param in self.rgb_encoder.parameters():
                param.requires_grad = False
            for param in self.depth_encoder.parameters():
                param.requires_grad = False

            # Unfreeze last layer(s) of RGB encoder
            rgb_modules = list(self.rgb_encoder.modules())
            if len(rgb_modules) > 0:
                for param in rgb_modules[-1].parameters():
                    param.requires_grad = True

            # Unfreeze last layer(s) of Depth encoder
            depth_modules = list(self.depth_encoder.modules())
            if len(depth_modules) > 0:
                for param in depth_modules[-1].parameters():
                    param.requires_grad = True

        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        hidden_size = model_config.STATE_ENCODER.hidden_size
        self._hidden_size = hidden_size

        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.rgb_encoder.output_shape[0],
                model_config.RGB_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                model_config.DEPTH_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )

        # Init the RNN state decoder
        rnn_input_size = model_config.DEPTH_ENCODER.output_size
        rnn_input_size += model_config.RGB_ENCODER.output_size
        rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )

        self._output_size = (
            model_config.STATE_ENCODER.hidden_size
            + model_config.RGB_ENCODER.output_size
            + model_config.DEPTH_ENCODER.output_size
            + self.instruction_encoder.output_size
        )

        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            hidden_size // 2 + model_config.RGB_ENCODER.output_size,
            1,
        )

        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + model_config.DEPTH_ENCODER.output_size,
            1,
        )

        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size, hidden_size // 2, 1
        )
        self.text_q = nn.Linear(
            self.instruction_encoder.output_size, hidden_size // 2
        )

        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5))
        )

        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._output_size + self.prev_action_embedding.embedding_dim,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )

        self.second_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )
        self._output_size = model_config.STATE_ENCODER.hidden_size

        self.progress_monitor = nn.Linear(self.output_size, 1)

        self._init_layers()

        self.train()

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def is_blind(self) -> bool:
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers + (
            self.second_state_encoder.num_recurrent_layers
        )

    def _init_layers(self) -> None:
        if self.model_config.PROGRESS_MONITOR.use:
            nn.init.kaiming_normal_(
                self.progress_monitor.weight, nonlinearity="tanh"
            )
            nn.init.constant_(self.progress_monitor.bias, 0)

    def _attn(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v)

    def forward(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        instruction_embedding = self.instruction_encoder(observations)

        print("DEBUG")
        print("Input Shapes:" )
        for k, v in observations.items():
            print(f"  {k}: {v.shape}")
        # Compute clean and noisy depth embeddings
        depth_clean_emb = self.depth_encoder({**observations, 'depth': observations.get('depth_clean', observations.get('depth'))})
        depth_clean_emb = torch.flatten(depth_clean_emb, 2)
        depth_noisy_emb = self.depth_encoder({**observations, 'depth': observations.get('depth_noisy', observations.get('depth'))})
        depth_noisy_emb = torch.flatten(depth_noisy_emb, 2)

        # Compute clean and noisy rgb embeddings
        rgb_clean_emb = self.rgb_encoder({**observations, 'rgb': observations.get('rgb_clean', observations.get('rgb'))})
        rgb_clean_emb = torch.flatten(rgb_clean_emb, 2)
        rgb_noisy_emb = self.rgb_encoder({**observations, 'rgb': observations.get('rgb_noisy', observations.get('rgb'))})
        rgb_noisy_emb = torch.flatten(rgb_noisy_emb, 2)

        # Use noisy embeddings for main policy (default behavior)
        rgb_embedding = rgb_noisy_emb
        depth_embedding = depth_noisy_emb

        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        rgb_in = self.rgb_linear(rgb_embedding)
        depth_in = self.depth_linear(depth_embedding)

        state_in = torch.cat([rgb_in, depth_in, prev_actions], dim=1)
        rnn_states_out = rnn_states.detach().clone()
        (
            state,
            rnn_states_out[:, 0 : self.state_encoder.num_recurrent_layers],
        ) = self.state_encoder(
            state_in,
            rnn_states[:, 0 : self.state_encoder.num_recurrent_layers],
            masks,
        )

        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )

        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1
        )
        depth_k, depth_v = torch.split(
            self.depth_kv(depth_embedding), self._hidden_size // 2, dim=1
        )

        text_q = self.text_q(text_embedding)
        rgb_embedding = self._attn(text_q, rgb_k, rgb_v)
        depth_embedding = self._attn(text_q, depth_k, depth_v)

        x = torch.cat(
            [
                state,
                text_embedding,
                rgb_embedding,
                depth_embedding,
                prev_actions,
            ],
            dim=1,
        )
        x = self.second_state_compress(x)
        (
            x,
            rnn_states_out[:, self.state_encoder.num_recurrent_layers :],
        ) = self.second_state_encoder(
            x,
            rnn_states[:, self.state_encoder.num_recurrent_layers :],
            masks,
        )

        if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1),
                observations["progress"],
                reduction="none",
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.model_config.PROGRESS_MONITOR.alpha,
            )

        # Add MSE loss between clean and noisy instruction features
        if AuxLosses.is_active() and hasattr(
            self.instruction_encoder, "clean_features"
        ):
            if (
                self.instruction_encoder.clean_features is not None
                and self.instruction_encoder.noisy_features is not None
            ):
                # Compute MSE loss between clean and noisy encoded features
                noisy_feat = self.instruction_encoder.noisy_features
                clean_feat = self.instruction_encoder.clean_features.detach()
                # Handle sequence length mismatch by padding to the same length
                if noisy_feat.shape[-1] != clean_feat.shape[-1]:
                    max_len = max(noisy_feat.shape[-1], clean_feat.shape[-1])
                    if noisy_feat.shape[-1] < max_len:
                        pad_size = max_len - noisy_feat.shape[-1]
                        noisy_feat = F.pad(noisy_feat, (0, pad_size))
                    if clean_feat.shape[-1] < max_len:
                        pad_size = max_len - clean_feat.shape[-1]
                        clean_feat = F.pad(clean_feat, (0, pad_size))

                # Compute MSE loss between clean and noisy encoded features
                text_denoising_loss = F.mse_loss(
                    noisy_feat,
                    clean_feat,
                    reduction="none",
                )
                print(f"text_denoising_loss: {text_denoising_loss.mean()}")

                # import pdb; pdb.set_trace()
                # Average over feature dimension
                text_denoising_loss = text_denoising_loss.sum(dim=[1, -1])

                # Register the auxiliary loss
                # You can configure the alpha weight in your config
                alpha = getattr(
                    self.model_config, "TEXT_DENOISING_ALPHA", 0.1
                )
                AuxLosses.register_loss(
                    "text_denoising",
                    text_denoising_loss,
                    alpha,
                )

        return x, rnn_states_out, {
            'rgb_clean_emb': rgb_clean_emb,
            'rgb_noisy_emb': rgb_noisy_emb,
            'depth_clean_emb': depth_clean_emb,
            'depth_noisy_emb': depth_noisy_emb,
        }