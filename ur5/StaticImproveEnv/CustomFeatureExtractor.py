import gym
import torch as th
from torch import nn
from gym import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        
        self.queries_fc_layer = nn.Conv1d(in_channels=pts_feat_dim, out_channels=n_filters_qkv, kernel_size=1)
        self.keys_fc_layer = nn.Conv1d(in_channels=img_feat_dim, out_channels=n_filters_qkv, kernel_size=1)
        self.values_fc_layer = nn.Conv1d(in_channels=img_feat_dim, out_channels=n_filters_qkv, kernel_size=1)

        self.y_out_fc_layer = nn.Conv1d(in_channels=n_filters_qkv, out_channels=learned_feat_dim, kernel_size=1)
        if out_dim:
            self.fused_feat_fc_layer = nn.Conv1d(
                in_channels=(learned_feat_dim + pts_feat_dim), out_channels=out_dim, kernel_size=1
            )

        self.dropout = nn.Dropout(p=0.3)

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                n_input_channels = subspace.shape[0]
                
                queries = nn.Linear()
                
                extractors[key] = nn.Sequential(
                                                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                            )
                
                with th.no_grad():
                    total_concat_size += extractors[key](
                        th.as_tensor(subspace.sample()[None]).float()
                    ).shape[1]
                
                #total_concat_size += 64
            elif key == "position":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)