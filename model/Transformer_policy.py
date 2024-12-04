from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
import gymnasium as gym



import torch as th
import torch.nn as nn
from typing import Union, List, Dict, Type, Tuple
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


from stable_baselines3.common.utils import get_device
from stable_baselines3.common.type_aliases import Schedule


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence



def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    Used by the ``FlattenExtractor`` to compute the input shape.

    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)
               

'''
Transformer Block
'''    
class MaskedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MaskedAttention, self).__init__()
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first = True)

    def forward(self, x, mask):
        # x: [batch_size, n_nodes, hidden_dim]
        # mask: [batch_size, n_nodes]

        # Perform attention with the modified mask
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=mask)
        
        # Transpose the attention output back to the original input shape
        # attn_output = attn_output.permute(1, 0, 2)  # [batch_size, n_nodes, hidden_dim]

        return attn_output

class TransformerLayerWithMaskedAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(TransformerLayerWithMaskedAttention, self).__init__()
        self.masked_attn = MaskedAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * num_heads),
            nn.ReLU(),
            nn.Linear(hidden_dim * num_heads, hidden_dim)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        # x: [batch_size, n_nodes, hidden_dim]
        # mask: [batch_size, n_nodes]

        # Apply masked attention
        attn_output = self.masked_attn(x, mask)
        
        # Add & norm
        x = self.layer_norm1(x + attn_output)
        
        # Feedforward network
        ff_output = self.feedforward(x)
        
        # Add & norm
        x = self.layer_norm2(x + self.dropout(ff_output))

        return x

class TransformerWithMaskedAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_layers=4):
        super(TransformerWithMaskedAttention, self).__init__()
        
        self.num_heads = num_heads
        
        # Stacking multiple Transformer layers with masked attention
        self.layers = nn.ModuleList([
            TransformerLayerWithMaskedAttention(hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])
        

    def forward(self, x, mask):
        # x: [batch_size, n_nodes, hidden_dim]
        # mask: [batch_size, n_nodes]
        
        # Expand mask to match the dimensions of the attention matrix
        mask = mask.unsqueeze(2) | mask.unsqueeze(1)  # [batch_size, n_nodes, n_nodes]

        # Modify the attention mechanism by applying the mask
        mask = mask.float().masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float('-inf'))

        # Repeat the mask for multi-head attention
        mask = mask.repeat_interleave(self.num_heads, dim=0)  # [batch_size * num_heads, n_nodes, n_nodes]

        # MultiheadAttention expects the mask to be of shape [n_nodes, n_nodes] or [batch_size*num_heads, n_nodes, n_nodes]
        # Here, we'll use [batch_size, n_nodes, n_nodes] which works with MultiheadAttention

        for layer in self.layers:
            x = layer(x, mask)


        return x


'''
Transformer With Dummy
'''
class TransformerWithMaskedAttention_cls(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_layers=4, dummy_size=32):
        super(TransformerWithMaskedAttention_cls, self).__init__()
        
        self.num_heads = num_heads
        self.dummy_size = dummy_size
        
        # Stacking multiple Transformer layers with masked attention
        self.layers = nn.ModuleList([
            TransformerLayerWithMaskedAttention(hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        # Projection layer to match the hidden_dim with dummy_size
        self.proj = nn.Linear(hidden_dim, dummy_size)
        

    def forward(self, x, mask):
        batch_size = x.size(0)
        n_nodes = x.size(1)
        hidden_dim = x.size(2)
        
        # Create a dummy tensor and concatenate it to the input
        dummy_tensor = torch.ones(batch_size, 1, hidden_dim, device=x.device)
        x = torch.cat([dummy_tensor, x], dim=1)  # [batch_size, 1 + n_nodes, hidden_dim]

        # Adjust the mask to include the dummy
        mask = torch.cat([torch.ones((batch_size, 1), dtype=torch.bool, device=mask.device), mask], dim=1)  # [batch_size, 1 + n_nodes]

        mask = mask.unsqueeze(2) | mask.unsqueeze(1)   # [batch_size, 1 + n_nodes, 1 + n_nodes]
        mask = mask.float().masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float('-inf'))
        mask[:, :, 0] = 0.0

        mask = mask.repeat_interleave(self.num_heads, dim=0)  # [batch_size * num_heads, 1 + n_nodes, 1 + n_nodes]

        # Pass through each transformer layer
        for layer in self.layers:
            x = layer(x, mask)

        # Extract and return the dummy tensor (first element)
        dummy_output = x[:, 0, :]  # [batch_size, hidden_dim]

        # Optionally project the dummy tensor to match the dummy size
        dummy_output = self.proj(dummy_output)  # [batch_size, dummy_size]

        return dummy_output
'''
end
'''    
          
class CustomTransformerPolicy(nn.Module):
    """
    Constructs a transformer-based model that receives the output from a previous features extractor
    (i.e., a CNN) or directly the observations (if no features extractor is applied) as input and outputs
    a latent representation for the policy and a value network.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    :param hidden_dim: The hidden dimension size of the transformer.
    :param num_layers: Number of transformer layers.
    :param num_heads: Number of attention heads in the transformer.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        num_heads = 4,
        num_layers = 4,
        input_dim = 48,
        max_seq = 20,
        computing_setting = "custom",
        ) -> None:
        super().__init__()
        
        device = get_device(device)
        self.device = device
        self.max_seq = max_seq
        self.setting = computing_setting
        self.input_dim = input_dim
        
        self.embedder = nn.Linear(input_dim, feature_dim).to(device) 
        self.transformer_pi = TransformerWithMaskedAttention_cls(feature_dim, num_heads=num_heads, num_layers=num_layers, dummy_size=feature_dim)
        self.transformer_v = TransformerWithMaskedAttention_cls(feature_dim, num_heads=num_heads, num_layers=num_layers, dummy_size=feature_dim)
                
        #general flatten function
        self.flatten = nn.Flatten(start_dim=1).to(device)
        
        value_net: List[nn.Module] = []
        policy_net: List[nn.Module] = []
                
        last_layer_dim_vf = feature_dim
        last_layer_dim_pi = feature_dim
        
        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specified, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
            
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        
        # Iterate through the value layers and build the value net
        value_net.append(self.flatten)
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim
        

        # Save dim, used to create the distributions
        #NOT IMPORTANT
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.value_net = nn.Sequential(*value_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
 
        
    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        features, mask = self.process_obs(features)
        transformer_inputs = self.embedder(features)
        transformer_output = self.transformer_pi(transformer_inputs, mask)
        pi_output = self.policy_net(transformer_output)
        return pi_output 
        
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features, mask = self.process_obs(features)
        transformer_inputs = self.embedder(features) 
        transformer_output = self.transformer_v(transformer_inputs, mask)
        vf_output = self.value_net(transformer_output)
        return vf_output
    
    def observations_length(self, observations):
        lengths = []
        for obs in observations:
            length = self.max_seq
            for i, elem in enumerate(obs):
                if elem[-1] == -1:
                    length = i
                    break
            lengths.append(length)
        return lengths
    
    def process_obs(self, observations):
        if isinstance(observations, list):
            observations = torch.stack(observations, dim=0)

        lengths = self.observations_length(observations)
        

        if self.setting == "simple":
            observations = [F.one_hot(obs[:lengths[i], -1].to(torch.int64), num_classes=self.input_dim) for i, obs in enumerate(observations)]
        else:
            observations = [obs[:lengths[i], :-1] for i, obs in enumerate(observations)]
        padded_sequences = pad_sequence(observations, batch_first=True).float()

        attention_mask = torch.zeros(padded_sequences.shape[:2], dtype=torch.bool, device=self.device) 
        for i, length in enumerate(lengths):
            attention_mask[i, length:] = True
        
        return padded_sequences.to(self.device), attention_mask.to(self.device)
    
class NullFeatureExtractor(BaseFeaturesExtractor):
    """
    acts nothing, gives the tensor of the exact shape to the transformer.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space, features_dim: int) -> None:
        super().__init__(observation_space, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # return (observations["node_features"]
        return observations
    
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NullFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        # Disable orthogonal initialization
        ortho_init = False
        self.input_dim = features_extractor_kwargs["input_dim"]
        self.num_heads =  features_extractor_kwargs["num_heads"]
        self.num_layers = features_extractor_kwargs["num_layers"]
        self.transformer_setting = features_extractor_kwargs["computing_setting"]
        self.max_seq = features_extractor_kwargs["max_seq"]
        features_extractor_kwargs = dict(features_dim=features_extractor_kwargs["features_dim"])
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.share_features_extractor = True
        self._build(lr_schedule)
        
               
    def _build_mlp_extractor(self) -> None:
         self.mlp_extractor = CustomTransformerPolicy(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            num_heads = self.num_heads,
            num_layers = self.num_layers,
            input_dim = self.input_dim,
            max_seq = self.max_seq,
            computing_setting = self.transformer_setting)