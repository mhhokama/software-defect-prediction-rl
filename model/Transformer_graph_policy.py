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
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from stable_baselines3.common.distributions import Distribution

from stable_baselines3.common.distributions import CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution, Categorical
from functools import partial
import numpy as np

SelfTransformerCategoricalDistribution = TypeVar("SelfTransformerCategoricalDistribution", bound="TransformerCategoricalDistribution")



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
        mask = mask.reshape(-1, mask.shape[-1], mask.shape[-1])
        mask = mask.repeat_interleave(x.size(0), dim=0)
        mask = mask.float().masked_fill(mask == 0, float(0.0)).masked_fill(mask == 1, float('-inf'))

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
        

    def forward(self, x):
        batch_size = x.size(0)
        n_nodes = x.size(1)
        hidden_dim = x.size(2)
        
        # Create a dummy tensor and concatenate it to the input
        dummy_tensor = torch.ones(batch_size, 1, hidden_dim, device=x.device)
        x = torch.cat([dummy_tensor, x], dim=1)  # [batch_size, 1 + n_nodes, hidden_dim]

        mask = torch.zeros(batch_size*self.num_heads, n_nodes+1, n_nodes+1)  # [batch_size * num_heads, 1 + n_nodes, 1 + n_nodes]

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
          
class CustomGraphTransformerPolicy(nn.Module):
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
        mask_tensor = None,
        ) -> None:
        super().__init__()
        
        device = get_device(device)
        self.device = device

        self.mask = torch.tensor(mask_tensor)
    
        self.input_dim = input_dim
        
        self.embedder = nn.Linear(input_dim, feature_dim).to(device) 
        self.transformer_pi = TransformerWithMaskedAttention(feature_dim, num_heads=num_heads, num_layers=num_layers)
        self.transformer_v = TransformerWithMaskedAttention_cls(feature_dim, num_heads=4, num_layers=4, dummy_size=feature_dim)
                
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
        policy_net.append(nn.Linear(last_layer_dim_pi, 1))   
        policy_net.append(self.flatten)
        
        # Iterate through the value layers and build the value net
        value_net.append(self.flatten)
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim
        

        # Save dim, used to create the distributions
        self.latent_dim_pi = 3860
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
        features = self.process_obs(features)
        transformer_inputs = self.embedder(features)
        transformer_output = self.transformer_pi(transformer_inputs, self.mask)
        pi_output = self.policy_net(transformer_output)
        return pi_output , self.output_mask
        
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features = self.process_obs(features)
        transformer_inputs = self.embedder(features) 
        transformer_output = self.transformer_v(transformer_inputs)
        vf_output = self.value_net(transformer_output)
        return vf_output
    
    
    def process_obs(self, observations):
        if isinstance(observations, list):
            observations = torch.stack(observations, dim=0)
        observations = torch.tensor(observations)
                
        self.output_mask = observations[:,:,4]
        
        return observations.to(self.device)
    
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

class TransformerCategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self: SelfTransformerCategoricalDistribution, action_logits: th.Tensor) -> SelfTransformerCategoricalDistribution:
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.argmax(self.distribution.probs, dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


def custom_make_proba_distribution(
    action_space: spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    return TransformerCategoricalDistribution(int(action_space.n), **dist_kwargs)


class CustomGraphActorCriticPolicy(ActorCriticPolicy):
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
        self.mask_tensor = features_extractor_kwargs["mask_tensor"]
        self.input_dim = features_extractor_kwargs["input_dim"]
        self.num_heads = features_extractor_kwargs["num_heads"]
        self.num_layers = features_extractor_kwargs["num_layers"]
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
        self.action_dist = custom_make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=self.dist_kwargs)
        self._build(lr_schedule)
        
           
    def _build_mlp_extractor(self) -> None:
         self.mlp_extractor = CustomGraphTransformerPolicy(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            num_heads = self.num_heads,
            num_layers = self.num_layers,
            input_dim = self.input_dim,
            mask_tensor = self.mask_tensor)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            (latent_pi, mask_output), latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi, mask_output = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, mask_output)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob


    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]


    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, mask_output: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions_score = latent_pi * mask_output

        if isinstance(self.action_dist, TransformerCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions_score)
        else:   
            raise ValueError("Invalid action distribution")

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            (latent_pi, mask_output), latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi, mask_output = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi, mask_output)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs)
        latent_pi, mask_output = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi,mask_output)
    