import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

class DQNCNN(nn.Module):
    """A Convolutional Neural Network for the lane DQN agent"""
    def __init__(
        self,
        n_in_channels: int,
        image_dim,
        state_size,
        num_actions,
        hidden_dim: int=128,
        activation=nn.ReLU,
    ):
        super(DQNCNN, self).__init__()

        # Define the input network.
        self.im_feature = nn.Sequential(
            nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
            ),
            activation(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
            ),
            activation(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            activation(),
            Flatten(),
        )

        # Create a dummy tensor of batch size 1, with the appropriate
        # number of channels, and image size (the '*' unpacks the
        # dimensions of the image and passes the row and column to the
        # torch.zeros function). Get the size of the output of the
        # input network by passing this dummy tensor through the network,
        # getting the result, and then obtaining the size of this output.
        dummy = torch.zeros((1, n_in_channels, *image_dim))
        im_feature_size = self.im_feature(dummy).data.cpu().numpy().size

        # A ModuleList is like a Python list, designed to store any
        # desired number of nn.Module's. Useful when designing a neural
        # network whose number of layers is passed as input.
        self.q_outs = nn.ModuleList()
        for action_num in num_actions:
            q_out = nn.Sequential(
                nn.Linear(
                    in_features=(im_feature_size + state_size),
                    out_features=hidden_dim
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=hidden_dim,
                    out_features=action_num
                )
            )
            self.q_outs.append(q_out)

        self.init()

    def init(self):
        # The function constant_ filles the input Tensor with the
        # input value passed to the function.
        for q_out in self.q_outs:
            # Initialize the bias' of the last linear layer in the
            # output to be all zeroes.
            nn.init.constant_(q_out[-1].bias.data, 0.0)

    def forward(self, image, state_size):
        im_feature = self.im_feature(image)
        x = torch.cat([im_feature, state_size], dim=-1)
        x = [e(x) for e in self.q_outs]
        return x

class DQNFC(nn.Module):
    def __init__(
        self,
        num_actions,
        state_size,
        hidden_dim: int=256,
        activation=nn.ReLU,
    ):
        super(DQNFC, self).__init__()

        self.q_outs = nn.ModuleList()
        for action_num in num_actions:
            q_out = nn.Sequential(
                nn.Linear(
                    in_features=state_size,
                    out_features=hidden_dim
                ),
                activation(),
                nn.Linear(
                    in_features=hidden_dim,
                    out_features=(hidden_dim // 2)
                ),
                activation(),
                nn.Linear(
                    in_features=(hidden_dim // 2),
                    out_features=(hidden_dim // 4)
                ),
                activation(),
                nn.Linear(
                    in_features=(hidden_dim // 4),
                    out_features=action_num
                ),
            )
            self.q_outs.append(q_out)
        
        self.init()

    def init(self):
        for q_out in self.q_outs:
            nn.init.constant_(q_out[-1].bias.data, 0.0)

    def forward(self, state, training=False):
        low_dim_state = state["low_dim_states"]
        
        if len(low_dim_state.shape) == 1:
            low_dim_state = torch.unsqueeze(low_dim_state, 0)
            unsqueezed = True
        else:
            unsqueezed = False

        x = low_dim_state
        x = [e(x) for e in self.q_outs]

        if unsqueezed:
            x = [torch.squeeze(e, 0) for e in x]

        if training:
            aux_losses = {}
            return x, aux_losses
        else:
            return x

class DQNWithSocialEncoder(nn.Module):
    def __init__(
        self,
        num_actions,
        state_size,
        hidden_dim: int=256,
        activation=nn.ReLU,
        social_feature_encoder_class=None,
        social_feature_encoder_params=None,
    ):
        super(DQNWithSocialEncoder, self).__init__()

        # Create the social vehicle state encoder if there desired.
        self.social_feature_encoder = (
            social_feature_encoder_class(**social_feature_encoder_params)
            if social_feature_encoder_class else None
        )

        self.state_size = state_size

        # Define the Q value neural networks for each action.
        self.q_outs = nn.ModuleList()
        for action_num in num_actions:
            q_out = nn.Sequential(
                nn.Linear(
                    in_features=state_size,
                    out_features=hidden_dim
                ),
                activation(),
                nn.Linear(
                    in_features=hidden_dim,
                    out_features=(hidden_dim // 2)
                ),
                activation(),
                nn.Linear(
                    in_features=(hidden_dim // 2),
                    out_features=(hidden_dim // 4)
                ),
                activation(),
                nn.Linear(
                    in_features=(hidden_dim // 4),
                    out_features=action_num
                ),
            )
            self.q_outs.append(q_out)

        # Set the biases of each last Linear layer in each Q value neural
        # network to zero.
        self.init()

    def init(self):
        for q_out in self.q_outs:
            nn.init.constant_(q_out[-1].bias.data, 0.0)

    def forward(self, state, training=False):
        low_dim_state = state["low_dim_states"]
        social_vehicles_state = state["social_vehicles"]

        aux_losses = {}
        social_feature = []

        if self.social_feature_encoder is not None:
            # Get the encoding of the social vehicles state (and the losses).
            social_feature, social_encoder_aux_losses = self.social_feature_encoder(
                social_vehicles_state, training
            )
            # Put the social_encoder_aux_losses in the aux_losses.
            aux_losses.update(social_encoder_aux_losses)
        else:
            # There is no social vehicles state encoder. Simply present the
            # social vehicle state as is for each social vehicle.
            social_feature = [e.reshape(1, -1) for e in social_vehicles_state]

        social_feature = torch.cat(social_feature, 0) if len(social_feature) > 0 else []

        x = (
            torch.cat([low_dim_state, social_feature], -1)
            if len(social_feature) > 0 else low_dim_state
        )
        # Get the Q value of each action based on each action Q value's network.
        x = [e(x) for e in self.q_outs]

        if training:
            return x, aux_losses
        else:
            return x
