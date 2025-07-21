from model.mamba_block import *


class MambaSimple(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """

    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        n_rnn_layer=3,
        bidirectional=False,
        dropout=0,
    ):
        super(MambaSimple, self).__init__()
        args = ModelArgs(
            d_model=n_hidden,
            n_layers=n_rnn_layer,
            input_size=n_input,
        )
        self.activation = torch.nn.SiLU()

        self.mamba = nn.ModuleList(
            [ResidualJointInitBlock(args) for _ in range(args.n_layers)]
        )
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = torch.nn.Dropout(dropout)
        self.joint_layer = nn.Sequential(
            nn.Linear(24 * 3, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
        )

    def reset_state(self):
        for layer in self.mamba:
            layer.reset_state()

    def forward(self, x, init_joint):
        x = self.linear1(x)
        joint_emb = self.joint_layer(init_joint.reshape(-1, 24 * 3))
        for layer in self.mamba:
            x[:, 0] += joint_emb
            x = layer(x)

        return self.linear2(x)

    def step(self, x, init_joint=None):
        x = self.linear1(x)

        if init_joint is not None:
            joint_emb = self.joint_layer(init_joint.reshape(-1, 24 * 3))

        for layer in self.mamba:
            if init_joint is not None:
                x += joint_emb
            x = layer.forward_online(x)

        return self.linear2(x)


class RUMamba(torch.nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """

    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        exp_scale=1,
        n_layers=5,
        dropout=0.1,
        n_reduced=10,
    ):
        super(RUMamba, self).__init__()
        args = ModelArgs(
            d_model=n_hidden,
            n_layers=n_layers,
            input_size=n_input,
        )

        self.mamba = nn.ModuleList(
            [ResidualJointInitBlock(args) for _ in range(args.n_layers)]
        )

        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linearR = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_reduced * 9),
        )
        self.joint_layer = nn.Sequential(
            nn.Linear(24 * 3, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
        )
        self.linearU = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_reduced * 3),
            nn.Sigmoid(),
        )
        self.exp_scale = exp_scale

    def forward(self, x, init_joint):

        x = self.linear1(x)
        joint_emb = self.joint_layer(init_joint.reshape(-1, 24 * 3))

        for layer in self.mamba:
            x[:, 0] += joint_emb
            x = layer(x)

        posteriorR = self.linearR(x)
        posteriorU = self.linearU(x)

        posteriorU_exp = torch.exp(posteriorU * self.exp_scale)
        posteriorRU_exp = torch.cat((posteriorR, posteriorU_exp), dim=-1)
        posteriorRU = torch.cat((posteriorR, posteriorU), dim=-1)
        return posteriorRU_exp, posteriorRU

    def reset_state(self):
        for layer in self.mamba:
            layer.reset_state()

    def step(self, x, init_joint=None):
        x = self.linear1(x)
        if init_joint is not None:
            joint_emb = self.joint_layer(init_joint.reshape(-1, 24 * 3))

        for layer in self.mamba:
            if init_joint is not None:
                x += joint_emb
            x = layer.forward_online(x)

        posteriorR = self.linearR(x)
        posteriorU = self.linearU(x)
        posteriorU_exp = torch.exp(posteriorU * self.exp_scale)

        posteriorRU_exp = torch.cat((posteriorR, posteriorU_exp), dim=-1)
        posteriorRU = torch.cat((posteriorR, posteriorU), dim=-1)

        return posteriorRU_exp, posteriorRU
