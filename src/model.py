from __future__ import annotations

import numpy as np
import torch
import functorch
from torch import nn


class HyperNetwork(nn.Module):
    def __init__(self, hypnet: nn.Module, mainnet1: nn.Module, mainnet2: nn.Module):
        super().__init__()

        m_func1, m_params1 = functorch.make_functional(mainnet1)
        self._sp_shapes1 = [param.shape for param in m_params1]
        self._sp_offsets1 = [0] + np.cumsum([param.numel() for param in m_params1]).tolist()
        self._mainnet_batched_func1 = functorch.vmap(m_func1)

        m_func2, m_params2 = functorch.make_functional(mainnet2)
        self._sp_shapes2 = [param.shape for param in m_params2]
        self._sp_offsets2 = [0] + np.cumsum([param.numel() for param in m_params2]).tolist()
        self._mainnet_batched_func2 = functorch.vmap(m_func2)

        self.hidden_size = mainnet1.hidden_size
        self._hypnet = hypnet

    @staticmethod
    def _reshape_params(flat_params: torch.Tensor, shapes: list[torch.Size], offsets: list[int]) -> list[torch.Tensor]:
        params_list = []
        for index, shape in enumerate(shapes):
            start = offsets[index]
            end = offsets[index + 1]
            params_list.append(flat_params[..., start:end].reshape(-1, *shape))
        return params_list

    def forward(self, hyp_input: torch.Tensor, main_input: torch.Tensor, h0: torch.Tensor, device: torch.device):
        params = self._hypnet(hyp_input)

        split_idx = self._sp_offsets1[-1]
        params1 = params[:, :split_idx]
        params2 = params[:, split_idx:]

        params_list1 = self._reshape_params(params1, self._sp_shapes1, self._sp_offsets1)
        params_list2 = self._reshape_params(params2, self._sp_shapes2, self._sp_offsets2)

        if main_input.dim() != 3 or h0.dim() != 3:
            raise ValueError("main_input and h0 must be rank-3 tensors.")

        output1 = torch.empty(
            main_input.shape[0],
            main_input.shape[1],
            self.hidden_size,
            device=device,
            dtype=main_input.dtype,
        )
        h_next = h0[0]

        for step in range(main_input.shape[1]):
            h_next = self._mainnet_batched_func1(params_list1, main_input[:, step], h_next)
            output1[:, step, :] = h_next

        h = h_next.unsqueeze(0)
        output2 = self._mainnet_batched_func2(params_list2, output1)
        return output1, h, params1, output2, params2


class EndtoEndModel(nn.Module):
    def __init__(self, hypernet: HyperNetwork, lambda_act: float, lambda_bldi: float, device: torch.device):
        super().__init__()
        self.hypernet = hypernet
        self.hidden_size = hypernet.hidden_size
        self.lambda_act = lambda_act
        self.lambda_bldi = lambda_bldi
        self.device = device
        self.sigmoid = nn.Sigmoid()

    def _run_core(self, bld: torch.Tensor, stimulus: torch.Tensor):
        stimulus = torch.cat(
            [torch.zeros((stimulus.shape[0], 1, stimulus.shape[2]), device=self.device, dtype=stimulus.dtype), stimulus],
            dim=1,
        )
        rnn_input = stimulus[:, :-1]
        h0 = torch.zeros(1, bld.shape[0], self.hidden_size, device=self.device, dtype=stimulus.dtype)
        _, h, params1, output, params2 = self.hypernet(bld, rnn_input, h0, self.device)

        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)

        output_act = self.sigmoid(output[:, :, :3])
        output_bldi = output[:, :, 3:]
        return output_act, output_bldi, h, params1, params2

    def forward(self, bld: torch.Tensor, stimulus: torch.Tensor):
        return self._run_core(bld, stimulus)

    def open_loop(self, bld: torch.Tensor, stimulus: torch.Tensor):
        return self._run_core(bld, stimulus)

    def calc_loss(
        self,
        action_prediction: torch.Tensor,
        action_true: torch.Tensor,
        bldi_prediction: torch.Tensor,
        bldi_true: torch.Tensor,
    ):
        action_loss = self.lambda_act * nn.BCELoss()(action_prediction, action_true)
        bldi_loss = self.lambda_bldi * nn.MSELoss()(bldi_prediction, bldi_true)
        loss = action_loss + bldi_loss
        return loss, {"action_loss": action_loss, "bldi_loss": bldi_loss}


def build_model(config, device: torch.device) -> EndtoEndModel:
    mainnet1 = nn.RNN(config.stimulus_dim, config.hidden_size)
    mainnet2 = nn.Linear(config.hidden_size, config.action_dim + config.bldi_dim)

    _, mainnet_params1 = functorch.make_functional(mainnet1)
    n_mainnet_params1 = sum(param.numel() for param in mainnet_params1)

    _, mainnet_params2 = functorch.make_functional(mainnet2)
    n_mainnet_params2 = sum(param.numel() for param in mainnet_params2)
    n_mainnet_params = n_mainnet_params1 + n_mainnet_params2

    hypnet = nn.Sequential(
        nn.Dropout(config.dropout),
        nn.Linear(config.bld_dim, config.hypnet_mid1),
        nn.Mish(),
        nn.Linear(config.hypnet_mid1, config.hypnet_mid2),
        nn.Mish(),
        nn.Linear(config.hypnet_mid2, n_mainnet_params),
    ).to(device)

    hypernet = HyperNetwork(
        hypnet,
        mainnet1=nn.RNNCell(mainnet1.input_size, mainnet1.hidden_size),
        mainnet2=mainnet2,
    ).to(device)
    return EndtoEndModel(
        hypernet=hypernet,
        lambda_act=config.lambda_act,
        lambda_bldi=config.lambda_bldi,
        device=device,
    ).to(device)
