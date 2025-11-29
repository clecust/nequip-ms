from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import torch
from torch import Tensor

from e3nn import o3,nn
from e3nn.util.jit import compile_mode
from ase.units import Bohr

from .dftd2_params import get_dftd2_params
from .dftd3_xc_params import get_dftd3_default_params

# conversion factors used in grimme d3 code

d3_autoang = 0.52917726  # for converting distance from bohr to angstrom
d3_autoev = 27.21138505  # for converting a.u. to eV

d3_k1 = 16.000
d3_k2 = 4 / 3
d3_k3 = -4.000
d3_maxc = 5  # maximum number of coordination complexes

@compile_mode("script")
class Poly_Smoothing(torch.nn.Module):

    def __init__(
        self,
        cutoff: float = 10.0,
    ):
        super().__init__()
        self.cutoff = cutoff
    
    def forward(
        self,
        r: Tensor,
    ) -> Tensor:
        cuton = self.cutoff - 1
        x = (self.cutoff - r) / (self.cutoff - cuton)
        x2 = x**2
        x3 = x2 * x
        x4 = x3 * x
        x5 = x4 * x
        return torch.where(
            r <= cuton,
            torch.ones_like(x),
            torch.where(r >= self.cutoff, torch.zeros_like(x), 6 * x5 - 15 * x4 + 10 * x3),
        )


@compile_mode("script")
class DFTD2_Calc(torch.nn.Module):

    def __init__(
        self,
        damping: str = "zero",
        xc: str = "pbe",
        old: bool = True,
        cutoff: float = 95.0 * Bohr,
        bidirectional: bool = True,
        dtype:torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.cutoff = cutoff
        self.bidirectional = bidirectional

        self.params = get_dftd3_default_params(damping, xc, old=old)
        self.alp6 = self.params["alp"]
        self.s6 = self.params["s6"]
        self.rs6 = self.params["rs6"]

        self.damping = damping
        r0ab, c6ab = get_dftd2_params()
        r0ab = r0ab.to(dtype)
        c6ab = c6ab.to(dtype)
        # atom pair coefficient (87, 87)
        self.register_buffer("c6ab", c6ab)
        # atom pair distance (95, 95)
        self.register_buffer("r0ab", r0ab)
        self.poly_smoothing = Poly_Smoothing(cutoff)
        # training weights
        prefix_6 = torch.zeros(3, dtype=self.dtype)  # 调控势井的深度和水平偏移
        self.prefix_6 = torch.nn.Parameter(prefix_6)
        prefix_c6ab = torch.zeros((87, 87), dtype=self.dtype)  # 调控势井的深度
        self.prefix_c6ab = torch.nn.Parameter(prefix_c6ab)
        # prefix transform
        self.register_buffer("d3_autoang", torch.tensor(d3_autoang, dtype=self.dtype))
        self.register_buffer("d3_autoev", torch.tensor(d3_autoev, dtype=self.dtype))
        self.act = torch.nn.Tanh()
        # self.act = torch.nn.ReLU()

    def forward(
        self,
        r: Tensor,
        edge_index: Tensor,
        Z: Tensor,
        batch: Tensor,
    ) -> Tensor:
        r = r / self.d3_autoang  # angstrom -> bohr
        r2 = r**2
        r6 = r2**3
        idx_i = edge_index[0]
        idx_j = edge_index[1]

        # compute all necessary quantities
        Zi = Z[idx_i]  # (n_edges,)
        Zj = Z[idx_j]

        if self.damping != "zero":
            raise ValueError(
                f"Only zero-damping can be used with the D2 dispersion correction method!"
            )
        c6ab = self.c6ab #  * (1.0 + 0.5 * (self.act(self.prefix_c6ab)))
        c6 = c6ab[Zi, Zj].unsqueeze(-1)  # (n_edges,)
        factors = 1.0  + 0.5 * self.act(self.prefix_6)
        alp6 = self.alp6 * factors[0] 
        s6 = self.s6 * factors[1]
        rs6 = self.rs6 * factors[2]  
        damp6 = 1.0 / (1.0 + torch.exp(-alp6 * (r / (rs6 * self.r0ab[Zi, Zj]).unsqueeze(-1) - 1.0)))
        e6 = damp6 / r6
        e6 = -0.5 * s6 * c6 * e6  # (n_edges,)

        # e6 *= self.poly_smoothing(r)
        e6 = e6 * (1 + 2 * (r / self.cutoff) ** 3 - 3 * (r / self.cutoff) ** 2)

        g = e6.new_zeros((int(batch[-1]) + 1,1))

        if not self.bidirectional:

            g.scatter_add_(0, batch[idx_i].unsqueeze(-1), e6)
            g.scatter_add_(0, batch[idx_j].unsqueeze(-1), e6)
        else:
            g.scatter_add_(0, batch[idx_i].unsqueeze(-1), e6)

        E_disp = self.d3_autoev * g
        return E_disp  # E_disp (n_graphs,): Energy in eV unit


@compile_mode("script")
class DFTD2_Calc_v2(torch.nn.Module):

    def __init__(
        self,
        damping: str = "zero",
        xc: str = "pbe",
        old: bool = True,
        cutoff: float = 95.0 * Bohr,
        bidirectional: bool = True,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.cutoff = cutoff
        self.bidirectional = bidirectional

        self.params = get_dftd3_default_params(damping, xc, old=old)
        self.alp6 = self.params["alp"]
        self.s6 = self.params["s6"]
        self.rs6 = self.params["rs6"]

        self.damping = damping
        r0ab, c6ab = get_dftd2_params()
        r0ab = r0ab.to(dtype)
        c6ab = c6ab.to(dtype)
        # atom pair coefficient (87, 87)
        self.register_buffer("c6ab", c6ab)
        # atom pair distance (95, 95)
        self.register_buffer("r0ab", r0ab)
        self.poly_smoothing = Poly_Smoothing(cutoff)
        # training weights
        prefix_6 = torch.zeros(3, dtype=self.dtype)  # 调控势井的深度和水平偏移
        self.prefix_6 = torch.nn.Parameter(prefix_6)
        prefix_c6ab = torch.zeros((87, 87), dtype=self.dtype)  # 调控势井的深度
        self.prefix_c6ab = torch.nn.Parameter(prefix_c6ab)
        # prefix transform
        self.register_buffer("d3_autoang", torch.tensor(d3_autoang, dtype=self.dtype))
        self.register_buffer("d3_autoev", torch.tensor(d3_autoev, dtype=self.dtype))
        self.act = torch.nn.Tanh()

    def forward(
        self,
        r: Tensor,
        edge_index: Tensor,
        Z: Tensor,
        batch: Tensor,
    ) -> Tensor:

        idx_i = edge_index[0]
        idx_j = edge_index[1]

        # compute all necessary quantities
        Zi = Z[idx_i]  # (n_edges,)
        Zj = Z[idx_j]

        # 
        r0ab_zij = self.r0ab[Zi, Zj].unsqueeze(-1)

        r = r / self.d3_autoang  # angstrom -> bohr
        r = torch.where(r < r0ab_zij, r0ab_zij, r)
        r2 = r**2
        r6 = r2**3
        if self.damping != "zero":
            raise ValueError(
                f"Only zero-damping can be used with the D2 dispersion correction method!"
            )
        c6ab = self.c6ab * (1.0 + 0.9 * self.act(self.prefix_c6ab))
        c6 = c6ab[Zi, Zj].unsqueeze(-1)  # (n_edges,)
        factors = 1.0 + 0.9 * self.act(self.prefix_6)
        alp6 = self.alp6 * factors[0]
        s6 = self.s6 * factors[1]
        rs6 = self.rs6 * factors[2]
        damp6 = 1.0 / (
            1.0 + torch.exp(-alp6 * (r / (rs6 * r0ab_zij) - 1.0))
        )
        e6 = damp6 / r6
        e6 = -0.5 * s6 * c6 * e6  # (n_edges,)

        # e6 *= self.poly_smoothing(r)
        e6 = e6 * (1 + 2 * (r / self.cutoff) ** 3 - 3 * (r / self.cutoff) ** 2)

        g = e6.new_zeros((int(batch[-1]) + 1, 1))

        if not self.bidirectional:

            g.scatter_add_(0, batch[idx_i].unsqueeze(-1), e6)
            g.scatter_add_(0, batch[idx_j].unsqueeze(-1), e6)
        else:
            g.scatter_add_(0, batch[idx_i].unsqueeze(-1), e6)

        E_disp = self.d3_autoev * g
        return E_disp  # E_disp (n_graphs,): Energy in eV unit


@compile_mode("script")
class DFTD2_Calc_C6(torch.nn.Module):
    # 简化，只保留C6
    def __init__(
        self,
        damping: str = "zero",
        xc: str = "pbe",
        old: bool = True,
        cutoff: float = 95.0 * Bohr,
        bidirectional: bool = True,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.cutoff = cutoff
        self.bidirectional = bidirectional

        self.params = get_dftd3_default_params(damping, xc, old=old)
        self.alp6 = self.params["alp"]
        self.s6 = self.params["s6"]
        self.rs6 = self.params["rs6"]

        self.damping = damping
        r0ab, c6ab = get_dftd2_params()
        r0ab = r0ab.to(dtype)
        c6ab = c6ab.to(dtype)
        # atom pair coefficient (87, 87)
        self.register_buffer("c6ab", c6ab)
        # atom pair distance (95, 95)
        self.register_buffer("r0ab", r0ab)
        self.poly_smoothing = Poly_Smoothing(cutoff)
        # training weights
        prefix_6 = torch.zeros(3, dtype=self.dtype)  # 调控势井的深度和水平偏移
        self.prefix_6 = torch.nn.Parameter(prefix_6)
        prefix_c6ab = torch.zeros((87, 87), dtype=self.dtype)  # 调控势井的深度
        self.prefix_c6ab = torch.nn.Parameter(prefix_c6ab)
        # prefix transform
        self.register_buffer("d3_autoang", torch.tensor(d3_autoang, dtype=self.dtype))
        self.register_buffer("d3_autoev", torch.tensor(d3_autoev, dtype=self.dtype))
        self.act = torch.nn.Tanh()

    def forward(
        self,
        r: Tensor,
        edge_index: Tensor,
        Z: Tensor,
        batch: Tensor,
    ) -> Tensor:

        idx_i = edge_index[0]
        idx_j = edge_index[1]

        # compute all necessary quantities
        Zi = Z[idx_i]  # (n_edges,)
        Zj = Z[idx_j]

        r = r / self.d3_autoang  # angstrom -> bohr
        # 只保留C6的吸引势
        r2 = (r + self.r0ab[Zi, Zj].unsqueeze(-1) ) ** 2
        r6 = r2**3

        c6ab = self.c6ab   * ( 1 + 0.99 * self.act(self.prefix_c6ab))
        c6 = c6ab[Zi, Zj].unsqueeze(-1)  # (n_edges,)
        s6 = self.s6  * (1 + 0.99 * self.act(self.prefix_6[1]))
        e6 = -0.5 * s6 * c6  / r6   # (n_edges,)

        # e6 *= self.poly_smoothing(r)
        e6 = e6 * (1 + 2 * (r / self.cutoff) ** 3 - 3 * (r / self.cutoff) ** 2)

        g = e6.new_zeros((int(batch[-1]) + 1, 1))

        if not self.bidirectional:
            g.scatter_add_(0, batch[idx_i].unsqueeze(-1), e6)
            g.scatter_add_(0, batch[idx_j].unsqueeze(-1), e6)
        else:
            g.scatter_add_(0, batch[idx_i].unsqueeze(-1), e6)

        E_disp = self.d3_autoev * g
        return E_disp  # E_disp (n_graphs,): Energy in eV unit


@compile_mode("script")
class DFTD2_Calc_r6(torch.nn.Module):
    # 简化，只保留C6
    def __init__(
        self,
        cutoff: float = 8.0,
        bidirectional: bool = True,
        dtype: torch.dtype = torch.float32,
        num_elements: int = 118,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.cutoff = cutoff
        self.bidirectional = bidirectional
        self.act = torch.nn.ReLU()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(num_elements, 16, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 1, bias=False),
            torch.nn.SiLU(),
        )

    def forward(
        self,
        r: Tensor,
        edge_index: Tensor,
        node_attr: Tensor,
        batch: Tensor,
    ) -> Tensor:
        x = 1.0 / (r**2 + 1.0)
        x = x * (1 + 2 * (r / self.cutoff) ** 3 - 3 * (r / self.cutoff) ** 2)

        idx_i = edge_index[0]
        idx_j = edge_index[1]
        edge_attr = node_attr[idx_i] + node_attr[idx_j]
        edge_attr = self.mlp(edge_attr)
        e6 = edge_attr * x

        g = e6.new_zeros((int(batch[-1]) + 1, 1))

        if not self.bidirectional:
            g.scatter_add_(0, batch[idx_i].unsqueeze(-1), e6)
            g.scatter_add_(0, batch[idx_j].unsqueeze(-1), e6)
        else:
            g.scatter_add_(0, batch[idx_i].unsqueeze(-1), e6)

        return g  # E_disp (n_graphs,): Energy in eV unit


if __name__ == "__main__":
    # test

    TYPE = "CC"

    # 初始距离（埃）
    initial_distance = 0.5
    # 最终距离（埃）
    final_distance = 8.0
    # 距离步长（埃）
    distance_step = 0.1

    # 原子间的距离数组
    distances = np.arange(initial_distance, final_distance, distance_step)

    # test
    r = torch.tensor(distances, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    Z = torch.tensor([6, 6], dtype=torch.long)
    batch = torch.tensor([0, 0], dtype=torch.long)
    dftd2 = DFTD2_Calc(
        damping="zero",
        xc="b3-lyp",
        old=True,
        cutoff=12.0 / d3_autoang,
        bidirectional=True,
    )
    energy2 = []
    # forces1 = []
    for rs in r:
        rs = rs.repeat(2)
        E_disp = dftd2(rs, edge_index, Z, batch)
        # f = torch.autograd.grad(E_disp, rs)[0]
        energy2.append(E_disp.item())

        # forces2.append(forces)
    energy2 = np.array(energy2)
    print(E_disp)

    # true
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    from ase import Atoms
    atoms = []
    for distance in distances:
        # 创建两个氢原子的体系
        atoms.append(Atoms(TYPE, positions=[[0, 0, 0], [0, 0, distance]]))

    energy1 = []
    forces1 = []
    for atom in atoms:
        atom.set_calculator(
            TorchDFTD3Calculator(
                atoms=atom,
                device="cuda",
                damping="zero",
                xc="b3-lyp",
                old=True,
                cutoff=12.0 / d3_autoang,
                bidirectional=True,
            )
        )
        energy1.append(atom.get_potential_energy())
        forces1.append(atom.get_forces())
        atom.calc = None
    energy1 = np.array(energy1)
    forces1 = np.array(forces1)

    import scienceplots
    import matplotlib.pyplot as plt

    with plt.style.context("science"):
        # 密度分布图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), dpi=250)
        ax1.plot(distances, energy1, label="true")
        ax1.plot(distances, energy2, "--", label="test")
        ax1.set_xlabel("distance")
        ax1.set_ylabel("energy")
        ax1.set_title(TYPE)
        ax1.legend()

        # ax2.plot(distances, forces1[:, 1, 2], label="forces_z")
        # ax2.plot(distances, forces2[:, 1, 2], "--", label="forces_z")
        # ax2.plot(distances, forces1[:, 1, 1], label="forces_y")
        # ax2.plot(distances, forces2[:, 1, 1], "--", label="forces_y")
        # ax2.plot(distances, forces1[:, 1, 0], label="forces_x")
        # ax2.plot(distances, forces2[:, 1, 0], "--", label="forces_x")
        # ax2.set_xlabel("distance")
        # ax2.set_ylabel("forces")
        # ax2.set_title(TYPE)
        # ax2.legend()

        plt.show()
    print()
