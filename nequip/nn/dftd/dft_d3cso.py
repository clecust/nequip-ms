from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


from e3nn.util.jit import compile_mode
import numpy as np

try:
    from .dftd2_params import r0, c6, d3_autoang, c6conv
except:
    from dftd2_params import r0, c6, d3_autoang, c6conv


@compile_mode("script")
class D3CSO_Calculator_edge_forces(nn.Module):
    def __init__(
        self,
        cutoff: float = 10.0,
        device: float = "cpu",
        xc: str = "PBE",
        bidirectional: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "bidirectional", torch.tensor(bidirectional, dtype=torch.bool)
        )
        # constants for dispersion correction
        self.c6_emb = torch.nn.Embedding.from_pretrained(
            torch.tensor(c6 * c6conv, device=device, dtype=torch.get_default_dtype())
            .unsqueeze(1)
            .clone()
            .detach(),
            freeze=True,
        ).requires_grad_(False)
        self.r0_emb = torch.nn.Embedding.from_pretrained(
            torch.tensor(
                2.0 * r0 / d3_autoang,
                device=device,
                dtype=torch.get_default_dtype(),
            )
            .unsqueeze(1)
            .clone()
            .detach(),
            freeze=True,
        ).requires_grad_(False)
        self.d3_autoang = 0.52917726  # for converting distance from bohr to angstrom
        self.d3_autoev = 27.21138505  # for converting a.u. to eV
        self.cutoff = cutoff / self.d3_autoang
        self.is_train: bool = False

        s6 = 0.73
        a2 = 2.5
        a4 = 6.25
        xc = xc.upper()

        if xc == 'TRAIN':
            self.is_train = True
            a1 = -s6
        else:
            if xc == "BLYP":
                a1 = 1.28
            elif xc == "BP86":
                a1 = 1.01
            elif xc == "PBE":
                a1 = 0.24
            elif xc == "TPSS":
                a1 = 0.72
            elif xc == "B3LYP":
                a1 = 0.86
            elif xc == "PBE0":
                a1 = 0.20
            elif xc == "PW6B95":
                a1 = -0.15
            elif xc == "B2PLYP":
                a1 = 0.24
            elif xc == "BAMBOO":
                a1 = 0.82
                s6 = 0.85
                a4 = 4.5
            elif xc == "ZERO":
                a1 = -0.73
            else:
                raise ValueError(f"[ERROR] Unexpected value xc={xc}")
        self.register_buffer("a1", torch.tensor([a1], dtype=torch.get_default_dtype()))
        self.register_buffer("a2", torch.tensor([a2], dtype=torch.get_default_dtype()))
        self.register_buffer("a4", torch.tensor([a4], dtype=torch.get_default_dtype()))
        self.register_buffer("s6", torch.tensor([s6], dtype=torch.get_default_dtype()))
        self.prefix = torch.nn.Parameter(
            torch.tensor([s6], dtype=torch.get_default_dtype())
        )

    def forward(
        self,
        dij: torch.Tensor,  # vec
        rij: torch.Tensor,  # length
        edge_index: torch.Tensor,
        Z: torch.Tensor,
        compute_virials: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Compute D3-CSO dispersion energy and pairwise dispersion forces from C6 and r0 parameters.
        """
        rij = rij / self.d3_autoang  # angstrom -> bohr
        dij = dij / self.d3_autoang  # angstrom -> bohr
        c6 = self.c6_emb(Z)
        r0 = self.r0_emb(Z)
        row = edge_index[0]
        col = edge_index[1]
        c6ij = torch.sqrt(c6[row] * c6[col])
        r0ij = 0.5 * (r0[row] + r0[col])
        node_num = Z.shape[0]
        # a1 = self.a1 * (torch.tanh(self.prefix)+1.0)

        # self.prefix+self.s6 = 0.0 则色散作用＝0.0; 避免不起作用，但不设置上限
        # a1 = self.a1 * torch.relu(self.prefix + self.s6)
        # a1 = torch.relu(self.a1 + self.prefix) - self.s6
        if self.is_train:
            a1 = torch.relu(self.prefix) - self.s6
        else:
            a1 = self.a1 

        # D3-CSO dispersion correction
        edisp = (
            -c6ij
            / (rij**6 + self.a4**6)
            * (self.s6 + a1 / (1.0 + torch.exp(rij - self.a2 * r0ij)))
        )
        fdisp = 6 * c6ij * rij**5 / ((rij**6 + (self.a4) ** 6) ** 2) * (
            self.s6 + a1 / (1.0 + torch.exp(rij - self.a2 * r0ij))
        ) + c6ij / (rij**6 + (self.a4) ** 6) * (
            a1
            * torch.exp(rij - self.a2 * r0ij)
            / ((1.0 + torch.exp(rij - self.a2 * r0ij)) ** 2)
        )

        # # 应用截断函数
        if self.cutoff is not None:
            ### 采用bamboo框架使用的, 水平平移
            edisp += self.s6 * c6ij / (self.cutoff**6 + self.a4**6)
            e6 = edisp
            f6 = fdisp
        else:
            e6 = edisp
            f6 = fdisp
        force_contribution = -dij * (f6 / (rij + 1e-15))

        # 初始化力张量
        force = torch.zeros((node_num, 3), device=dij.device, dtype=dij.dtype)
        # 聚合节点能量
        node_g = e6.new_zeros((node_num, 1))
        node_g.index_add_(0, col, e6)
        force.index_add_(0, col, force_contribution)

        if not self.bidirectional:
            # 单向图 index_add_  scatter_add_
            node_g *= 2.0
            force.index_add_(0, col, -force_contribution)

        # 2.0　：　每个原子对的能量，有两个相反的原子力；（与ase的保持一致）
        # virial为负值，能与自动微分的对上
        return (
            node_g.squeeze(-1) * self.d3_autoev,
            -2.0*force_contribution * self.d3_autoev / self.d3_autoang,
        )  # node, edge_forces

    def __repr__(self):
        # 调用父类的__repr__方法作为基础
        base_repr = super().__repr__()
        # 提取类名部分
        class_name = base_repr.split("(")[0]
        # 添加自定义参数信息
        params = [
            f"a1={self.a1.item():.4f}",
            f"a2={self.a2.item():.4f}",
            f"a4={self.a4.item():.4f}",
            f"s6={self.s6.item():.4f}",
            f"cutoff={self.cutoff:.4f} bohr",
        ]
        # 组合成新的字符串表示
        return f"{class_name}({', '.join(params)})"


@compile_mode("script")
class D3CSO_Calculator_f(nn.Module):
    # 　能量和力,以及维利都正常
    def __init__(
        self,
        cutoff: float = 10.0,
        device: float = "cpu",
        xc: str = "PBE",
        bidirectional: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "bidirectional", torch.tensor(bidirectional, dtype=torch.bool)
        )
        # constants for dispersion correction
        self.c6_emb = torch.nn.Embedding.from_pretrained(
            torch.tensor(c6 * c6conv, device=device, dtype=torch.get_default_dtype())
            .unsqueeze(1)
            .clone()
            .detach(),
            freeze=True,
        ).requires_grad_(False)
        self.r0_emb = torch.nn.Embedding.from_pretrained(
            torch.tensor(
                2.0 * r0 / d3_autoang,
                device=device,
                dtype=torch.get_default_dtype(),
            )
            .unsqueeze(1)
            .clone()
            .detach(),
            freeze=True,
        ).requires_grad_(False)
        self.d3_autoang = 0.52917726  # for converting distance from bohr to angstrom
        self.d3_autoev = 27.21138505  # for converting a.u. to eV
        self.cutoff = cutoff / self.d3_autoang

        s6 = 0.73
        a2 = 2.5
        a4 = 6.25
        xc = xc.upper()
        if xc == "BLYP":
            a1 = 1.28
        elif xc == "BP86":
            a1 = 1.01
        elif xc == "PBE":
            a1 = 0.24
        elif xc == "TPSS":
            a1 = 0.72
        elif xc == "B3LYP":
            a1 = 0.86
        elif xc == "PBE0":
            a1 = 0.20
        elif xc == "PW6B95":
            a1 = -0.15
        elif xc == "B2PLYP":
            a1 = 0.24
        elif xc == "BAMBOO":
            a1 = 0.82
            s6 = 0.85
            a4 = 4.5
        elif xc == "ZERO":
            a1 = -0.73
        else:
            raise ValueError(f"[ERROR] Unexpected value xc={xc}")
        self.register_buffer("a1", torch.tensor([a1], dtype=torch.get_default_dtype()))
        self.register_buffer("a2", torch.tensor([a2], dtype=torch.get_default_dtype()))
        self.register_buffer("a4", torch.tensor([a4], dtype=torch.get_default_dtype()))
        self.register_buffer("s6", torch.tensor([s6], dtype=torch.get_default_dtype()))

    def forward(
        self,
        dij: torch.Tensor,  # vec
        rij: torch.Tensor,  # length
        edge_index: torch.Tensor,
        Z: torch.Tensor,
        compute_virials: bool = False,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Compute D3-CSO dispersion energy and pairwise dispersion forces from C6 and r0 parameters.
        """
        rij = rij / self.d3_autoang  # angstrom -> bohr
        dij = dij / self.d3_autoang  # angstrom -> bohr
        c6 = self.c6_emb(Z)
        r0 = self.r0_emb(Z)
        row = edge_index[0]
        col = edge_index[1]
        c6ij = torch.sqrt(c6[row] * c6[col])
        r0ij = 0.5 * (r0[row] + r0[col])
        node_num = Z.shape[0]

        a1 = self.a1

        # D3-CSO dispersion correction
        edisp = (
            -c6ij
            / (rij**6 + self.a4**6)
            * (self.s6 + a1 / (1.0 + torch.exp(rij - self.a2 * r0ij)))
        )
        fdisp = 6 * c6ij * rij**5 / ((rij**6 + (self.a4) ** 6) ** 2) * (
            self.s6 + a1 / (1.0 + torch.exp(rij - self.a2 * r0ij))
        ) + c6ij / (rij**6 + (self.a4) ** 6) * (
            a1
            * torch.exp(rij - self.a2 * r0ij)
            / ((1.0 + torch.exp(rij - self.a2 * r0ij)) ** 2)
        )

        # # 应用截断函数
        if self.cutoff is not None:
            ### 采用bamboo框架使用的, 水平平移
            edisp += self.s6 * c6ij / (self.cutoff**6 + self.a4**6)
            e6 = edisp
            f6 = fdisp
        else:
            e6 = edisp
            f6 = fdisp
        # 力　＝　关于坐标的梯度，的负值　-　*　-　＝　+
        force_contribution = dij * (f6 / (rij + 1e-15))
        # virial
        node_virials = e6.new_zeros((node_num, 3, 3))
        if compute_virials:
            virials = force_contribution.unsqueeze(-2) * dij.unsqueeze(-1)
            node_virials.index_add_(0, row, virials)
            if not self.bidirectional:
                # 随机单向图 index_add_  scatter_add_
                node_virials *= 2.0

        # 初始化力张量
        force = torch.zeros((node_num, 3), device=dij.device, dtype=dij.dtype)
        # 聚合节点能量
        node_g = e6.new_zeros((node_num, 1))
        node_g.index_add_(0, row, e6)

        if not self.bidirectional:
            # 单向图 index_add_  scatter_add_
            node_g *= 2.0
            force.index_add_(0, row, force_contribution)
            force.index_add_(0, col, -force_contribution)

        else:
            # 双向图
            force.index_add_(0, row, force_contribution)

        # 2.0　：　每个原子对的能量，有两个相反的原子力；（与ase的保持一致）
        # virial为负值，能与自动微分的对上
        return (
            0.5 * node_g.squeeze(-1) * self.d3_autoev,
            force * self.d3_autoev / self.d3_autoang,
            0.5 * -node_virials * self.d3_autoev,
            force_contribution * self.d3_autoev / self.d3_autoang,
        )  # node, node, node , edge


@compile_mode("script")
class D3CSO_Calculator(nn.Module):
    # 使用自动微分算力
    def __init__(
        self,
        cutoff: float = 10.0,
        device: float = "cpu",
        xc: str = "PBE",
        bidirectional: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "bidirectional", torch.tensor(bidirectional, dtype=torch.bool)
        )
        # constants for dispersion correction
        self.c6_emb = torch.nn.Embedding.from_pretrained(
            torch.tensor(c6 * c6conv, device=device, dtype=torch.get_default_dtype())
            .unsqueeze(1)
            .clone()
            .detach(),
            freeze=True,
        ).requires_grad_(False)
        self.r0_emb = torch.nn.Embedding.from_pretrained(
            torch.tensor(
                2.0 * r0 / d3_autoang,
                device=device,
                dtype=torch.get_default_dtype(),
            )
            .unsqueeze(1)
            .clone()
            .detach(),
            freeze=True,
        ).requires_grad_(False)
        self.d3_autoang = 0.52917726  # for converting distance from bohr to angstrom
        self.d3_autoev = 27.21138505  # for converting a.u. to eV
        self.cutoff = cutoff / self.d3_autoang

        s6 = 0.73
        a2 = 2.5
        a4 = 6.25
        xc = xc.upper()
        if xc == "BLYP":
            a1 = 1.28
        elif xc == "BP86":
            a1 = 1.01
        elif xc == "PBE":
            a1 = 0.24
        elif xc == "TPSS":
            a1 = 0.72
        elif xc == "B3LYP":
            a1 = 0.86
        elif xc == "PBE0":
            a1 = 0.20
        elif xc == "PW6B95":
            a1 = -0.15
        elif xc == "B2PLYP":
            a1 = 0.24
        elif xc == "BAMBOO":
            a1 = 0.82
            s6 = 0.85
            a4 = 4.5
        elif xc == "ZERO":
            a1 = -0.73
        else:
            raise ValueError(f"[ERROR] Unexpected value xc={xc}")
        self.register_buffer("a1", torch.tensor([a1], dtype=torch.get_default_dtype()))
        self.register_buffer("a2", torch.tensor([a2], dtype=torch.get_default_dtype()))
        self.register_buffer("a4", torch.tensor([a4], dtype=torch.get_default_dtype()))
        self.register_buffer("s6", torch.tensor([s6], dtype=torch.get_default_dtype()))
        # self.coefficient = torch.nn.Parameter(
        #     torch.tensor([1.0], dtype=torch.get_default_dtype())
        # )
        self.register_buffer(
            "coefficient", torch.tensor([0.0], dtype=torch.get_default_dtype())
        )

    def forward(
        self,
        dij: torch.Tensor,  # vec
        rij: torch.Tensor,  # length
        edge_index: torch.Tensor,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute D3-CSO dispersion energy and pairwise dispersion forces from C6 and r0 parameters.
        """
        rij = rij / self.d3_autoang  # angstrom -> bohr
        # dij = dij / self.d3_autoang  # angstrom -> bohr
        c6 = self.c6_emb(Z)
        r0 = self.r0_emb(Z)
        row = edge_index[0]
        col = edge_index[1]
        c6ij = torch.sqrt(c6[row] * c6[col])
        r0ij = 0.5 * (r0[row] + r0[col])

        a1 = self.a1

        # D3-CSO dispersion correction
        edisp = (
            -c6ij
            / (rij**6 + self.a4**6)
            * (self.s6 + a1 / (1.0 + torch.exp(rij - self.a2 * r0ij)))
        )
        # # 应用截断函数
        if self.cutoff is not None:
            ### 采用bamboo框架使用的, 水平平移
            edisp += self.s6 * c6ij / (self.cutoff**6 + self.a4**6)
            e6 = edisp
        else:
            e6 = edisp

        # 聚合节点能量
        node_g = e6.new_zeros((Z.shape[0], 1))
        node_g.index_add_(0, col, e6)
        if not self.bidirectional:
            # 单向图
            node_g *= 2.0
        return node_g.squeeze(-1) * self.d3_autoev * self.coefficient

    def __repr__(self):
        # 调用父类的__repr__方法作为基础
        base_repr = super().__repr__()
        # 提取类名部分
        class_name = base_repr.split("(")[0]
        # 添加自定义参数信息
        params = [
            f"a1={self.a1.item():.4f}",
            f"a2={self.a2.item():.4f}",
            f"a4={self.a4.item():.4f}",
            f"s6={self.s6.item():.4f}",
            f"cutoff={self.cutoff:.4f} bohr",
            f"coefficient={self.coefficient.item():.4f}",
            # f"negetive node_g"
        ]
        # 组合成新的字符串表示
        return f"{class_name}({', '.join(params)})"


if __name__ == "__main__":
    from ase import Atoms
    import scienceplots
    import matplotlib.pyplot as plt
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

    # test
    # TYPE = "KrKr" # 36
    # TYPE = "HeHe"  #2
    TYPE = "OO"  # 6
    Z = torch.tensor([8, 8], dtype=torch.long)
    label1 = ["zero", "b3-lyp",True]  # b-lyp
    # dftd2 = D3CSO_Calculator_edge_forces(cutoff=10.0, xc="B3LYP")
    dftd2 = D3CSO_Calculator(cutoff=10.0, xc="zero")
    dftd2.a1 = torch.ones_like(dftd2.a1) * (-0.67)
    # dftd2.c6_emb.weight *= -0.01
    # dftd2.coefficient = torch.ones_like(dftd2.coefficient) * (-0.3)

    dftd3 = D3CSO_Calculator(cutoff=10.0, xc="zero")
    dftd3.a1 = torch.ones_like(dftd3.a1) * (-0.73)

    # 初始距离（埃）
    initial_distance = 10.0
    # 最终距离（埃）
    final_distance = 1.0
    # 距离步长（埃）
    distance_step = -0.1

    # 原子间的距离数组
    distances = np.arange(initial_distance, final_distance, distance_step)

    positions_list = [[[0, 0, 0], [0, 0, distance.item()]] for distance in distances]

    atoms = []
    for idx, distance in enumerate(distances):
        # 创建两个氢原子的体系
        atoms.append(Atoms(TYPE, positions=positions_list[idx]))

    # test
    r = torch.tensor(distances, dtype=torch.float32)
    # [i1,i2,...] [j1,j2,...]
    edge_index = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.long,
    )
    batch = torch.tensor([0, 0], dtype=torch.long)
    energy2 = []
    forces2 = []

    # true
    energy1 = []
    forces1 = []

    for idx, rs in enumerate(r):
        pos = torch.tensor(
            positions_list[idx], dtype=torch.get_default_dtype()
        ).requires_grad_(True)
        vec = pos[edge_index[1]] - pos[edge_index[0]]
        rs = torch.linalg.norm(vec, dim=-1, keepdim=True)
        E_disp = dftd2(vec, rs, edge_index, Z) 
        # E_disp, f  = dftd2(vec, rs, edge_index, Z, batch)
        f = torch.autograd.grad(
            outputs=[-E_disp],  # [n_graphs, ]
            inputs=[pos],  # [n_nodes, 3]
            grad_outputs=torch.ones_like(E_disp),
            retain_graph=True,  # Make sure the graph is not destroyed during training
            create_graph=True,  # Create graph for second derivative
            allow_unused=True,
        )[0]
        energy2.append(E_disp.sum().item())
        forces2.append(f.detach().cpu().numpy())
        # E_disp = dftd3(vec, rs, edge_index, Z) 
        # # E_disp, f  = dftd2(vec, rs, edge_index, Z, batch)
        # f = torch.autograd.grad(
        #     outputs=[-E_disp],  # [n_graphs, ]
        #     inputs=[pos],  # [n_nodes, 3]
        #     grad_outputs=torch.ones_like(E_disp),
        #     retain_graph=True,  # Make sure the graph is not destroyed during training
        #     create_graph=True,  # Create graph for second derivative
        #     allow_unused=True,
        # )[0]
        # energy1.append(E_disp.sum().item())
        # forces1.append(f.detach().cpu().numpy())

        # forces2.append(forces)
    energy2 = np.array(energy2)
    forces2 = np.array(forces2)

    # # b-lyp
    for atom in atoms:
        atom.set_calculator(
            TorchDFTD3Calculator(
                atoms=atom,
                device="cuda",
                damping=label1[0],
                xc=label1[1],
                old=label1[2],
                cutoff=10.0,
            )
        )
        energy1.append(atom.get_potential_energy())
        forces1.append(atom.get_forces())
    energy1 = np.array(energy1)
    forces1 = np.array(forces1)

    # print(energy2)

    ## plot
    with plt.style.context("science"):
        # 密度分布图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), dpi=250)
        ax1.plot(distances, energy1, label=label1)
        ax1.plot(distances, energy2, "--", label="cso")
        ax1.set_xlabel("distance")
        ax1.set_ylabel("energy")
        ax1.set_title(TYPE)
        ax1.legend()

        ax2.plot(distances, forces1[:, 1, 2], label="forces_z")
        ax2.plot(distances, forces2[:, 1, 2], "--", label="forces_z")
        ax2.plot(distances, forces1[:, 1, 1], label="forces_y")
        ax2.plot(distances, forces2[:, 1, 1], "--", label="forces_y")
        ax2.plot(distances, forces1[:, 1, 0], label="forces_x")
        ax2.plot(distances, forces2[:, 1, 0], "--", label="forces_x")
        ax2.set_xlabel("distance")
        ax2.set_ylabel("forces")
        ax2.set_title(TYPE)
        ax2.legend()

        plt.show()
