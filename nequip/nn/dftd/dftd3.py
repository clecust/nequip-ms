import torch
from torch import nn, Tensor
from typing import Dict, Optional

from e3nn.util.jit import compile_mode

from .dftd3_xc_params import get_dftd3_default_params


import numpy as np
from pathlib import Path
import os


@compile_mode("script")
class D3Calculator(nn.Module):

    def __init__(
        self,
        damping: str = "zero",
        xc: str = "pbe",
        old: bool = True,
        cutoff: float = 95.0, # Angstrom
        cnthr: float = 40.0,
        bidirectional: bool = True,
        abc: bool = False,
        n_chunks: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dtype = dtype  
        self.register_buffer(
            "d3_autoang", torch.tensor(0.52917726, dtype=self.dtype)
        )  # for converting distance from bohr to angstrom
        self.register_buffer("d3_autoev", torch.tensor(27.21138505, dtype=self.dtype)) # for converting a.u. to eV

        self.register_buffer("k1", torch.tensor(16.000, dtype=self.dtype))
        self.register_buffer("k2", torch.tensor(4 / 3, dtype=self.dtype))
        self.register_buffer("k3", torch.tensor(-4.000, dtype=self.dtype))
        self.register_buffer(
            "maxc", torch.tensor(5, dtype=torch.int64)
        )  # maximum number of coordination complexes

        d3_filepath = str(Path(os.path.abspath(__file__)).parent / "dftd3_params.npz")
        d3_params = np.load(d3_filepath)
        c6ab = torch.tensor(d3_params["c6ab"], dtype=self.dtype)
        r0ab = torch.tensor(d3_params["r0ab"], dtype=self.dtype)
        rcov = torch.tensor(d3_params["rcov"], dtype=self.dtype)
        r2r4 = torch.tensor(d3_params["r2r4"], dtype=self.dtype)
        # (95, 95, 5, 5, 3) c0, c1, c2 for coordination number dependent c6ab term.
        self.register_buffer("c6ab", c6ab)
        self.register_buffer("r0ab", r0ab)  # atom pair distance (95, 95)
        self.register_buffer("rcov", rcov)  # atom covalent distance (95)
        self.register_buffer("r2r4", r2r4)  # (95,)

        if cnthr > cutoff:
            print(
                f"WARNING: cnthr {cnthr} is larger than cutoff {cutoff}. "
                f"cutoff distance is used for cnthr"
            )
            cnthr = cutoff
        self.register_buffer(
            "cutoff", torch.tensor(cutoff / self.d3_autoang, dtype=self.dtype)
        )
        self.register_buffer(
            "cnthr", torch.tensor(cnthr / self.d3_autoang, dtype=self.dtype)
        )
        self.register_buffer("abc", torch.tensor(abc, dtype=torch.bool))
        self.register_buffer(
            "bidirectional", torch.tensor(bidirectional, dtype=torch.bool)
        )
        # self.register_buffer("n_chunks", torch.tensor(n_chunks, dtype=torch.bool))
        self.n_chunks = n_chunks

        #######

        self.params = get_dftd3_default_params(damping, xc, old=old)
        self.damping = damping

        # training weights
        # prefix_6 = torch.zeros(3, dtype=self.dtype)  # 调控势井的深度和水平偏移
        # self.prefix_6 = torch.nn.Parameter(prefix_6)
        # prefix_c6ab = torch.zeros((87, 87), dtype=self.dtype)  # 调控势井的深度
        # self.prefix_c6ab = torch.nn.Parameter(prefix_c6ab)

    def _compute_coordination_number(
        self,
        Z: Tensor,
        r: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> Tensor:
        if self.cutoff is not None:
            indices = torch.nonzero(r <= self.cutoff).reshape(-1)
            r = r[indices]
            idx_i = idx_i[indices]
            idx_j = idx_j[indices]

        Zi = Z[idx_i]
        Zj = Z[idx_j]
        rco = self.rcov[Zi] + self.rcov[Zj]
        rr = rco.type(r.dtype) / r
        damp = 1.0 / (1.0 + torch.exp(-self.k1 * (rr - 1.0)))

        if self.cutoff is not None :
            damp *= self._poly_smoothing( r, self.cutoff )

        n_atoms = Z.shape[0]
        g = damp.new_zeros((n_atoms,))
        g = g.scatter_add_(0, idx_i, damp)
        if not self.bidirectional:
            g = g.scatter_add_(0, idx_j, damp)

        return g

    def interpolate_c6(
        self,
        Zi: Tensor,
        Zj: Tensor,
        nci: Tensor,
        ncj: Tensor,
    ) -> Tensor:
        if self.n_chunks is None:
            return self._getc6_impl(Zi, Zj, nci, ncj)

        chunk_size = (Zi.shape[0] + self.n_chunks - 1) // self.n_chunks
        c6s = []
        for i in range(self.n_chunks):
            slc = slice(i * chunk_size, (i + 1) * chunk_size)
            c6s.append(self._getc6_impl(Zi[slc], Zj[slc], nci[slc], ncj[slc]))

        return torch.cat(c6s, dim=0)

    def _getc6_impl(
        self, Zi: Tensor, Zj: Tensor, nci: Tensor, ncj: Tensor 
    ) -> Tensor:
        cn0, cn1, cn2 = self.c6ab.reshape(-1, 5, 5, 3).split(1, dim=3)
        index = Zi * self.c6ab.size(1) + Zj

        cn0 = cn0.squeeze(dim=3)[index].type(nci.dtype)
        cn1 = cn1.squeeze(dim=3)[index].type(nci.dtype)
        cn2 = cn2.squeeze(dim=3)[index].type(nci.dtype)

        r = (cn1 - nci[:, None, None]) ** 2 + (cn2 - ncj[:, None, None]) ** 2
        n_edges, n_c6ab = r.shape[0], r.shape[1] * r.shape[2]

        k3_rnc = torch.where(cn0 > 0.0, self.k3 * r, -1.0e20).view(n_edges, n_c6ab)
        r_ratio = torch.softmax(k3_rnc, dim=1)
        c6 = (r_ratio * cn0.view(n_edges, n_c6ab)).sum(dim=1)
        return c6
    def _poly_smoothing(self,r: Tensor, cutoff: Tensor) -> Tensor:

        cuton = cutoff - 1
        x = (cutoff - r) / (cutoff - cuton)
        x2 = x**2
        x3 = x2 * x
        x4 = x3 * x
        x5 = x4 * x
        return torch.where(
            r <= cuton,
            torch.ones_like(x),
            torch.where(r >= cutoff, torch.zeros_like(x), 6 * x5 - 15 * x4 + 10 * x3),
        )

    def forward(
        self,
        r: Tensor,
        edge_index: Tensor,
        Z: Tensor,
        batch: Tensor,
    ) -> Tensor:
        r = r.squeeze(-1) / self.d3_autoang  # angstrom -> bohr
        r2 = r**2
        r6 = r2**3
        r8 = r6 * r2

        idx_i = edge_index[0]
        idx_j = edge_index[1]

        Zi, Zj = Z[idx_i], Z[idx_j]

        nc = self._compute_coordination_number(Z, r, idx_i, idx_j)
        nci, ncj = nc[idx_i], nc[idx_j]

        c6 = self.interpolate_c6(Zi, Zj, nci, ncj)
        c8 = 3 * c6 * self.r2r4[Zi].type(c6.dtype) * self.r2r4[Zj].type(c6.dtype)

        s6, s8 = self.params["s6"], self.params["s18"]
        # "bj", "bjm"
        tmp = self.params["rs6"] * torch.sqrt(c8 / c6) + self.params["rs18"]
        tmp2, tmp6, tmp8 = tmp**2, tmp**6, tmp**8
        e6 = -0.5 * s6 * c6 / (r6 + tmp6)
        e8 = -0.5 * s8 * c8 / (r8 + tmp8)

        e68 = e6 + e8
        if self.cutoff is not None:
            e68 *= self._poly_smoothing(r,self.cutoff)

        ## scatter to node
        node_g = e68.new_zeros((batch.shape[0],))

        if not self.bidirectional:
            node_g.scatter_add_(0, idx_i, e68)
            node_g.scatter_add_(0, idx_j, e68)
        else:
            node_g.scatter_add_(0, idx_i, e68)

        ## scatter to sample
        # g = e68.new_zeros((int(batch[-1]) + 1, ))
        # if not self.bidirectional:
        #     g.scatter_add_(0, batch[idx_i], e68)
        #     g.scatter_add_(0, batch[idx_j], e68)
        # else:
        #     g.scatter_add_(0, batch[idx_i], e68)
        return node_g * self.d3_autoev


@compile_mode("script")
class D3Calculator_forces(nn.Module):
    ## 给出解析力：
    ## DFTD3 的　c6　系数太复杂了，涉及到了r值的计算 目前，还存在梯度明显的差异

    def __init__(
        self,
        damping: str = "zero",
        xc: str = "pbe",
        old: bool = True,
        cutoff: float = 95.0,  # Angstrom
        cnthr: float = 40.0,
        bidirectional: bool = True,
        abc: bool = False,
        n_chunks: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.register_buffer(
            "d3_autoang", torch.tensor(0.52917726, dtype=self.dtype)
        )  # for converting distance from bohr to angstrom
        self.register_buffer(
            "d3_autoev", torch.tensor(27.21138505, dtype=self.dtype)
        )  # for converting a.u. to eV

        self.register_buffer("k1", torch.tensor(16.000, dtype=self.dtype))
        self.register_buffer("k2", torch.tensor(4 / 3, dtype=self.dtype))
        self.register_buffer("k3", torch.tensor(-4.000, dtype=self.dtype))
        self.register_buffer(
            "maxc", torch.tensor(5, dtype=torch.int64)
        )  # maximum number of coordination complexes

        d3_filepath = str(Path(os.path.abspath(__file__)).parent / "dftd3_params.npz")
        d3_params = np.load(d3_filepath)
        c6ab = torch.tensor(d3_params["c6ab"], dtype=self.dtype)
        r0ab = torch.tensor(d3_params["r0ab"], dtype=self.dtype)
        rcov = torch.tensor(d3_params["rcov"], dtype=self.dtype)
        r2r4 = torch.tensor(d3_params["r2r4"], dtype=self.dtype)
        # (95, 95, 5, 5, 3) c0, c1, c2 for coordination number dependent c6ab term.
        self.register_buffer("c6ab", c6ab)
        self.register_buffer("r0ab", r0ab)  # atom pair distance (95, 95)
        self.register_buffer("rcov", rcov)  # atom covalent distance (95)
        self.register_buffer("r2r4", r2r4)  # (95,)

        if cnthr > cutoff:
            print(
                f"WARNING: cnthr {cnthr} is larger than cutoff {cutoff}. "
                f"cutoff distance is used for cnthr"
            )
            cnthr = cutoff
        self.register_buffer(
            "cutoff", torch.tensor(cutoff / self.d3_autoang, dtype=self.dtype)
        )
        self.register_buffer(
            "cnthr", torch.tensor(cnthr / self.d3_autoang, dtype=self.dtype)
        )
        self.register_buffer("abc", torch.tensor(abc, dtype=torch.bool))
        self.register_buffer(
            "bidirectional", torch.tensor(bidirectional, dtype=torch.bool)
        )
        # self.register_buffer("n_chunks", torch.tensor(n_chunks, dtype=torch.bool))
        self.n_chunks = n_chunks

        #######

        self.params = get_dftd3_default_params(damping, xc, old=old)
        self.damping = damping

    def _compute_coordination_number(
        self,
        Z: Tensor,
        r: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> Tensor:
        if self.cutoff is not None:
            indices = torch.nonzero(r <= self.cutoff).reshape(-1)
            r = r[indices]
            idx_i = idx_i[indices]
            idx_j = idx_j[indices]

        Zi = Z[idx_i]
        Zj = Z[idx_j]
        rco = self.rcov[Zi] + self.rcov[Zj]
        rr = rco.type(r.dtype) / r
        damp = 1.0 / (1.0 + torch.exp(-self.k1 * (rr - 1.0)))

        if self.cutoff is not None:
            cut , _ = self.cosine_smooth_cutoff(r, self.cutoff - 1.0, self.cutoff)
            damp *= cut

        n_atoms = Z.shape[0]
        g = damp.new_zeros((n_atoms,))
        g = g.scatter_add_(0, idx_i, damp)
        if not self.bidirectional:
            g = g.scatter_add_(0, idx_j, damp)

        return g

    def interpolate_c6(
        self,
        Zi: Tensor,
        Zj: Tensor,
        nci: Tensor,
        ncj: Tensor,
    ) -> Tensor:
        if self.n_chunks is None:
            return self._getc6_impl(Zi, Zj, nci, ncj)

        chunk_size = (Zi.shape[0] + self.n_chunks - 1) // self.n_chunks
        c6s = []
        for i in range(self.n_chunks):
            slc = slice(i * chunk_size, (i + 1) * chunk_size)
            c6s.append(self._getc6_impl(Zi[slc], Zj[slc], nci[slc], ncj[slc]))

        return torch.cat(c6s, dim=0)

    def _getc6_impl(self, Zi: Tensor, Zj: Tensor, nci: Tensor, ncj: Tensor) -> Tensor:
        cn0, cn1, cn2 = self.c6ab.reshape(-1, 5, 5, 3).split(1, dim=3)
        index = Zi * self.c6ab.size(1) + Zj

        cn0 = cn0.squeeze(dim=3)[index].type(nci.dtype)
        cn1 = cn1.squeeze(dim=3)[index].type(nci.dtype)
        cn2 = cn2.squeeze(dim=3)[index].type(nci.dtype)

        r = (cn1 - nci[:, None, None]) ** 2 + (cn2 - ncj[:, None, None]) ** 2
        n_edges, n_c6ab = r.shape[0], r.shape[1] * r.shape[2]

        k3_rnc = torch.where(cn0 > 0.0, self.k3 * r, -1.0e20).view(n_edges, n_c6ab)
        r_ratio = torch.softmax(k3_rnc, dim=1)
        c6 = (r_ratio * cn0.view(n_edges, n_c6ab)).sum(dim=1)
        return c6

    def cosine_smooth_cutoff(self, x, rs, rc):
        # 计算截断函数及其导数
        condition = (x >= rs) & (x <= rc)
        cutoff = torch.zeros_like(x)
        d_cutoff = torch.zeros_like(x)

        # 当 rs <= x <= rc 时的截断函数和导数
        x_cond = x[condition]
        scaled = torch.pi * (x_cond - rs) / (rc - rs)
        cutoff[condition] = 0.5 * (torch.cos(scaled) + 1)
        d_cutoff[condition] = -0.5 * torch.pi / (rc - rs) * torch.sin(scaled)

        # 当 x < rs 时截断函数为1，导数为0
        cutoff[x < rs] = 1

        return cutoff, d_cutoff

    def forward(
        self,
        vec: Tensor,
        length: Tensor,
        edge_index: Tensor,
        Z: Tensor,
        batch: Tensor,
    ) -> Tensor:
        row, col = edge_index
        r = length.squeeze(-1) / self.d3_autoang  # angstrom -> bohr

        r2 = r**2
        r6 = r2**3
        r8 = r6 * r2

        Zi, Zj = Z[row], Z[col]

        nc = self._compute_coordination_number(Z, r, row, col)
        nci, ncj = nc[row], nc[col]

        c6 = self.interpolate_c6(Zi, Zj, nci, ncj)
        c8 = 3 * c6 * self.r2r4[Zi].type(c6.dtype) * self.r2r4[Zj].type(c6.dtype)

        s6, s8 = self.params["s6"], self.params["s18"]
        tmp = self.params["rs6"] * torch.sqrt(c8 / c6) + self.params["rs18"]
        tmp6, tmp8 = tmp**6, tmp**8

        # 计算基础能量项
        e6 = -0.5 * s6 * c6 / (r6 + tmp6)
        e8 = -0.5 * s8 * c8 / (r8 + tmp8)
        e68_bare = e6 + e8

        # 计算能量项对r的导数（玻尔单位）
        de6_dr = 3 * s6 * c6 * r**5 / (r6 + tmp6) ** 2
        de8_dr = 4 * s8 * c8 * r**7 / (r8 + tmp8) ** 2
        de68_dr = de6_dr + de8_dr

        # 应用截断函数
        de68_dr_total = de68_dr
        if  self.cutoff is not None:
            cutoff_factor, d_cutoff = self.cosine_smooth_cutoff(
                r, self.cutoff - 1.0, self.cutoff
            )
            e68 = e68_bare * cutoff_factor
            # 应用乘积法则计算总导数:d(e68)/dr = d(e68_bare)/dr * cutoff_factor + e68_bare * d_cutoff
            de68_dr_total = de68_dr * cutoff_factor + e68_bare * d_cutoff
        else:
            e68 = e68_bare
            de68_dr_total = de68_dr

        # 计算力贡献（负梯度）
        force_contribution = (-de68_dr_total).unsqueeze(-1) * (
            vec / (r.unsqueeze(-1) + 1e-15)
        )

        # 初始化力张量（修正原来的x未定义错误）
        force = torch.zeros((batch.size(0), 3), device=vec.device, dtype=vec.dtype)

        # 聚合节点能量
        node_g = e68.new_zeros(batch.size(0))
        if not self.bidirectional:
            # 单向图
            node_g.scatter_add_(0, row, e68)
            node_g.scatter_add_(0, col, e68)
            force.index_add_(0, row, force_contribution)
            force.index_add_(0, col, -force_contribution)

        else:
            # 双向图
            node_g.scatter_add_(0, row, e68)
            force.index_add_(0, row, force_contribution)



        return node_g * self.d3_autoev, force * self.d3_autoev


@compile_mode("script")
class D3Calculator_train(D3Calculator):
    # 可训练的参数
    def __init__(
        self,
        prefix_dict: Dict[str, float] = {"a1": 0.5, "s8": 4.0, "a2": 7.7},
        old: bool = False,
        cutoff: float = 95.0,  # Angstrom
        cnthr: float = 40.0,
        bidirectional: bool = True,
        abc: bool = False,
        n_chunks: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super(D3Calculator_train, self).__init__(damping='bj',  xc= "pbe",
                                    old=False, cutoff=cutoff,cnthr=cnthr,
                                    bidirectional=bidirectional,abc=abc,n_chunks=n_chunks,dtype=dtype )
        self.prefix_a1 = prefix_dict['a1']
        self.prefix_s8 = prefix_dict['s8']
        self.prefix_a2 = prefix_dict['a2']
        self.a1 = nn.Parameter(torch.tensor([1.0], dtype=torch.get_default_dtype()))
        self.s8 = nn.Parameter(torch.tensor([1.0], dtype=torch.get_default_dtype()))
        self.a2 = nn.Parameter(torch.tensor([1.0], dtype=torch.get_default_dtype()))
        self.nonlinear = nn.Sigmoid()

    def forward(
        self,
        r: Tensor,
        edge_index: Tensor,
        Z: Tensor,
        batch: Tensor,
    ) -> Tensor:
        r = r.squeeze(-1) / self.d3_autoang  # angstrom -> bohr
        r2 = r**2
        r6 = r2**3
        r8 = r6 * r2

        idx_i = edge_index[0]
        idx_j = edge_index[1]

        Zi, Zj = Z[idx_i], Z[idx_j]

        nc = self._compute_coordination_number(Z, r, idx_i, idx_j)
        nci, ncj = nc[idx_i], nc[idx_j]

        c6 = self.interpolate_c6(Zi, Zj, nci, ncj)
        c8 = 3 * c6 * self.r2r4[Zi].type(c6.dtype) * self.r2r4[Zj].type(c6.dtype)

        ###
        s6 = 1.0
        s8 = self.nonlinear(self.s8) * self.prefix_s8
        a1 = self.nonlinear(self.a1) * self.prefix_a1
        a2 = self.nonlinear(self.a2) * self.prefix_a2

        # "bj", "bjm"
        tmp = a1 * torch.sqrt(c8 / c6) + a2 + 2.0 # 2.0 aviod zero ..... 
        tmp6, tmp8 = tmp**6, tmp**8
        e6 = -0.5 * s6 * c6 / (r6 + tmp6)
        e8 = -0.5 * s8 * c8 / (r8 + tmp8)

        e68 = e6 + e8
        if self.cutoff is not None:
            e68 *= self._poly_smoothing(r,self.cutoff)

        ## scatter to node
        # node_g = e68.new_zeros((int(idx_j.max()) + 1,))
        node_g = e68.new_zeros((batch.shape[0],))

        if not self.bidirectional:
            node_g.scatter_add_(0, idx_i, e68)
            node_g.scatter_add_(0, idx_j, e68)
        else:
            node_g.scatter_add_(0, idx_i, e68)

        ## scatter to sample
        # g = e68.new_zeros((int(batch[-1]) + 1, ))
        # if not self.bidirectional:
        #     g.scatter_add_(0, batch[idx_i], e68)
        #     g.scatter_add_(0, batch[idx_j], e68)
        # else:
        #     g.scatter_add_(0, batch[idx_i], e68)

        return  node_g * self.d3_autoev


@compile_mode("script")
class D3Calculator_node_train(nn.Module):
    # 另外一种，可训练的版本
    def __init__(
        self,
        damping: str = "zero",
        xc: str = "pbe",
        old: bool = True,
        cutoff: float = 95.0,  # Angstrom
        cnthr: float = 40.0,
        bidirectional: bool = True,
        abc: bool = False,
        n_chunks: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dtype = dtype
        self.register_buffer(
            "d3_autoang", torch.tensor(0.52917726, dtype=self.dtype)
        )  # for converting distance from bohr to angstrom
        self.register_buffer(
            "d3_autoev", torch.tensor(27.21138505, dtype=self.dtype)
        )  # for converting a.u. to eV

        self.register_buffer("k1", torch.tensor(16.000, dtype=self.dtype))
        self.register_buffer("k2", torch.tensor(4 / 3, dtype=self.dtype))
        self.register_buffer("k3", torch.tensor(-4.000, dtype=self.dtype))
        self.register_buffer(
            "maxc", torch.tensor(5, dtype=torch.int64)
        )  # maximum number of coordination complexes

        d3_filepath = str(Path(os.path.abspath(__file__)).parent / "dftd3_params.npz")
        d3_params = np.load(d3_filepath)
        c6ab = torch.tensor(d3_params["c6ab"], dtype=self.dtype)
        r0ab = torch.tensor(d3_params["r0ab"], dtype=self.dtype)
        rcov = torch.tensor(d3_params["rcov"], dtype=self.dtype)
        r2r4 = torch.tensor(d3_params["r2r4"], dtype=self.dtype)
        # (95, 95, 5, 5, 3) c0, c1, c2 for coordination number dependent c6ab term.
        self.register_buffer("c6ab", c6ab)
        self.register_buffer("r0ab", r0ab)  # atom pair distance (95, 95)
        self.register_buffer("rcov", rcov)  # atom covalent distance (95)
        self.register_buffer("r2r4", r2r4)  # (95,)

        if cnthr > cutoff:
            print(
                f"WARNING: cnthr {cnthr} is larger than cutoff {cutoff}. "
                f"cutoff distance is used for cnthr"
            )
            cnthr = cutoff
        self.register_buffer(
            "cutoff", torch.tensor(cutoff / self.d3_autoang, dtype=self.dtype)
        )
        self.register_buffer(
            "cnthr", torch.tensor(cnthr / self.d3_autoang, dtype=self.dtype)
        )
        self.register_buffer("abc", torch.tensor(abc, dtype=torch.bool))
        self.register_buffer(
            "bidirectional", torch.tensor(bidirectional, dtype=torch.bool)
        )
        # self.register_buffer("n_chunks", torch.tensor(n_chunks, dtype=torch.bool))
        self.n_chunks = n_chunks

        #######

        self.params = get_dftd3_default_params(damping, xc, old=old)
        self.damping = damping

        # training weights
        # prefix_6 = torch.zeros(3, dtype=self.dtype)  # 调控势井的深度和水平偏移
        # self.prefix_6 = torch.nn.Parameter(prefix_6)
        # prefix_c6ab = torch.zeros((87, 87), dtype=self.dtype)  # 调控势井的深度
        # self.prefix_c6ab = torch.nn.Parameter(prefix_c6ab)
        self.nonlinear = nn.Sigmoid()

    def _compute_coordination_number(
        self,
        Z: Tensor,
        r: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
    ) -> Tensor:
        if self.cutoff is not None:
            indices = torch.nonzero(r <= self.cutoff).reshape(-1)
            r = r[indices]
            idx_i = idx_i[indices]
            idx_j = idx_j[indices]

        Zi = Z[idx_i]
        Zj = Z[idx_j]
        rco = self.rcov[Zi] + self.rcov[Zj]
        rr = rco.type(r.dtype) / r
        damp = 1.0 / (1.0 + torch.exp(-self.k1 * (rr - 1.0)))

        if self.cutoff is not None:
            damp *= self._poly_smoothing(r, self.cutoff)

        n_atoms = Z.shape[0]
        g = damp.new_zeros((n_atoms,))
        g = g.scatter_add_(0, idx_i, damp)
        if not self.bidirectional:
            g = g.scatter_add_(0, idx_j, damp)

        return g

    def interpolate_c6(
        self,
        Zi: Tensor,
        Zj: Tensor,
        nci: Tensor,
        ncj: Tensor,
    ) -> Tensor:
        if self.n_chunks is None:
            return self._getc6_impl(Zi, Zj, nci, ncj)

        chunk_size = (Zi.shape[0] + self.n_chunks - 1) // self.n_chunks
        c6s = []
        for i in range(self.n_chunks):
            slc = slice(i * chunk_size, (i + 1) * chunk_size)
            c6s.append(self._getc6_impl(Zi[slc], Zj[slc], nci[slc], ncj[slc]))

        return torch.cat(c6s, dim=0)

    def _getc6_impl(self, Zi: Tensor, Zj: Tensor, nci: Tensor, ncj: Tensor) -> Tensor:
        cn0, cn1, cn2 = self.c6ab.reshape(-1, 5, 5, 3).split(1, dim=3)
        index = Zi * self.c6ab.size(1) + Zj

        cn0 = cn0.squeeze(dim=3)[index].type(nci.dtype)
        cn1 = cn1.squeeze(dim=3)[index].type(nci.dtype)
        cn2 = cn2.squeeze(dim=3)[index].type(nci.dtype)

        r = (cn1 - nci[:, None, None]) ** 2 + (cn2 - ncj[:, None, None]) ** 2
        n_edges, n_c6ab = r.shape[0], r.shape[1] * r.shape[2]

        k3_rnc = torch.where(cn0 > 0.0, self.k3 * r, -1.0e20).view(n_edges, n_c6ab)
        r_ratio = torch.softmax(k3_rnc, dim=1)
        c6 = (r_ratio * cn0.view(n_edges, n_c6ab)).sum(dim=1)
        return c6

    def _poly_smoothing(self, r: Tensor, cutoff: Tensor) -> Tensor:

        cuton = cutoff - 1
        x = (cutoff - r) / (cutoff - cuton)
        x2 = x**2
        x3 = x2 * x
        x4 = x3 * x
        x5 = x4 * x
        return torch.where(
            r <= cuton,
            torch.ones_like(x),
            torch.where(r >= cutoff, torch.zeros_like(x), 6 * x5 - 15 * x4 + 10 * x3),
        )

    def forward(
        self,
        r: Tensor,
        edge_index: Tensor,
        Z: Tensor,
        batch: Tensor,
        node_prefix: Tensor,
    ) -> Tensor:
        r = r.squeeze(-1) / self.d3_autoang  # angstrom -> bohr
        r2 = r**2
        r6 = r2**3
        r8 = r6 * r2

        idx_i = edge_index[0]
        idx_j = edge_index[1]

        Zi, Zj = Z[idx_i], Z[idx_j]
        # node prefix scale to  0.5~1.5
        node_prefix_ij = node_prefix[idx_i] + node_prefix[idx_j]
        node_prefix_ij = self.nonlinear(node_prefix_ij) + 0.5

        nc = self._compute_coordination_number(Z, r, idx_i, idx_j)
        nci, ncj = nc[idx_i], nc[idx_j]

        c6 = self.interpolate_c6(Zi, Zj, nci, ncj)
        c8 = 3 * c6 * self.r2r4[Zi].type(c6.dtype) * self.r2r4[Zj].type(c6.dtype)

        s6, s8 = self.params["s6"], self.params["s18"]
        # "bj", "bjm"
        tmp = (
            node_prefix_ij[:,1]*self.params["rs6"] * torch.sqrt(c8 / c6)
            + self.params["rs18"] * node_prefix_ij[:,2]
        )
        tmp2, tmp6, tmp8 = tmp**2, tmp**6, tmp**8
        e6 = -0.5 * s6 * c6 / (r6 + tmp6)
        e8 = -0.5 * node_prefix_ij[:, 0] * s8 * c8 / (r8 + tmp8)

        e68 = e6 + e8
        if self.cutoff is not None:
            e68 *= self._poly_smoothing(r, self.cutoff)

        ## scatter to node
        # node_g = e68.new_zeros((int(idx_j.max()) + 1,))
        node_g = e68.new_zeros((batch.shape[0],))

        if not self.bidirectional:
            node_g.scatter_add_(0, idx_i, e68)
            node_g.scatter_add_(0, idx_j, e68)
        else:
            node_g.scatter_add_(0, idx_i, e68)

        ## scatter to sample
        # g = e68.new_zeros((int(batch[-1]) + 1, ))
        # if not self.bidirectional:
        #     g.scatter_add_(0, batch[idx_i], e68)
        #     g.scatter_add_(0, batch[idx_j], e68)
        # else:
        #     g.scatter_add_(0, batch[idx_i], e68)
        return node_g * self.d3_autoev
