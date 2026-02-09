import numpy as np
import torch
import torchtt as tntt


class MPOLinearTorchTT:
    """
    Linear layer y = W x + b using a TT/MPO representation of W.
    Intended as a drop-in replacement for a dense FFN second layer.
    """

    def __init__(self, factors, max_bond, cutoff=1e-12, device="cpu"):
        """
        factors: list[int] with prod(factors) = input_dim = output_dim
        max_bond: TT rank cap during TT-SVD
        cutoff: singular value cutoff during TT-SVD
        """
        self.factors = list(map(int, factors))
        self.K = len(self.factors)
        self.max_bond = int(max_bond)
        self.cutoff = float(cutoff)
        self.device = device

        self.tt_matrix = None   # torchtt.TT
        self.bias = None        # torch tensor (N, 1)

    @staticmethod
    def _prod(xs):
        out = 1
        for x in xs:
            out *= int(x)
        return out

    def _dense_to_tt_cores(self, W):
        """
        TT-SVD for a square matrix W of shape (N, N).
        Returns TT cores with shape (rL, p_out, p_in, rR).
        """
        factors = self.factors
        K = self.K
        N = self._prod(factors)

        W = np.asarray(W)
        if W.shape != (N, N):
            raise ValueError(f"W must be ({N},{N}), got {W.shape}")

        # reshape to (out_factors..., in_factors...)
        T = W.reshape(*factors, *factors)

        # interleave: (out0, in0, out1, in1, ...)
        perm = []
        for k in range(K):
            perm.append(k)
            perm.append(K + k)
        T = T.transpose(*perm)

        cores = []
        rL = 1

        for k in range(K - 1):
            pk = factors[k]
            T = T.reshape(rL * (pk * pk), -1)

            U, S, Vh = np.linalg.svd(T, full_matrices=False)

            if self.cutoff is not None:
                keep = max(1, int(np.sum(S > self.cutoff)))
            else:
                keep = S.shape[0]

            rR = min(keep, self.max_bond, S.shape[0])

            U = U[:, :rR]
            S = S[:rR]
            Vh = Vh[:rR]

            core = U.reshape(rL, pk, pk, rR)     # (rL, out, in, rR)
            core = core.transpose(0, 1, 2, 3)    # already correct
            cores.append(core)

            T = (S[:, None] * Vh)
            rL = rR

        # last core
        pk = factors[-1]
        core = T.reshape(rL, pk, pk, 1)
        cores.append(core)

        return cores

    def init_from_weights(self, W, b):
        """
        W: (N, N)
        b: (N,) or (N,1)
        """
        N = self._prod(self.factors)
        b = np.asarray(b).reshape(N, 1)

        # build TT cores
        cores_np = self._dense_to_tt_cores(W)

        # convert to torchtt format: (rL, p_out, p_in, rR)
        tt_cores = []
        for G in cores_np:
            Gt = torch.tensor(G, dtype=torch.float32, device=self.device)
            tt_cores.append(Gt)

        self.tt_matrix = tntt.TT(tt_cores)
        self.bias = torch.tensor(b, dtype=torch.float32, device=self.device)

    def forward(self, x):
        """
        x: (N,) or (N,1)
        returns: (N,1)
        """
        if self.tt_matrix is None:
            raise RuntimeError("Call init_from_weights() first")

        x = np.asarray(x).reshape(-1)
        N = self._prod(self.factors)
        if x.size != N:
            raise ValueError(f"x has size {x.size}, expected {N}")

        xt = torch.tensor(
            x.reshape(*self.factors),
            dtype=torch.float32,
            device=self.device,
        )

        y = self.tt_matrix @ xt
        y = y.reshape(N, 1)

        return (y + self.bias).cpu().numpy()
