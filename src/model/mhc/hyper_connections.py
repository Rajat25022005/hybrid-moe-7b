"""
Manifold-Constrained Hyper-Connections (mHC).
Replaces standard residual connections with expanded residual stream
constrained to the Birkhoff polytope (doubly stochastic matrices).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.normalization import RMSNorm


class ManifoldHyperConnection(nn.Module):
    """
    mHC: Expands residual stream by factor n_hc, with three linear mappings:
    - A (input): selects what enters the layer
    - B (residual): transforms the residual stream (constrained to doubly stochastic)
    - C (output): selects what exits the layer
    
    Update: X_{l+1} = B_l * X_l + C_l * F_l(A_l * X_l)
    
    Dynamic parameterization: mappings are input-dependent + static bias.
    """

    def __init__(self, hidden_dim: int, expansion: int = 4, sinkhorn_iters: int = 20):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_hc = expansion
        self.sinkhorn_iters = sinkhorn_iters

        # Normalization for input
        self.norm = RMSNorm(expansion * hidden_dim)

        # Dynamic projection weights
        nhc_d = expansion * hidden_dim
        self.W_pre = nn.Linear(nhc_d, expansion, bias=False)     # for A
        self.W_res = nn.Linear(nhc_d, expansion * expansion, bias=False)  # for B
        self.W_post = nn.Linear(nhc_d, expansion, bias=False)    # for C

        # Static biases
        self.S_pre = nn.Parameter(torch.zeros(1, expansion))
        self.S_res = nn.Parameter(torch.zeros(expansion, expansion))
        self.S_post = nn.Parameter(torch.zeros(expansion, 1))

        # Gating factors (initialized small for gradual activation)
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))

    def init_state(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Initialize the expanded residual state from the initial hidden states.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            X: (batch, seq_len, n_hc, hidden_dim)
        """
        B, N, D = hidden_states.shape
        X = torch.zeros(B, N, self.n_hc, D, device=hidden_states.device, dtype=hidden_states.dtype)
        X[:, :, 0] = hidden_states  # Put initial state in first slot
        return X

    def _compute_mappings(self, X: torch.Tensor) -> tuple:
        """
        Compute A, B, C mappings from the current residual state.
        
        Args:
            X: (batch, seq_len, n_hc, hidden_dim)
        Returns:
            A: (batch, seq_len, 1, n_hc) — input mapping
            B: (batch, seq_len, n_hc, n_hc) — residual mapping (doubly stochastic)
            C: (batch, seq_len, n_hc, 1) — output mapping
        """
        B_sz, N, n, D = X.shape

        # Flatten and normalize
        X_flat = X.reshape(B_sz, N, n * D)  # (B, N, n_hc * d)
        X_norm = self.norm(X_flat)

        # Generate raw parameters
        A_raw = self.alpha_pre * self.W_pre(X_norm) + self.S_pre  # (B, N, n_hc)
        B_raw = self.alpha_res * self.W_res(X_norm).view(B_sz, N, n, n) + self.S_res  # (B, N, n, n)
        C_raw = self.alpha_post * self.W_post(X_norm) + self.S_post.squeeze(-1)  # (B, N, n_hc)

        # Apply constraints
        A = torch.sigmoid(A_raw).unsqueeze(-2)  # (B, N, 1, n_hc) — non-negative, bounded
        C = (2 * torch.sigmoid(C_raw)).unsqueeze(-1)  # (B, N, n_hc, 1)

        # Project B onto doubly stochastic manifold via Sinkhorn-Knopp
        B_mat = self._sinkhorn(B_raw)  # (B, N, n_hc, n_hc)

        return A, B_mat, C

    def _sinkhorn(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn-Knopp algorithm to project onto doubly stochastic matrices.
        
        Args:
            raw: (batch, seq_len, n, n) — unconstrained matrix
        Returns:
            (batch, seq_len, n, n) — doubly stochastic matrix
        """
        M = torch.exp(raw)  # Ensure positivity

        for _ in range(self.sinkhorn_iters):
            # Column normalization
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-9)
            # Row normalization
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-9)

        return M

    def pre_block(self, X: torch.Tensor) -> tuple:
        """
        Compute the input to the layer from the expanded residual state.
        Returns the layer input and the mappings for later use.
        
        Args:
            X: (batch, seq_len, n_hc, hidden_dim)
        Returns:
            layer_input: (batch, seq_len, hidden_dim)
            mappings: (A, B, C) for use in post_block
        """
        A, B_mat, C = self._compute_mappings(X)

        # Layer input = A @ X -> (B, N, 1, n_hc) @ (B, N, n_hc, D) -> (B, N, 1, D)
        layer_input = torch.matmul(A, X).squeeze(-2)  # (B, N, D)

        return layer_input, (A, B_mat, C)

    def post_block(
        self,
        X: torch.Tensor,
        layer_output: torch.Tensor,
        mappings: tuple,
    ) -> torch.Tensor:
        """
        Update the residual state after the layer.
        X_{l+1} = B * X + C * F(A * X)
        
        Args:
            X: (batch, seq_len, n_hc, hidden_dim) — current state
            layer_output: (batch, seq_len, hidden_dim) — F(A * X)
            mappings: (A, B, C) from pre_block
        Returns:
            X_new: (batch, seq_len, n_hc, hidden_dim)
        """
        A, B_mat, C = mappings

        # Residual transformation: B @ X
        BX = torch.matmul(B_mat, X)  # (B, N, n_hc, D)

        # Output injection: C * layer_output
        # C: (B, N, n_hc, 1), layer_output: (B, N, D) -> (B, N, 1, D)
        CF = C * layer_output.unsqueeze(-2)  # (B, N, n_hc, D)

        X_new = BX + CF

        return X_new

    def extract_output(self, X: torch.Tensor) -> torch.Tensor:
        """
        Extract the final hidden state from the expanded residual state.
        Simply takes the first slot.
        
        Args:
            X: (batch, seq_len, n_hc, hidden_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        return X[:, :, 0]
