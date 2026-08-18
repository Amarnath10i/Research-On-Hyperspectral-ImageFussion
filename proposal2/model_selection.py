"""Model-selection rule and the fundamental lower bound for P2.

This is the *method-facing* contribution that turns ``r_id`` from a diagnostic
into a tool:

    select_rank(r_hat, r_id_hat) = min(r_hat, r_id_hat)

Theoretical justification (see theory/model_selection.md).  The HR-HSI is
X = U_r Z with spectral basis U_r (B x r) and coefficients Z (r x HW); the
MSI observation is Y_M = R^T X = G Z with G = R^T U_r (M x r), rank(G) = r_id.
Y_M determines only the component of Z in the *row space* of G; the component
in ker(G) (dimension r - r_id) is structurally unobservable.  Therefore ANY
fusion method satisfies the lower bound

    || X_hat - X ||  >=  || U_r (I - G^dagger G) Z || / ||X||   (= LB)

and the bound is *achieved* iff the method recovers the observable part exactly
and adds nothing in the null space.  Choosing model rank r* = r_id is the
largest rank for which G Z = Y_M is well posed (full column rank), so no model
capacity is wasted inventing unidentifiable directions.  Choosing r* > r_id
opens an (r - r_id)-dimensional null space a prior must fill -> spectral
distortion; choosing r* < r_id sacrifices observable content -> error above LB.
Hence min(r_hat, r_id_hat) is the optimal, observability-aware rank.

NOTE: the scene-aware projection uses G = R^T U_r, NOT R alone.  R^T U_r (not
R) is the operator that maps the scene's spectral coefficients to the
observation, so ker(G) is the genuinely unobservable part of THIS scene.
"""

from __future__ import annotations

import torch


def select_rank(r_hat, r_id_hat) -> int:
    """Observability-aware model rank: never exceed what the sensors see."""
    return int(min(int(r_hat), int(r_id_hat)))


def _g_projectors(U: torch.Tensor, R: torch.Tensor):
    """G = R^T U_r and the row-space / null-space projectors of Z (r x r)."""
    G = R.t() @ U                                     # (M, r)
    P_row = torch.linalg.pinv(G) @ G                  # (r, r) onto row space
    P_null = torch.eye(G.shape[1], device=G.device) - P_row
    return G, P_row, P_null


def spectral_null_fraction(X: torch.Tensor, U: torch.Tensor,
                           R: torch.Tensor) -> float:
    """Irreducible spectral-error lower bound LB = ||U_r (I-G^dagger G) Z||/||X||."""
    G, _, P_null = _g_projectors(U, R)
    Xf = X.reshape(X.shape[0], -1).float()
    Z = U.t() @ Xf
    null = P_null @ Z
    return float(null.norm().item() / Xf.norm().clamp_min(1e-12).item())


def observable_reconstruction(X: torch.Tensor, U: torch.Tensor,
                              R: torch.Tensor) -> torch.Tensor:
    """Rank-r_id reconstruction: recover the row-space component of Z exactly.
    Achieves the lower bound LB."""
    G, P_row, _ = _g_projectors(U, R)
    Xf = X.reshape(X.shape[0], -1).float()
    Z = U.t() @ Xf
    Z_obs = P_row @ Z
    return (U @ Z_obs).reshape(X.shape)


def reconstruct_with_prior(X: torch.Tensor, U: torch.Tensor, R: torch.Tensor,
                           prior_Z: torch.Tensor) -> torch.Tensor:
    """Rank-r reconstruction that adds a (generic, wrong) null-space prior
    prior_Z (r,) to the recovered row-space component.  A wrong prior pushes
    error above LB -- the method is *inventing* unobservable spectra."""
    G, P_row, _ = _g_projectors(U, R)
    Xf = X.reshape(X.shape[0], -1).float()
    Z = U.t() @ Xf
    Z_full = P_row @ Z + prior_Z
    return (U @ Z_full).reshape(X.shape)


def true_null_component(X: torch.Tensor, U: torch.Tensor,
                        R: torch.Tensor) -> torch.Tensor:
    """The genuinely unobservable spectral content U_r (I - G^dagger G) Z."""
    G, _, P_null = _g_projectors(U, R)
    Xf = X.reshape(X.shape[0], -1).float()
    Z = U.t() @ Xf
    return (U @ (P_null @ Z)).reshape(X.shape)
