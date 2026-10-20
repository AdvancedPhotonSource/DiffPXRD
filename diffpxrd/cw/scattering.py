from __future__ import annotations

from math import pi
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from .layers import ReflectionLayer
from .layers import ScatteringLayer

from ..xray_atomic_data import lookup_xray_dispersion_terms

ScatteringBackend = Literal["legacy", "same_topology", "segment_reduce", "auto"]


def _canonicalize_scattering_backend(scattering_backend: str) -> str:
    backend = str(scattering_backend).strip().lower()
    valid = {"legacy", "same_topology", "segment_reduce", "auto"}
    if backend not in valid:
        raise ValueError(f"unsupported scattering_backend: {scattering_backend!r}")
    return backend


def build_cell_matrix(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    float_dtype = batch["lengths"].dtype
    unit_vectors = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=float_dtype,
        device=self.device,
    )
    lengths = batch["lengths"]
    angles = batch["angles"]

    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(0)
        angles = angles.unsqueeze(0)

    angles_rad = angles * pi / 180
    a_vector = torch.mul(lengths[:, 0].unsqueeze(1), unit_vectors[0])
    b_vector = torch.mul(
        lengths[:, 1].unsqueeze(1),
        torch.cos(angles_rad[:, 2]).unsqueeze(1) * unit_vectors[0]
        + torch.sin(angles_rad[:, 2]).unsqueeze(1) * unit_vectors[1],
    )
    c_vector = torch.mul(
        lengths[:, 2].unsqueeze(1),
        torch.cos(angles_rad[:, 1]).unsqueeze(1) * unit_vectors[0]
        + (
            (
                torch.cos(angles_rad[:, 0])
                - torch.cos(angles_rad[:, 1]) * torch.cos(angles_rad[:, 2])
            )
            / torch.sin(angles_rad[:, 2])
        ).unsqueeze(1)
        * unit_vectors[1]
        + torch.sqrt(
            1
            - torch.square(torch.cos(angles_rad[:, 1]))
            - torch.square(
                (
                    torch.cos(angles_rad[:, 0])
                    - torch.cos(angles_rad[:, 1]) * torch.cos(angles_rad[:, 2])
                )
                / torch.sin(angles_rad[:, 2])
            )
        ).unsqueeze(1)
        * unit_vectors[2],
    )
    return torch.stack((a_vector, b_vector, c_vector), dim=1)

def get_points_in_sphere(
    self,
    max_h: torch.Tensor,
    max_k: torch.Tensor,
    max_l: torch.Tensor,
    recip_latt: torch.Tensor,
) -> torch.Tensor:
    float_dtype = recip_latt.dtype
    max_h = torch.nan_to_num(max_h, nan=6.0)
    max_k = torch.nan_to_num(max_k, nan=6.0)
    max_l = torch.nan_to_num(max_l, nan=6.0)

    indices = torch.stack((max_h, max_k, max_l), dim=1)
    total_points_per_box = (2 * indices + 1).prod(dim=1) - 1
    _ = total_points_per_box.max(dim=0)[1]
    dims = [max_h.max(), max_k.max(), max_l.max()]
    ranges = [
        torch.arange(-dim.item(), dim.item() + 1, device=self.device, dtype=float_dtype)
        for dim in dims
    ]
    mesh = torch.meshgrid(*ranges, indexing="ij")
    coords = torch.stack(mesh, dim=-1).reshape(-1, 3)
    coords = coords.repeat(indices.size()[0], 1, 1)
    recip_pts = torch.matmul(coords, recip_latt)
    recip_lengths = torch.linalg.norm(recip_pts, dim=2)
    origin = torch.tensor([0, 0, 0], dtype=float_dtype, device=self.device)
    max_r = self.q_max / (2 * pi)
    coords = torch.where(
        (recip_lengths > max_r).unsqueeze(-1), origin, coords
    )
    sphere_mask = (self.wavelength * recip_lengths / 2) > 1
    coords = torch.where(sphere_mask.unsqueeze(-1), origin, coords)
    min_r = max(self.q_min / (2 * pi), 0.0)
    if min_r > 0.0:
        coords = torch.where(
            (recip_lengths < min_r).unsqueeze(-1), origin, coords
        )
    sphere_mask = (coords != origin).any(dim=2)

    max_points = max(len(coords[i][sphere_mask[i]]) for i in range(coords.shape[0]))
    hkl_pts = torch.zeros(
        (coords.size(0), max_points, 3),
        dtype=float_dtype,
        device=self.device,
    )
    for i in range(coords.shape[0]):
        filtered = coords[i][sphere_mask[i]]
        hkl_pts[i, : len(filtered)] = filtered
    return hkl_pts

def compute_reflection_layer(self, batch: dict[str, torch.Tensor]) -> ReflectionLayer:
    cell_matrix = self.build_cell_matrix(batch)
    recip_latt = torch.transpose(torch.linalg.inv(cell_matrix), 1, 2)
    lengths = batch["lengths"]
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(0)
    max_r = self.q_max / (2 * pi)
    # Size the HKL search box from the direct-cell lengths. Using reciprocal-axis
    # norms can underbound oblique cells and drop valid edge reflections before
    # the sphere filter runs.
    max_h = torch.round(max_r * lengths[:, 0])
    max_k = torch.round(max_r * lengths[:, 1])
    max_l = torch.round(max_r * lengths[:, 2])
    hkl_pts = self.get_points_in_sphere(max_h, max_k, max_l, recip_latt)
    recip_pts = torch.matmul(hkl_pts, recip_latt)
    recip_lengths = torch.linalg.norm(recip_pts, dim=2)
    return ReflectionLayer(
        cell_matrix=cell_matrix,
        hkl_pts=hkl_pts,
        recip_lengths=recip_lengths,
    )

def compute_reflection_layer_from_hkls(
    self,
    batch: dict[str, torch.Tensor],
    hkl_pts: torch.Tensor | np.ndarray | list[list[float]] | list[tuple[int, int, int]],
) -> ReflectionLayer:
    cell_matrix = self.build_cell_matrix(batch)
    recip_latt = torch.transpose(torch.linalg.inv(cell_matrix), 1, 2)

    if isinstance(hkl_pts, torch.Tensor):
        hkl_tensor = hkl_pts.to(device=self.device, dtype=cell_matrix.dtype)
    else:
        hkl_tensor = torch.tensor(hkl_pts, dtype=cell_matrix.dtype, device=self.device)

    if hkl_tensor.ndim == 2:
        hkl_tensor = hkl_tensor.unsqueeze(0).expand(cell_matrix.shape[0], -1, -1)
    elif hkl_tensor.ndim != 3:
        raise ValueError("hkl_pts must have shape (n, 3) or (batch, n, 3)")

    if hkl_tensor.shape[0] == 1 and cell_matrix.shape[0] != 1:
        hkl_tensor = hkl_tensor.expand(cell_matrix.shape[0], -1, -1)
    if hkl_tensor.shape[0] != cell_matrix.shape[0]:
        raise ValueError("hkl_pts batch dimension must match the input batch size")

    recip_pts = torch.matmul(hkl_tensor, recip_latt)
    recip_lengths = torch.linalg.norm(recip_pts, dim=2)
    return ReflectionLayer(
        cell_matrix=cell_matrix,
        hkl_pts=hkl_tensor,
        recip_lengths=recip_lengths,
    )

def collect_scattering_inputs(
    self,
    batch: dict[str, torch.Tensor],
    hkl_pts: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]:
    num_atoms = batch["num_atoms"]
    frac_coords = batch["frac_coords"]
    atom_types = batch["atom_types"]
    site_uiso = batch.get("site_uiso")
    site_uaniso = batch.get("site_uaniso")
    site_species_zs = batch.get("site_species_zs")
    site_species_occs = batch.get("site_species_occs")
    site_species_ff_a = batch.get("site_species_ff_a")
    site_species_ff_b = batch.get("site_species_ff_b")
    site_species_ff_c = batch.get("site_species_ff_c")
    site_species_fp = batch.get("site_species_fp")
    site_species_fpp = batch.get("site_species_fpp")
    cumulative_starts = torch.cat(
        (torch.tensor([0], device=self.device), num_atoms.cumsum(dim=0)[:-1])
    )
    total_sites = int(num_atoms.sum().item())
    float_dtype = frac_coords.dtype
    occus = None
    zs = None
    coeffs = None

    use_exact_site_coeffs = (
        site_species_zs is not None
        and site_species_occs is not None
        and site_species_ff_a is not None
        and site_species_ff_b is not None
        and site_species_ff_c is not None
    )

    if not use_exact_site_coeffs:
        if site_species_zs is not None and site_species_occs is not None:
            valid_species_mask = (site_species_zs > 0) & (site_species_occs > 0)
            zs = torch.unique(site_species_zs[valid_species_mask])
        else:
            zs = torch.unique(atom_types)
        coeffs = torch.tensor(
            self.atom_form_factor_constants,
            dtype=float_dtype,
            device=self.device,
        )[zs - 1]
        occus = torch.zeros(
            (num_atoms.size(0), total_sites, zs.numel()),
            dtype=float_dtype,
            device=self.device,
        )

        if site_species_zs is not None and site_species_occs is not None:
            global_site_indices = torch.arange(total_sites, device=self.device)
            for i in range(num_atoms.size(0)):
                start_idx = int(cumulative_starts[i].item())
                end_idx = int((cumulative_starts[i] + num_atoms[i]).item())
                local_site_indices = global_site_indices[start_idx:end_idx]
                local_species_zs = site_species_zs[start_idx:end_idx]
                local_species_occs = site_species_occs[start_idx:end_idx]
                for species_col in range(local_species_zs.size(1)):
                    column_zs = local_species_zs[:, species_col]
                    column_occs = local_species_occs[:, species_col]
                    valid = (column_zs > 0) & (column_occs > 0)
                    if not torch.any(valid):
                        continue
                    z_indices = torch.searchsorted(zs, column_zs[valid])
                    occus[i, local_site_indices[valid], z_indices] += column_occs[valid]
        else:
            atom_indices = torch.searchsorted(zs, atom_types)
            occus = (
                F.one_hot(atom_indices, num_classes=zs.numel())
                .to(dtype=float_dtype)
                .repeat(num_atoms.size()[0], 1, 1)
            )
            atoms_index_tensor = (
                torch.arange(total_sites, device=self.device)
                .unsqueeze(0)
                .expand(num_atoms.size(0), -1)
            )
            occus_mask = (
                (atoms_index_tensor >= cumulative_starts.unsqueeze(-1))
                & (
                    atoms_index_tensor
                    < cumulative_starts.unsqueeze(-1) + num_atoms.unsqueeze(-1)
                )
            )
            occus[~occus_mask] = 0

    g_dot_r = torch.zeros(
        (num_atoms.size(0), hkl_pts.size(1), total_sites),
        dtype=float_dtype,
        device=self.device,
    )
    for i in range(num_atoms.size(0)):
        start_idx = cumulative_starts[i]
        end_idx = start_idx + num_atoms[i]
        sliced_fcoords = frac_coords[start_idx:end_idx, :]
        g_dot_r[i, :, start_idx:end_idx] = torch.matmul(hkl_pts[i], sliced_fcoords.T)

    return (
        frac_coords,
        site_uiso,
        site_uaniso,
        occus,
        zs,
        coeffs,
        site_species_ff_a,
        site_species_ff_b,
        site_species_ff_c,
        site_species_fp,
        site_species_fpp,
        g_dot_r,
    )

def calculate_atomic_scattering_factor(
    self,
    recip_lengths: torch.Tensor,
    ff_a: torch.Tensor,
    ff_b: torch.Tensor,
    ff_c: torch.Tensor,
) -> torch.Tensor:
    squared_recip_lengths = torch.square(recip_lengths / 2).reshape(-1, 1, 1, 1)
    return torch.sum(
        ff_a.unsqueeze(0) * torch.exp(-ff_b.unsqueeze(0) * squared_recip_lengths),
        dim=-1,
    ) + ff_c.unsqueeze(0)

def calculate_atomic_scattering_factor_approximate(
    self,
    zs: torch.Tensor,
    recip_lengths: torch.Tensor,
    coeffs: torch.Tensor,
) -> torch.Tensor:
    squared_recip_lengths = torch.square(recip_lengths / 2).unsqueeze(1)
    extinct_exp = torch.exp(-coeffs[:, :, 1].unsqueeze(2) * squared_recip_lengths.unsqueeze(1))
    fitted_factor = torch.sum(coeffs[:, :, 0].unsqueeze(2) * extinct_exp, axis=2)
    angle_fs = 41.78214 * squared_recip_lengths * fitted_factor
    return zs.unsqueeze(0).unsqueeze(2) - angle_fs

def calculate_isotropic_debye_waller_factor(
    self,
    recip_lengths: torch.Tensor,
    site_uiso: torch.Tensor,
) -> torch.Tensor:
    squared_sthovl = torch.square(recip_lengths.reshape(-1, 1) / 2.0)
    return torch.exp((-8.0 * (pi ** 2)) * squared_sthovl * site_uiso.reshape(1, -1))

def calculate_anisotropic_debye_waller_factor(
    self,
    hkl_pts: torch.Tensor,
    cell_matrix: torch.Tensor,
    site_uaniso: torch.Tensor,
) -> torch.Tensor:
    real_metric = torch.matmul(cell_matrix, torch.transpose(cell_matrix, 0, 1))
    recip_metric = torch.linalg.inv(real_metric)
    ast = torch.sqrt(torch.clamp(torch.diagonal(recip_metric), min=1.0e-12))
    mast = torch.outer(ast, ast)
    uaniso_tensor = self._expand_six_term_symmetric_matrix(site_uaniso)
    beta_tensor = mast.unsqueeze(0) * uaniso_tensor
    hbih = torch.einsum("ni,sij,nj->ns", hkl_pts, beta_tensor, hkl_pts)
    return torch.exp((-2.0 * (pi ** 2)) * hbih)

def lookup_site_species_dispersion_terms(
    self,
    site_species_zs: torch.Tensor,
    wavelength: float,
    *,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype is None:
        dtype = torch.get_default_dtype()
    fp = torch.zeros_like(site_species_zs, dtype=dtype, device=self.device)
    fpp = torch.zeros_like(site_species_zs, dtype=dtype, device=self.device)
    unique_zs = torch.unique(site_species_zs[site_species_zs > 0]).detach().cpu().tolist()
    for atomic_number in unique_zs:
        key = (int(atomic_number), float(wavelength), dtype, self.device)
        if not hasattr(self, "_dispersion_cache"):
            self._dispersion_cache = {}
        if key not in self._dispersion_cache:
            fp_value, fpp_value = lookup_xray_dispersion_terms(int(atomic_number), float(wavelength))
            self._dispersion_cache[key] = (
                torch.tensor(float(fp_value), dtype=dtype, device=self.device),
                torch.tensor(float(fpp_value), dtype=dtype, device=self.device),
            )
        cached_fp, cached_fpp = self._dispersion_cache[key]
        mask = site_species_zs == int(atomic_number)
        fp[mask] = cached_fp
        fpp[mask] = cached_fpp
    return fp, fpp


def _validate_same_topology_scattering_inputs(
    batch: dict[str, torch.Tensor],
    reflection_layer: ReflectionLayer,
    *,
    store_intermediates: bool,
    require_marker: bool = False,
) -> int:
    if require_marker and batch.get("_same_topology_repeated") is not True:
        raise ValueError("same_topology auto backend requires a repeated-topology batch marker")
    if store_intermediates:
        raise ValueError("same_topology scattering does not support store_intermediates=True")

    lengths = batch["lengths"]
    batch_size = int(lengths.shape[0]) if lengths.ndim > 1 else 1
    if batch_size <= 1:
        raise ValueError("same_topology scattering requires batch size greater than one")
    if reflection_layer.hkl_pts.shape[0] != batch_size:
        raise ValueError("same_topology scattering requires HKLs batched to match lengths")
    if reflection_layer.recip_lengths.shape[0] != batch_size:
        raise ValueError("same_topology scattering requires reciprocal lengths batched to match lengths")

    num_atoms = batch["num_atoms"]
    if num_atoms.ndim != 1 or int(num_atoms.shape[0]) != batch_size:
        raise ValueError("same_topology scattering requires one num_atoms entry per batch item")
    if not bool(torch.all(num_atoms == num_atoms[0]).detach().cpu().item()):
        raise ValueError("same_topology scattering requires equal site counts per batch item")
    site_count = int(num_atoms[0].detach().cpu().item())
    if site_count <= 0:
        raise ValueError("same_topology scattering requires at least one site per batch item")
    expected_sites = batch_size * site_count
    if int(batch["frac_coords"].shape[0]) != expected_sites:
        raise ValueError("same_topology scattering requires frac_coords to have shape (B*S, 3)")

    required_keys = (
        "site_species_zs",
        "site_species_occs",
        "site_species_ff_a",
        "site_species_ff_b",
        "site_species_ff_c",
    )
    missing = [key for key in required_keys if batch.get(key) is None]
    if missing:
        raise ValueError(f"same_topology scattering requires exact site species arrays, missing {missing}")
    for key in required_keys + ("site_species_fp", "site_species_fpp"):
        tensor = batch.get(key)
        if tensor is not None and int(tensor.shape[0]) != expected_sites:
            raise ValueError(f"same_topology scattering requires {key} to have first dimension B*S")
    if batch.get("site_uiso") is not None and int(batch["site_uiso"].shape[0]) != expected_sites:
        raise ValueError("same_topology scattering requires site_uiso to have first dimension B*S")
    if batch.get("site_uaniso") is not None:
        raise ValueError("same_topology scattering does not yet support anisotropic displacement tensors")
    return site_count


def _can_use_same_topology_scattering(
    batch: dict[str, torch.Tensor],
    reflection_layer: ReflectionLayer,
    *,
    store_intermediates: bool,
) -> bool:
    try:
        _validate_same_topology_scattering_inputs(
            batch,
            reflection_layer,
            store_intermediates=store_intermediates,
            require_marker=True,
        )
    except ValueError:
        return False
    return True


def _validate_segment_reduce_scattering_inputs(
    batch: dict[str, torch.Tensor],
    reflection_layer: ReflectionLayer,
    *,
    store_intermediates: bool,
    require_batch: bool = False,
) -> int:
    if store_intermediates:
        raise ValueError("segment_reduce scattering does not support store_intermediates=True")

    lengths = batch["lengths"]
    batch_size = int(lengths.shape[0]) if lengths.ndim > 1 else 1
    if require_batch and batch_size <= 1:
        raise ValueError("segment_reduce auto backend requires batch size greater than one")
    if reflection_layer.hkl_pts.shape[0] != batch_size:
        raise ValueError("segment_reduce scattering requires HKLs batched to match lengths")
    if reflection_layer.recip_lengths.shape[0] != batch_size:
        raise ValueError("segment_reduce scattering requires reciprocal lengths batched to match lengths")
    if reflection_layer.cell_matrix.shape[0] != batch_size:
        raise ValueError("segment_reduce scattering requires cell matrices batched to match lengths")

    num_atoms = batch["num_atoms"]
    if num_atoms.ndim != 1 or int(num_atoms.shape[0]) != batch_size:
        raise ValueError("segment_reduce scattering requires one num_atoms entry per batch item")
    total_sites = int(num_atoms.sum().detach().cpu().item())
    if total_sites <= 0:
        raise ValueError("segment_reduce scattering requires at least one site")
    if int(batch["frac_coords"].shape[0]) != total_sites:
        raise ValueError("segment_reduce scattering requires frac_coords to have first dimension total_sites")

    required_keys = (
        "site_species_zs",
        "site_species_occs",
        "site_species_ff_a",
        "site_species_ff_b",
        "site_species_ff_c",
    )
    missing = [key for key in required_keys if batch.get(key) is None]
    if missing:
        raise ValueError(f"segment_reduce scattering requires exact site species arrays, missing {missing}")
    for key in required_keys + ("site_species_fp", "site_species_fpp"):
        tensor = batch.get(key)
        if tensor is not None and int(tensor.shape[0]) != total_sites:
            raise ValueError(f"segment_reduce scattering requires {key} to have first dimension total_sites")
    if batch.get("site_uiso") is not None and int(batch["site_uiso"].shape[0]) != total_sites:
        raise ValueError("segment_reduce scattering requires site_uiso to have first dimension total_sites")
    if batch.get("site_uaniso") is not None:
        site_uaniso = batch["site_uaniso"]
        if int(site_uaniso.shape[0]) != total_sites or site_uaniso.shape[-1] != 6:
            raise ValueError("segment_reduce scattering requires site_uaniso to have shape (total_sites, 6)")
    return total_sites


def _can_use_segment_reduce_scattering(
    batch: dict[str, torch.Tensor],
    reflection_layer: ReflectionLayer,
    *,
    store_intermediates: bool,
) -> bool:
    try:
        _validate_segment_reduce_scattering_inputs(
            batch,
            reflection_layer,
            store_intermediates=store_intermediates,
            require_batch=True,
        )
    except ValueError:
        return False
    return True


def compute_exact_site_scattering_for_batch(
    self,
    recip_lengths: torch.Tensor,
    hkl_pts: torch.Tensor,
    cell_matrix: torch.Tensor,
    g_dot_r: torch.Tensor,
    site_species_ff_a: torch.Tensor,
    site_species_ff_b: torch.Tensor,
    site_species_ff_c: torch.Tensor,
    site_species_occs: torch.Tensor,
    site_species_zs: torch.Tensor,
    site_uiso: torch.Tensor | None,
    site_uaniso: torch.Tensor | None,
    *,
    wavelength: float,
    site_species_fp: torch.Tensor | None = None,
    site_species_fpp: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    atomic_scattering = self.calculate_atomic_scattering_factor(
        recip_lengths,
        site_species_ff_a,
        site_species_ff_b,
        site_species_ff_c,
    )
    if site_species_fp is None or site_species_fpp is None:
        site_species_fp, site_species_fpp = self.lookup_site_species_dispersion_terms(
            site_species_zs,
            wavelength,
            dtype=site_species_occs.dtype,
        )
    else:
        site_species_fp = site_species_fp.to(dtype=site_species_occs.dtype, device=self.device)
        site_species_fpp = site_species_fpp.to(dtype=site_species_occs.dtype, device=self.device)
    complex_atomic_scattering = torch.complex(
        atomic_scattering + site_species_fp.unsqueeze(0),
        site_species_fpp.unsqueeze(0),
    )
    weighted_scattering = torch.sum(
        complex_atomic_scattering * site_species_occs.unsqueeze(0),
        dim=2,
    )
    if site_uiso is not None:
        weighted_scattering = weighted_scattering * self.calculate_isotropic_debye_waller_factor(
            recip_lengths,
            site_uiso,
        )
    if site_uaniso is not None:
        weighted_scattering = weighted_scattering * self.calculate_anisotropic_debye_waller_factor(
            hkl_pts,
            cell_matrix,
            site_uaniso,
        )
    structure_factor = torch.sum(
        weighted_scattering * torch.exp(2j * pi * g_dot_r),
        dim=1,
    )
    return atomic_scattering, weighted_scattering, structure_factor


def compute_same_topology_scattering_layer(
    self,
    batch: dict[str, torch.Tensor],
    reflection_layer: ReflectionLayer,
) -> ScatteringLayer:
    site_count = _validate_same_topology_scattering_inputs(
        batch,
        reflection_layer,
        store_intermediates=False,
        require_marker=False,
    )
    batch_size = int(batch["lengths"].shape[0])
    float_dtype = reflection_layer.recip_lengths.dtype
    species_count = int(batch["site_species_occs"].shape[1])

    frac_coords_local = batch["frac_coords"].reshape(batch_size, site_count, 3)
    site_species_ff_a = batch["site_species_ff_a"].reshape(batch_size, site_count, species_count, 4).to(
        dtype=float_dtype,
        device=self.device,
    )
    site_species_ff_b = batch["site_species_ff_b"].reshape(batch_size, site_count, species_count, 4).to(
        dtype=float_dtype,
        device=self.device,
    )
    site_species_ff_c = batch["site_species_ff_c"].reshape(batch_size, site_count, species_count).to(
        dtype=float_dtype,
        device=self.device,
    )
    site_species_occs = batch["site_species_occs"].reshape(batch_size, site_count, species_count).to(
        dtype=float_dtype,
        device=self.device,
    )
    site_species_fp = batch.get("site_species_fp")
    site_species_fpp = batch.get("site_species_fpp")
    if site_species_fp is None or site_species_fpp is None:
        site_species_fp, site_species_fpp = self.lookup_site_species_dispersion_terms(
            batch["site_species_zs"],
            wavelength=float(self.wavelength),
            dtype=float_dtype,
        )
    site_species_fp = site_species_fp.reshape(batch_size, site_count, species_count).to(
        dtype=float_dtype,
        device=self.device,
    )
    site_species_fpp = site_species_fpp.reshape(batch_size, site_count, species_count).to(
        dtype=float_dtype,
        device=self.device,
    )

    g_dot_r = torch.einsum("bnh,bsh->bns", reflection_layer.hkl_pts, frac_coords_local)
    squared_sthovl = torch.square(reflection_layer.recip_lengths / 2.0)
    atomic_scattering = (
        site_species_ff_a[:, None, :, :, :]
        * torch.exp(-site_species_ff_b[:, None, :, :, :] * squared_sthovl[:, :, None, None, None])
    ).sum(dim=-1) + site_species_ff_c[:, None, :, :]
    complex_atomic_scattering = torch.complex(
        atomic_scattering + site_species_fp[:, None, :, :],
        site_species_fpp[:, None, :, :].expand_as(atomic_scattering),
    )
    weighted_scattering = torch.sum(
        complex_atomic_scattering * site_species_occs[:, None, :, :],
        dim=-1,
    )
    site_uiso = batch.get("site_uiso")
    if site_uiso is not None:
        site_uiso_local = site_uiso.reshape(batch_size, site_count).to(dtype=float_dtype, device=self.device)
        weighted_scattering = weighted_scattering * torch.exp(
            (-8.0 * (pi ** 2)) * squared_sthovl[:, :, None] * site_uiso_local[:, None, :]
        )
    structure_factors = torch.sum(
        weighted_scattering * torch.exp(2j * pi * g_dot_r),
        dim=-1,
    )
    base_intensities = torch.square(structure_factors.real) + torch.square(structure_factors.imag)
    return ScatteringLayer(
        frac_coords=batch["frac_coords"],
        g_dot_r=g_dot_r,
        num_atoms=batch["num_atoms"],
        wavelength=float(self.wavelength),
        phase_layout="same_topology_local",
        site_uiso=site_uiso,
        site_uaniso=None,
        site_species_occs=batch.get("site_species_occs"),
        site_species_zs=batch.get("site_species_zs"),
        site_species_ff_a=batch.get("site_species_ff_a"),
        site_species_ff_b=batch.get("site_species_ff_b"),
        site_species_ff_c=batch.get("site_species_ff_c"),
        site_species_fp=batch.get("site_species_fp"),
        site_species_fpp=batch.get("site_species_fpp"),
        atomic_scattering_factors=None,
        weighted_scattering_factors=None,
        structure_factors=structure_factors,
        base_intensities=base_intensities,
    )


def compute_segment_reduce_structure_factors(
    self,
    reflection_layer: ReflectionLayer,
    *,
    num_atoms: torch.Tensor,
    frac_coords: torch.Tensor,
    site_species_ff_a: torch.Tensor,
    site_species_ff_b: torch.Tensor,
    site_species_ff_c: torch.Tensor,
    site_species_occs: torch.Tensor,
    site_species_zs: torch.Tensor,
    site_uiso: torch.Tensor | None,
    site_uaniso: torch.Tensor | None,
    wavelength: float,
    site_species_fp: torch.Tensor | None = None,
    site_species_fpp: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(num_atoms.shape[0])
    float_dtype = reflection_layer.recip_lengths.dtype
    site_batch_id = torch.repeat_interleave(
        torch.arange(batch_size, device=self.device, dtype=torch.long),
        num_atoms.to(device=self.device, dtype=torch.long),
    )
    site_hkls = reflection_layer.hkl_pts.index_select(0, site_batch_id)
    site_recip_lengths = reflection_layer.recip_lengths.index_select(0, site_batch_id)
    frac_coords = frac_coords.to(dtype=float_dtype, device=self.device)
    g_dot_r = torch.einsum("snh,sh->sn", site_hkls, frac_coords)

    site_species_ff_a = site_species_ff_a.to(dtype=float_dtype, device=self.device)
    site_species_ff_b = site_species_ff_b.to(dtype=float_dtype, device=self.device)
    site_species_ff_c = site_species_ff_c.to(dtype=float_dtype, device=self.device)
    site_species_occs = site_species_occs.to(dtype=float_dtype, device=self.device)
    if site_species_fp is None or site_species_fpp is None:
        site_species_fp, site_species_fpp = self.lookup_site_species_dispersion_terms(
            site_species_zs,
            wavelength=float(wavelength),
            dtype=float_dtype,
        )
    site_species_fp = site_species_fp.to(dtype=float_dtype, device=self.device)
    site_species_fpp = site_species_fpp.to(dtype=float_dtype, device=self.device)

    squared_sthovl = torch.square(site_recip_lengths / 2.0)
    atomic_scattering = (
        site_species_ff_a[:, None, :, :]
        * torch.exp(-site_species_ff_b[:, None, :, :] * squared_sthovl[:, :, None, None])
    ).sum(dim=-1) + site_species_ff_c[:, None, :]
    complex_atomic_scattering = torch.complex(
        atomic_scattering + site_species_fp[:, None, :],
        site_species_fpp[:, None, :].expand_as(atomic_scattering),
    )
    weighted_scattering = torch.sum(
        complex_atomic_scattering * site_species_occs[:, None, :],
        dim=-1,
    )

    if site_uiso is not None:
        site_uiso_float = site_uiso.to(dtype=float_dtype, device=self.device)
        weighted_scattering = weighted_scattering * torch.exp(
            (-8.0 * (pi ** 2)) * squared_sthovl * site_uiso_float[:, None]
        )
    if site_uaniso is not None:
        site_cell_matrix = reflection_layer.cell_matrix.index_select(0, site_batch_id)
        real_metric = torch.matmul(site_cell_matrix, site_cell_matrix.transpose(-1, -2))
        recip_metric = torch.linalg.inv(real_metric)
        ast = torch.sqrt(torch.clamp(torch.diagonal(recip_metric, dim1=-2, dim2=-1), min=1.0e-12))
        mast = ast[:, :, None] * ast[:, None, :]
        uaniso_tensor = self._expand_six_term_symmetric_matrix(
            site_uaniso.to(dtype=float_dtype, device=self.device)
        )
        beta_tensor = mast * uaniso_tensor
        hbih = torch.einsum("sni,sij,snj->sn", site_hkls, beta_tensor, site_hkls)
        weighted_scattering = weighted_scattering * torch.exp((-2.0 * (pi ** 2)) * hbih)

    site_contributions = weighted_scattering * torch.exp(2j * pi * g_dot_r)
    structure_factors = site_contributions.new_zeros(
        (batch_size, reflection_layer.hkl_pts.shape[1])
    )
    structure_factors.index_add_(0, site_batch_id, site_contributions)
    return g_dot_r, structure_factors


def compute_segment_reduce_scattering_layer(
    self,
    batch: dict[str, torch.Tensor],
    reflection_layer: ReflectionLayer,
) -> ScatteringLayer:
    _validate_segment_reduce_scattering_inputs(
        batch,
        reflection_layer,
        store_intermediates=False,
        require_batch=False,
    )
    num_atoms = batch["num_atoms"]
    site_uiso = batch.get("site_uiso")
    site_uaniso = batch.get("site_uaniso")
    g_dot_r, structure_factors = self.compute_segment_reduce_structure_factors(
        reflection_layer,
        num_atoms=num_atoms,
        frac_coords=batch["frac_coords"],
        site_species_ff_a=batch["site_species_ff_a"],
        site_species_ff_b=batch["site_species_ff_b"],
        site_species_ff_c=batch["site_species_ff_c"],
        site_species_occs=batch["site_species_occs"],
        site_species_zs=batch["site_species_zs"],
        site_uiso=site_uiso,
        site_uaniso=site_uaniso,
        wavelength=float(self.wavelength),
        site_species_fp=batch.get("site_species_fp"),
        site_species_fpp=batch.get("site_species_fpp"),
    )
    base_intensities = torch.square(structure_factors.real) + torch.square(structure_factors.imag)
    return ScatteringLayer(
        frac_coords=batch["frac_coords"],
        g_dot_r=g_dot_r,
        num_atoms=num_atoms,
        wavelength=float(self.wavelength),
        phase_layout="segment_reduce_sites",
        site_uiso=site_uiso,
        site_uaniso=site_uaniso,
        site_species_occs=batch.get("site_species_occs"),
        site_species_zs=batch.get("site_species_zs"),
        site_species_ff_a=batch.get("site_species_ff_a"),
        site_species_ff_b=batch.get("site_species_ff_b"),
        site_species_ff_c=batch.get("site_species_ff_c"),
        site_species_fp=batch.get("site_species_fp"),
        site_species_fpp=batch.get("site_species_fpp"),
        atomic_scattering_factors=None,
        weighted_scattering_factors=None,
        structure_factors=structure_factors,
        base_intensities=base_intensities,
    )


def compute_scattering_layer(
    self,
    batch: dict[str, torch.Tensor],
    reflection_layer: ReflectionLayer,
    *,
    store_intermediates: bool = False,
    scattering_backend: ScatteringBackend | str = "auto",
) -> ScatteringLayer:
    backend = _canonicalize_scattering_backend(scattering_backend)
    if backend == "segment_reduce":
        _validate_segment_reduce_scattering_inputs(
            batch,
            reflection_layer,
            store_intermediates=store_intermediates,
            require_batch=False,
        )
        return self.compute_segment_reduce_scattering_layer(batch, reflection_layer)
    if backend == "same_topology":
        _validate_same_topology_scattering_inputs(
            batch,
            reflection_layer,
            store_intermediates=store_intermediates,
            require_marker=False,
        )
        return self.compute_same_topology_scattering_layer(batch, reflection_layer)
    if backend == "auto" and _can_use_same_topology_scattering(
        batch,
        reflection_layer,
        store_intermediates=store_intermediates,
    ):
        return self.compute_same_topology_scattering_layer(batch, reflection_layer)
    if backend == "auto" and _can_use_segment_reduce_scattering(
        batch,
        reflection_layer,
        store_intermediates=store_intermediates,
    ):
        return self.compute_segment_reduce_scattering_layer(batch, reflection_layer)

    (
        frac_coords,
        site_uiso,
        site_uaniso,
        occus,
        zs,
        coeffs,
        site_species_ff_a,
        site_species_ff_b,
        site_species_ff_c,
        site_species_fp,
        site_species_fpp,
        g_dot_r,
    ) = self.collect_scattering_inputs(
        batch,
        reflection_layer.hkl_pts,
    )
    num_atoms = batch["num_atoms"]
    cumulative_starts = torch.cat(
        (torch.tensor([0], device=self.device), num_atoms.cumsum(dim=0)[:-1])
    )

    if (
        site_species_ff_a is not None
        and site_species_ff_b is not None
        and site_species_ff_c is not None
        and batch.get("site_species_occs") is not None
        and batch.get("site_species_zs") is not None
    ):
        complex_dtype = (
            torch.complex128
            if reflection_layer.hkl_pts.dtype == torch.float64
            else torch.complex64
        )
        structure_factors = torch.zeros(
            (num_atoms.size(0), reflection_layer.hkl_pts.size(1)),
            dtype=complex_dtype,
            device=self.device,
        )
        weighted_scattering_factors = (
            torch.zeros(
                (num_atoms.size(0), reflection_layer.hkl_pts.size(1), int(num_atoms.sum().item())),
                dtype=complex_dtype,
                device=self.device,
            )
            if store_intermediates
            else None
        )
        atomic_scattering_factors = None
        for i in range(num_atoms.size(0)):
            start_idx = int(cumulative_starts[i].item())
            end_idx = int((cumulative_starts[i] + num_atoms[i]).item())
            local_ff_a = site_species_ff_a[start_idx:end_idx]
            local_ff_b = site_species_ff_b[start_idx:end_idx]
            local_ff_c = site_species_ff_c[start_idx:end_idx]
            local_occs = batch["site_species_occs"][start_idx:end_idx]
            local_zs = batch["site_species_zs"][start_idx:end_idx]
            local_fp = site_species_fp[start_idx:end_idx] if site_species_fp is not None else None
            local_fpp = site_species_fpp[start_idx:end_idx] if site_species_fpp is not None else None
            local_uiso = site_uiso[start_idx:end_idx] if site_uiso is not None else None
            local_uaniso = site_uaniso[start_idx:end_idx] if site_uaniso is not None else None
            _, local_weighted_scattering, local_structure_factor = (
                self.compute_exact_site_scattering_for_batch(
                    reflection_layer.recip_lengths[i],
                    reflection_layer.hkl_pts[i],
                    reflection_layer.cell_matrix[i],
                    g_dot_r[i, :, start_idx:end_idx],
                    local_ff_a,
                    local_ff_b,
                    local_ff_c,
                    local_occs,
                    local_zs,
                    local_uiso,
                    local_uaniso,
                    wavelength=self.wavelength,
                    site_species_fp=local_fp,
                    site_species_fpp=local_fpp,
                )
            )
            if weighted_scattering_factors is not None:
                weighted_scattering_factors[i, :, start_idx:end_idx] = local_weighted_scattering
            structure_factors[i] = local_structure_factor
    else:
        approximate_atomic_scattering_factors = self.calculate_atomic_scattering_factor_approximate(
            zs,
            reflection_layer.recip_lengths,
            coeffs,
        )
        approximate_weighted_scattering_factors = torch.matmul(occus, approximate_atomic_scattering_factors)
        structure_factors = torch.sum(
            torch.transpose(approximate_weighted_scattering_factors, 1, 2)
            * torch.exp(2j * pi * g_dot_r),
            axis=2,
        )
        atomic_scattering_factors = (
            approximate_atomic_scattering_factors if store_intermediates else None
        )
        weighted_scattering_factors = (
            approximate_weighted_scattering_factors if store_intermediates else None
        )
    base_intensities = torch.square(structure_factors.real) + torch.square(structure_factors.imag)
    return ScatteringLayer(
        frac_coords=frac_coords,
        g_dot_r=g_dot_r,
        num_atoms=num_atoms,
        wavelength=float(self.wavelength),
        phase_layout="legacy_global",
        site_uiso=site_uiso,
        site_uaniso=site_uaniso,
        site_species_occs=batch.get("site_species_occs"),
        site_species_zs=batch.get("site_species_zs"),
        site_species_ff_a=site_species_ff_a,
        site_species_ff_b=site_species_ff_b,
        site_species_ff_c=site_species_ff_c,
        site_species_fp=batch.get("site_species_fp"),
        site_species_fpp=batch.get("site_species_fpp"),
        atomic_scattering_factors=atomic_scattering_factors,
        weighted_scattering_factors=weighted_scattering_factors,
        structure_factors=structure_factors,
        base_intensities=base_intensities,
    )

__all__ = [
    "ScatteringBackend",
    "build_cell_matrix",
    "get_points_in_sphere",
    "compute_reflection_layer",
    "compute_reflection_layer_from_hkls",
    "collect_scattering_inputs",
    "calculate_atomic_scattering_factor",
    "calculate_atomic_scattering_factor_approximate",
    "calculate_isotropic_debye_waller_factor",
    "calculate_anisotropic_debye_waller_factor",
    "lookup_site_species_dispersion_terms",
    "compute_exact_site_scattering_for_batch",
    "compute_same_topology_scattering_layer",
    "compute_segment_reduce_structure_factors",
    "compute_segment_reduce_scattering_layer",
    "compute_scattering_layer",
]
