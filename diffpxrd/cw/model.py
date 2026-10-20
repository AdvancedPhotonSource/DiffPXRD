from __future__ import annotations

from math import pi
from typing import Any

import numpy as np
import torch

from . import contracts as _cw_contracts
from . import layers as _cw_layers
from . import profile as _cw_profile
from . import scattering as _cw_scattering
from . import texture as _cw_texture
from ..xray_atomic_data import lookup_xray_form_factor_coeffs

ReflectionLayer = _cw_layers.ReflectionLayer
FrozenReflectionSet = _cw_layers.FrozenReflectionSet
ScatteringLayer = _cw_layers.ScatteringLayer
CorrectionLayer = _cw_layers.CorrectionLayer
ProfileLayer = _cw_layers.ProfileLayer
WeightedProfileLayer = _cw_layers.WeightedProfileLayer
MixtureProfileLayer = _cw_layers.MixtureProfileLayer
PeakGroupLayer = _cw_layers.PeakGroupLayer
ReflectionContractLayer = _cw_layers.ReflectionContractLayer
CompiledReflectionContract = _cw_contracts.CompiledReflectionContract
default_atomic_form_factor_constants = _cw_layers.default_atomic_form_factor_constants
CW_PROFILE_WIDTH_FLOOR_CENTIDEG = _cw_layers.CW_PROFILE_WIDTH_FLOOR_CENTIDEG

class ModularCWForwardModel:
    """
    Constant-wavelength powder X-ray diffraction forward-model facade.

    The model converts one or more ``pymatgen.Structure`` objects into tensor
    batches, enumerates reciprocal-space reflections over the configured
    q-range, computes x-ray structure factors with occupancies, thermal
    factors, and anomalous terms, applies CW diffraction corrections, and
    exposes FCJ, TCH-style, and simpler pseudo-Voigt profile builders.

    The default forward path re-enumerates reflections each call and is
    piecewise differentiable. Smooth-local refinement should use
    ``compile_reflections`` with ``forward_with_layers_fixed_hkls``.
    """

    def __init__(
        self,
        *,
        q_min: float,
        q_max: float,
        wavelength: float,
        atom_form_factor_constants: list[list[list[float]]] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.q_min = q_min
        self.q_max = q_max
        self.wavelength = wavelength
        self.atom_form_factor_constants = (
            atom_form_factor_constants
            if atom_form_factor_constants is not None
            else default_atomic_form_factor_constants()
        )
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._dispersion_cache: dict[tuple[int, float, torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _as_tensor_scalar(self, value: float | int | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(value):
            tensor = value.to(dtype=reference.dtype, device=reference.device)
            if tensor.ndim > 0 and reference.ndim > tensor.ndim and tensor.shape[0] == reference.shape[0]:
                while tensor.ndim < reference.ndim:
                    tensor = tensor.unsqueeze(-1)
            return tensor
        return torch.tensor(float(value), dtype=reference.dtype, device=reference.device)

    def structure_to_batch(
        self,
        structure: Any,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        """Read structure data and scattering coefficients at the requested precision."""
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        numpy_dtype = np.float64 if dtype == torch.float64 else np.float32
        frac_coords_array = np.array([site.frac_coords for site in structure], dtype=numpy_dtype)
        has_uaniso = any(site.properties.get("u_aniso") is not None for site in structure)
        site_uiso = np.array(
            [
                0.0 if site.properties.get("u_aniso") is not None else float(site.properties.get("u_iso", 0.0))
                for site in structure
            ],
            dtype=numpy_dtype,
        )
        site_uaniso_array = None
        if has_uaniso:
            site_uaniso_values: list[np.ndarray] = []
            for site in structure:
                raw_uaniso = site.properties.get("u_aniso")
                if raw_uaniso is None:
                    site_uaniso_values.append(np.zeros(6, dtype=numpy_dtype))
                    continue
                uaniso_array = np.asarray(raw_uaniso, dtype=numpy_dtype)
                if uaniso_array.shape == (3, 3):
                    uaniso_array = np.array(
                        [
                            uaniso_array[0, 0],
                            uaniso_array[1, 1],
                            uaniso_array[2, 2],
                            uaniso_array[0, 1],
                            uaniso_array[0, 2],
                            uaniso_array[1, 2],
                        ],
                        dtype=numpy_dtype,
                    )
                elif uaniso_array.size == 6:
                    uaniso_array = uaniso_array.reshape(6).astype(numpy_dtype)
                else:
                    raise ValueError("u_aniso site property must have shape (6,) or (3, 3)")
                site_uaniso_values.append(uaniso_array)
            site_uaniso_array = np.stack(site_uaniso_values, axis=0)
        max_species_per_site = max(len(site.species) for site in structure)
        site_species_zs = np.zeros((structure.num_sites, max_species_per_site), dtype=np.int64)
        site_species_occs = np.zeros((structure.num_sites, max_species_per_site), dtype=numpy_dtype)
        site_species_ff_a = np.zeros((structure.num_sites, max_species_per_site, 4), dtype=numpy_dtype)
        site_species_ff_b = np.zeros((structure.num_sites, max_species_per_site, 4), dtype=numpy_dtype)
        site_species_ff_c = np.zeros((structure.num_sites, max_species_per_site), dtype=numpy_dtype)
        dominant_atom_types = np.zeros(structure.num_sites, dtype=np.int64)
        for site_index, site in enumerate(structure):
            species_items = sorted(
                site.species.items(),
                key=lambda item: (-float(item[1]), int(item[0].Z)),
            )
            dominant_fa, dominant_fb, dominant_fc, dominant_z, _ = lookup_xray_form_factor_coeffs(
                species_items[0][0], dtype=numpy_dtype
            )
            dominant_atom_types[site_index] = int(dominant_z)
            for species_index, (species, occupancy) in enumerate(species_items):
                fa, fb, fc, atomic_number, _ = lookup_xray_form_factor_coeffs(species, dtype=numpy_dtype)
                site_species_zs[site_index, species_index] = int(atomic_number)
                site_species_occs[site_index, species_index] = float(occupancy)
                site_species_ff_a[site_index, species_index] = fa
                site_species_ff_b[site_index, species_index] = fb
                site_species_ff_c[site_index, species_index] = fc
        site_species_zs_tensor = torch.tensor(
            site_species_zs,
            dtype=torch.long,
            device=self.device,
        )
        site_species_fp, site_species_fpp = self.lookup_site_species_dispersion_terms(
            site_species_zs_tensor,
            wavelength=float(self.wavelength),
            dtype=dtype,
        )
        return {
            "lengths": torch.tensor(
                [[structure.lattice.a, structure.lattice.b, structure.lattice.c]],
                dtype=dtype,
                device=self.device,
            ),
            "angles": torch.tensor(
                [[structure.lattice.alpha, structure.lattice.beta, structure.lattice.gamma]],
                dtype=dtype,
                device=self.device,
            ),
            "frac_coords": torch.tensor(frac_coords_array, dtype=dtype, device=self.device),
            "num_atoms": torch.tensor([structure.num_sites], dtype=torch.long, device=self.device),
            "atom_types": torch.tensor(
                dominant_atom_types,
                dtype=torch.long,
                device=self.device,
            ),
            "site_uiso": torch.tensor(
                site_uiso,
                dtype=dtype,
                device=self.device,
            ),
            **(
                {
                    "site_uaniso": torch.tensor(
                        site_uaniso_array,
                        dtype=dtype,
                        device=self.device,
                    )
                }
                if site_uaniso_array is not None
                else {}
            ),
            "site_species_zs": site_species_zs_tensor,
            "site_species_occs": torch.tensor(
                site_species_occs,
                dtype=dtype,
                device=self.device,
            ),
            "site_species_ff_a": torch.tensor(
                site_species_ff_a,
                dtype=dtype,
                device=self.device,
            ),
            "site_species_ff_b": torch.tensor(
                site_species_ff_b,
                dtype=dtype,
                device=self.device,
            ),
            "site_species_ff_c": torch.tensor(
                site_species_ff_c,
                dtype=dtype,
                device=self.device,
            ),
            "site_species_fp": site_species_fp,
            "site_species_fpp": site_species_fpp,
        }

    def combine_batches(
        self,
        batches: list[dict[str, torch.Tensor]] | tuple[dict[str, torch.Tensor], ...],
    ) -> dict[str, torch.Tensor]:
        """Combine batches without reducing any input floating-point precision."""
        if len(batches) == 0:
            raise ValueError("at least one batch is required")
        if len(batches) == 1:
            return batches[0]

        reference_device = self.device
        float_dtype = batches[0]["lengths"].dtype
        floating_keys = (
            "lengths", "angles", "frac_coords", "site_uiso", "site_uaniso",
            "site_species_occs", "site_species_ff_a", "site_species_ff_b",
            "site_species_ff_c", "site_species_fp", "site_species_fpp",
        )
        for batch in batches:
            for key in floating_keys:
                if key in batch:
                    float_dtype = torch.promote_types(float_dtype, batch[key].dtype)
        lengths = torch.cat(
            [batch["lengths"].to(device=reference_device, dtype=float_dtype) for batch in batches],
            dim=0,
        )
        angles = torch.cat(
            [batch["angles"].to(device=reference_device, dtype=float_dtype) for batch in batches],
            dim=0,
        )
        num_atoms = torch.cat(
            [batch["num_atoms"].to(device=reference_device, dtype=torch.long) for batch in batches],
            dim=0,
        )
        frac_coords = torch.cat(
            [batch["frac_coords"].to(device=reference_device, dtype=float_dtype) for batch in batches],
            dim=0,
        )
        atom_types = torch.cat(
            [batch["atom_types"].to(device=reference_device, dtype=torch.long) for batch in batches],
            dim=0,
        )

        combined: dict[str, torch.Tensor] = {
            "lengths": lengths,
            "angles": angles,
            "num_atoms": num_atoms,
            "frac_coords": frac_coords,
            "atom_types": atom_types,
        }

        if any("site_uiso" in batch for batch in batches):
            site_uiso_parts = []
            for batch in batches:
                local_num_atoms = int(batch["frac_coords"].shape[0])
                if "site_uiso" in batch:
                    site_uiso_parts.append(batch["site_uiso"].to(device=reference_device, dtype=float_dtype))
                else:
                    site_uiso_parts.append(torch.zeros(local_num_atoms, dtype=float_dtype, device=reference_device))
            combined["site_uiso"] = torch.cat(site_uiso_parts, dim=0)

        if any("site_uaniso" in batch for batch in batches):
            site_uaniso_parts = []
            for batch in batches:
                local_num_atoms = int(batch["frac_coords"].shape[0])
                if "site_uaniso" in batch:
                    site_uaniso_parts.append(batch["site_uaniso"].to(device=reference_device, dtype=float_dtype))
                else:
                    site_uaniso_parts.append(torch.zeros((local_num_atoms, 6), dtype=float_dtype, device=reference_device))
            combined["site_uaniso"] = torch.cat(site_uaniso_parts, dim=0)

        max_species_per_site = max(
            int(batch["site_species_zs"].shape[1])
            for batch in batches
            if "site_species_zs" in batch
        )
        if max_species_per_site > 0:
            species_shapes = {
                "site_species_zs": (max_species_per_site,),
                "site_species_occs": (max_species_per_site,),
                "site_species_ff_a": (max_species_per_site, 4),
                "site_species_ff_b": (max_species_per_site, 4),
                "site_species_ff_c": (max_species_per_site,),
                "site_species_fp": (max_species_per_site,),
                "site_species_fpp": (max_species_per_site,),
            }
            species_dtypes = {
                "site_species_zs": torch.long,
                "site_species_occs": float_dtype,
                "site_species_ff_a": float_dtype,
                "site_species_ff_b": float_dtype,
                "site_species_ff_c": float_dtype,
                "site_species_fp": float_dtype,
                "site_species_fpp": float_dtype,
            }
            for key, suffix_shape in species_shapes.items():
                if not any(key in batch for batch in batches):
                    continue
                padded_parts = []
                for batch in batches:
                    local_num_atoms = int(batch["frac_coords"].shape[0])
                    target_shape = (local_num_atoms, *suffix_shape)
                    if key not in batch:
                        padded_parts.append(torch.zeros(target_shape, dtype=species_dtypes[key], device=reference_device))
                        continue
                    tensor = batch[key].to(device=reference_device, dtype=species_dtypes[key])
                    if tensor.shape[1:] == suffix_shape:
                        padded_parts.append(tensor)
                        continue
                    padded = torch.zeros(target_shape, dtype=species_dtypes[key], device=reference_device)
                    source_width = tensor.shape[1]
                    padded[:, :source_width, ...] = tensor
                    padded_parts.append(padded)
                combined[key] = torch.cat(padded_parts, dim=0)

        return combined

    def structures_to_batch(
        self,
        structures: list[Any] | tuple[Any, ...],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, torch.Tensor]:
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        if len(structures) == 0:
            raise ValueError("at least one structure is required")
        if len(structures) == 1:
            return self.structure_to_batch(structures[0], dtype=dtype)
        return self.combine_batches([self.structure_to_batch(structure, dtype=dtype) for structure in structures])

    def select_batch(
        self,
        batch: dict[str, torch.Tensor],
        indices: list[int] | tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        if indices is None:
            return batch

        start, stop = int(indices[0]), int(indices[1])
        num_atoms = batch["num_atoms"]
        atom_start = int(torch.sum(num_atoms[:start]).item())
        atom_stop = int(torch.sum(num_atoms[:stop]).item())
        selected = {
            "lengths": batch["lengths"][start:stop],
            "angles": batch["angles"][start:stop],
            "num_atoms": batch["num_atoms"][start:stop],
            "frac_coords": batch["frac_coords"][atom_start:atom_stop],
            "atom_types": batch["atom_types"][atom_start:atom_stop],
        }
        if "site_uiso" in batch:
            selected["site_uiso"] = batch["site_uiso"][atom_start:atom_stop]
        if "site_uaniso" in batch:
            selected["site_uaniso"] = batch["site_uaniso"][atom_start:atom_stop]
        for key in (
            "site_species_zs",
            "site_species_occs",
            "site_species_ff_a",
            "site_species_ff_b",
            "site_species_ff_c",
            "site_species_fp",
            "site_species_fpp",
        ):
            if key in batch:
                selected[key] = batch[key][atom_start:atom_stop]
        return selected

    def get_two_theta_bounds(self) -> tuple[float, float]:
        two_theta_min = np.arcsin((self.q_min * self.wavelength) / (4 * pi)) * 360 / pi
        two_theta_max = np.arcsin((self.q_max * self.wavelength) / (4 * pi)) * 360 / pi
        return two_theta_min, two_theta_max

    def build_pattern_domain(
        self,
        *,
        num_steps: int = 256,
        domain: torch.Tensor | np.ndarray | list[float] | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if domain is not None:
            if isinstance(domain, torch.Tensor):
                pattern_domain = domain.to(device=self.device, dtype=dtype)
            else:
                pattern_domain = torch.tensor(domain, dtype=dtype, device=self.device)
            if pattern_domain.ndim != 1:
                raise ValueError("pattern domain must be one-dimensional")
            return pattern_domain

        two_theta_min, two_theta_max = self.get_two_theta_bounds()
        step_size = (two_theta_max - two_theta_min) / num_steps
        return torch.arange(num_steps, dtype=dtype, device=self.device) * step_size + two_theta_min

    build_cell_matrix = _cw_scattering.build_cell_matrix
    get_points_in_sphere = _cw_scattering.get_points_in_sphere
    compute_reflection_layer = _cw_scattering.compute_reflection_layer
    compute_reflection_layer_from_hkls = _cw_scattering.compute_reflection_layer_from_hkls
    collect_scattering_inputs = _cw_scattering.collect_scattering_inputs
    calculate_atomic_scattering_factor = _cw_scattering.calculate_atomic_scattering_factor
    calculate_atomic_scattering_factor_approximate = _cw_scattering.calculate_atomic_scattering_factor_approximate
    calculate_isotropic_debye_waller_factor = _cw_scattering.calculate_isotropic_debye_waller_factor
    calculate_anisotropic_debye_waller_factor = _cw_scattering.calculate_anisotropic_debye_waller_factor
    lookup_site_species_dispersion_terms = _cw_scattering.lookup_site_species_dispersion_terms
    compute_exact_site_scattering_for_batch = _cw_scattering.compute_exact_site_scattering_for_batch
    compute_same_topology_scattering_layer = _cw_scattering.compute_same_topology_scattering_layer
    compute_segment_reduce_structure_factors = _cw_scattering.compute_segment_reduce_structure_factors
    compute_segment_reduce_scattering_layer = _cw_scattering.compute_segment_reduce_scattering_layer
    compute_scattering_layer = _cw_scattering.compute_scattering_layer

    calculate_lorentz_factor = _cw_texture.calculate_lorentz_factor
    calculate_march_dollase_factor = _cw_texture.calculate_march_dollase_factor
    _canonicalize_sample_symmetry = staticmethod(_cw_texture._canonicalize_sample_symmetry)
    _canonicalize_laue_group = staticmethod(_cw_texture._canonicalize_laue_group)
    _odf_check = staticmethod(_cw_texture._odf_check)
    get_texture_coefficient_names = staticmethod(_cw_texture.get_texture_coefficient_names)
    _prepare_texture_coefficients = _cw_texture._prepare_texture_coefficients
    _extract_cell_parameters = _cw_texture._extract_cell_parameters
    calculate_crystal_angles = _cw_texture.calculate_crystal_angles
    calculate_sample_angles = _cw_texture.calculate_sample_angles
    calculate_normalized_associated_legendre = _cw_texture.calculate_normalized_associated_legendre
    calculate_spherical_harmonic_angle = _cw_texture.calculate_spherical_harmonic_angle
    calculate_cubic_harmonic = _cw_texture.calculate_cubic_harmonic
    calculate_kcl = _cw_texture.calculate_kcl
    calculate_ksl = _cw_texture.calculate_ksl
    _resolve_texture_metadata = _cw_texture._resolve_texture_metadata
    calculate_sh_preferred_orientation_factor = _cw_texture.calculate_sh_preferred_orientation_factor
    calculate_topas_spherical_harmonic_component = _cw_texture.calculate_topas_spherical_harmonic_component
    calculate_topas_sh_preferred_orientation_factor = _cw_texture.calculate_topas_sh_preferred_orientation_factor
    calculate_sh_texture_factor = _cw_texture.calculate_sh_texture_factor
    calculate_preferred_orientation_factor = _cw_texture.calculate_preferred_orientation_factor

    _canonicalize_cw_profile_mode = _cw_profile._canonicalize_cw_profile_mode
    _expand_spacegroup_symbols = staticmethod(_cw_profile._expand_spacegroup_symbols)
    get_spacegroup_metadata_for_symbol = staticmethod(_cw_profile.get_spacegroup_metadata_for_symbol)
    get_mustrain_names = staticmethod(_cw_profile.get_mustrain_names)
    calculate_mustrain_terms = _cw_profile.calculate_mustrain_terms
    _resolve_generalized_mustrain_metadata = _cw_profile._resolve_generalized_mustrain_metadata
    _prepare_generalized_mustrain_coefficients = _cw_profile._prepare_generalized_mustrain_coefficients
    _expand_six_term_symmetric_matrix = staticmethod(_cw_profile._expand_six_term_symmetric_matrix)
    _prepare_generalized_size_coefficients = _cw_profile._prepare_generalized_size_coefficients
    _prepare_profile_geometry_tensor = _cw_profile._prepare_profile_geometry_tensor
    _prepare_profile_axis_tensor = _cw_profile._prepare_profile_axis_tensor
    calculate_reciprocal_axis_angles = _cw_profile.calculate_reciprocal_axis_angles
    calculate_cw_polarization_factor = _cw_profile.calculate_cw_polarization_factor
    calculate_cw_absorption_factor = _cw_profile.calculate_cw_absorption_factor
    calculate_cw_powder_extinction_factor = _cw_profile.calculate_cw_powder_extinction_factor
    compute_correction_layer = _cw_profile.compute_correction_layer
    caglioti_fwhm = _cw_profile.caglioti_fwhm
    build_pattern_tensor = _cw_profile.build_pattern_tensor
    compute_profile_layer_from_pattern = _cw_profile.compute_profile_layer_from_pattern
    compute_profile_layer = _cw_profile.compute_profile_layer
    compute_topas_pv_profile_layer_from_pattern = _cw_profile.compute_topas_pv_profile_layer_from_pattern
    get_cw_sig = _cw_profile.get_cw_sig
    get_cw_gam = _cw_profile.get_cw_gam
    get_cw_sample_profile_terms = _cw_profile.get_cw_sample_profile_terms
    get_cw_peak_window = _cw_profile.get_cw_peak_window
    apply_cw_position_corrections = _cw_profile.apply_cw_position_corrections
    apply_cw_position_corrections_to_pattern = _cw_profile.apply_cw_position_corrections_to_pattern
    apply_cw_intensity_corrections_to_pattern = _cw_profile.apply_cw_intensity_corrections_to_pattern
    compute_tch_profile_layer_from_pattern = _cw_profile.compute_tch_profile_layer_from_pattern
    compute_tch_profile_layer = _cw_profile.compute_tch_profile_layer
    compute_fcj_profile_layer_from_pattern = _cw_profile.compute_fcj_profile_layer_from_pattern
    compute_fcj_profile_layer = _cw_profile.compute_fcj_profile_layer
    compute_weighted_profile_layer = _cw_profile.compute_weighted_profile_layer

    combine_profile_layers = _cw_contracts.combine_profile_layers
    group_peak_list = _cw_contracts.group_peak_list
    compute_peak_group_layer = _cw_contracts.compute_peak_group_layer
    get_reciprocal_symmetry_rotations = staticmethod(_cw_contracts.get_reciprocal_symmetry_rotations)
    reciprocal_orbit = _cw_contracts.reciprocal_orbit
    compute_reciprocal_multiplicities = _cw_contracts.compute_reciprocal_multiplicities
    build_reflection_contract = _cw_contracts.build_reflection_contract
    compute_reflection_contract_layer = _cw_contracts.compute_reflection_contract_layer
    compile_reflection_contract = _cw_contracts.compile_reflection_contract
    apply_compiled_reflection_contract_single = _cw_contracts.apply_compiled_reflection_contract_single
    apply_compiled_reflection_contract_to_pattern = _cw_contracts.apply_compiled_reflection_contract_to_pattern

    def compile_reflections(
        self,
        structure_or_batch: Any,
    ) -> FrozenReflectionSet:
        """
        Enumerate and freeze the HKL list for smooth-local differentiable forwards.

        The returned set is tied to this model's configured ``q_min``/``q_max``.
        Lattice-dependent quantities are still recomputed from the input batch in
        ``forward_with_layers_fixed_hkls``; only the discrete HKL membership is
        frozen.
        """
        batch = (
            structure_or_batch
            if isinstance(structure_or_batch, dict)
            else self.structure_to_batch(structure_or_batch)
        )
        reflection_layer = self.compute_reflection_layer(batch)
        return FrozenReflectionSet(
            hkl_pts=reflection_layer.hkl_pts.detach().clone(),
            q_min=float(self.q_min),
            q_max=float(self.q_max),
        )

    def _select_fixed_hkl_tensor(
        self,
        batch: dict[str, torch.Tensor],
        selected_batch: dict[str, torch.Tensor],
        frozen_hkls: FrozenReflectionSet | torch.Tensor | np.ndarray | list[list[float]],
        indices: list[int] | tuple[int, int] | None,
    ) -> torch.Tensor | np.ndarray | list[list[float]]:
        if isinstance(frozen_hkls, FrozenReflectionSet):
            if abs(frozen_hkls.q_min - float(self.q_min)) > 1.0e-12 or abs(
                frozen_hkls.q_max - float(self.q_max)
            ) > 1.0e-12:
                raise ValueError(
                    "FrozenReflectionSet q-bounds do not match this model "
                    f"(frozen=({frozen_hkls.q_min}, {frozen_hkls.q_max}), "
                    f"model=({self.q_min}, {self.q_max}))"
                )
            hkl_pts = frozen_hkls.hkl_pts
        else:
            hkl_pts = frozen_hkls

        if not isinstance(hkl_pts, torch.Tensor):
            return hkl_pts

        if hkl_pts.ndim == 2:
            return hkl_pts
        if hkl_pts.ndim != 3:
            raise ValueError("fixed HKLs must have shape (n, 3) or (batch, n, 3)")

        selected_size = int(selected_batch["lengths"].shape[0]) if selected_batch["lengths"].ndim > 1 else 1
        if hkl_pts.shape[0] == selected_size:
            return hkl_pts

        batch_size = int(batch["lengths"].shape[0]) if batch["lengths"].ndim > 1 else 1
        if indices is not None and hkl_pts.shape[0] == batch_size:
            start, stop = int(indices[0]), int(indices[1])
            return hkl_pts[start:stop]

        raise ValueError("fixed HKL batch dimension must match the selected input batch size")

    def forward_with_layers(
        self,
        batch: dict[str, torch.Tensor],
        *,
        indices: list[int] | tuple[int, int] | None = None,
        uvw: torch.Tensor | None = None,
        num_steps: int = 256,
        fraction_gaussian: float = 0.9,
        domain: torch.Tensor | np.ndarray | list[float] | None = None,
    ) -> dict[str, Any]:
        selected_batch = self.select_batch(batch, indices)
        reflection_layer = self.compute_reflection_layer(selected_batch)
        scattering_layer = self.compute_scattering_layer(selected_batch, reflection_layer)
        correction_layer = self.compute_correction_layer(reflection_layer, scattering_layer)

        profile_layer = None
        if uvw is not None:
            if uvw.ndim == 1:
                uvw = uvw.unsqueeze(0)
            profile_layer = self.compute_profile_layer(
                correction_layer.pattern,
                uvw=uvw.to(device=self.device, dtype=correction_layer.pattern.dtype),
                num_steps=num_steps,
                fraction_gaussian=fraction_gaussian,
                domain=domain,
            )

        return {
            "batch": selected_batch,
            "reflection": reflection_layer,
            "scattering": scattering_layer,
            "correction": correction_layer,
            "profile": profile_layer,
        }

    def forward_with_layers_fixed_hkls(
        self,
        batch: dict[str, torch.Tensor],
        frozen_hkls: FrozenReflectionSet | torch.Tensor | np.ndarray | list[list[float]],
        *,
        indices: list[int] | tuple[int, int] | None = None,
        uvw: torch.Tensor | None = None,
        num_steps: int = 256,
        fraction_gaussian: float = 0.9,
        domain: torch.Tensor | np.ndarray | list[float] | None = None,
    ) -> dict[str, Any]:
        selected_batch = self.select_batch(batch, indices)
        selected_hkls = self._select_fixed_hkl_tensor(batch, selected_batch, frozen_hkls, indices)
        reflection_layer = self.compute_reflection_layer_from_hkls(selected_batch, selected_hkls)
        scattering_layer = self.compute_scattering_layer(selected_batch, reflection_layer)
        correction_layer = self.compute_correction_layer(reflection_layer, scattering_layer)

        profile_layer = None
        if uvw is not None:
            if uvw.ndim == 1:
                uvw = uvw.unsqueeze(0)
            profile_layer = self.compute_profile_layer(
                correction_layer.pattern,
                uvw=uvw.to(device=self.device, dtype=correction_layer.pattern.dtype),
                num_steps=num_steps,
                fraction_gaussian=fraction_gaussian,
                domain=domain,
            )

        return {
            "batch": selected_batch,
            "reflection": reflection_layer,
            "scattering": scattering_layer,
            "correction": correction_layer,
            "profile": profile_layer,
            "frozen_reflections": frozen_hkls,
        }
