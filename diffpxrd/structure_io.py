from __future__ import annotations

from math import pi
from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.cif import str2float

def _parse_cif_float(value: Any) -> float | None:
    try:
        return float(str2float(str(value)))
    except Exception:
        return None


def _extract_label_to_u_iso(cif_block: dict[str, Any]) -> dict[str, float]:
    label_to_u_iso: dict[str, float] = {}
    b_to_u = 1.0 / (8.0 * pi * pi)

    labels = cif_block.get("_atom_site_label", [])
    if "_atom_site_U_iso_or_equiv" in cif_block:
        for label, raw_value in zip(labels, cif_block["_atom_site_U_iso_or_equiv"]):
            parsed = _parse_cif_float(raw_value)
            if parsed is not None:
                label_to_u_iso[str(label)] = max(float(parsed), 0.0)
    elif "_atom_site_B_iso_or_equiv" in cif_block:
        for label, raw_value in zip(labels, cif_block["_atom_site_B_iso_or_equiv"]):
            parsed = _parse_cif_float(raw_value)
            if parsed is not None:
                label_to_u_iso[str(label)] = max(float(parsed) * b_to_u, 0.0)

    if label_to_u_iso:
        return label_to_u_iso

    aniso_labels = cif_block.get("_atom_site_aniso_label", [])
    if (
        aniso_labels
        and "_atom_site_aniso_U_11" in cif_block
        and "_atom_site_aniso_U_22" in cif_block
        and "_atom_site_aniso_U_33" in cif_block
    ):
        for label, u11, u22, u33 in zip(
            aniso_labels,
            cif_block["_atom_site_aniso_U_11"],
            cif_block["_atom_site_aniso_U_22"],
            cif_block["_atom_site_aniso_U_33"],
        ):
            diag_terms = [_parse_cif_float(u11), _parse_cif_float(u22), _parse_cif_float(u33)]
            if any(term is None for term in diag_terms):
                continue
            label_to_u_iso[str(label)] = max(float(sum(diag_terms) / 3.0), 0.0)
    elif (
        aniso_labels
        and "_atom_site_aniso_B_11" in cif_block
        and "_atom_site_aniso_B_22" in cif_block
        and "_atom_site_aniso_B_33" in cif_block
    ):
        for label, b11, b22, b33 in zip(
            aniso_labels,
            cif_block["_atom_site_aniso_B_11"],
            cif_block["_atom_site_aniso_B_22"],
            cif_block["_atom_site_aniso_B_33"],
        ):
            diag_terms = [_parse_cif_float(b11), _parse_cif_float(b22), _parse_cif_float(b33)]
            if any(term is None for term in diag_terms):
                continue
            label_to_u_iso[str(label)] = max(float(sum(diag_terms) / 3.0) * b_to_u, 0.0)

    return label_to_u_iso


def _extract_label_to_u_aniso(cif_block: dict[str, Any]) -> dict[str, list[float]]:
    label_to_u_aniso: dict[str, list[float]] = {}
    b_to_u = 1.0 / (8.0 * pi * pi)
    aniso_labels = cif_block.get("_atom_site_aniso_label", [])
    if not aniso_labels:
        return label_to_u_aniso

    u_keys = (
        "_atom_site_aniso_U_11",
        "_atom_site_aniso_U_22",
        "_atom_site_aniso_U_33",
        "_atom_site_aniso_U_12",
        "_atom_site_aniso_U_13",
        "_atom_site_aniso_U_23",
    )
    b_keys = (
        "_atom_site_aniso_B_11",
        "_atom_site_aniso_B_22",
        "_atom_site_aniso_B_33",
        "_atom_site_aniso_B_12",
        "_atom_site_aniso_B_13",
        "_atom_site_aniso_B_23",
    )

    if all(key in cif_block for key in u_keys):
        source_columns = [cif_block[key] for key in u_keys]
        scale = 1.0
    elif all(key in cif_block for key in b_keys):
        source_columns = [cif_block[key] for key in b_keys]
        scale = b_to_u
    else:
        return label_to_u_aniso

    for values in zip(aniso_labels, *source_columns):
        label = str(values[0])
        components = [_parse_cif_float(value) for value in values[1:]]
        if any(component is None for component in components):
            continue
        label_to_u_aniso[label] = [float(component) * scale for component in components if component is not None]
    return label_to_u_aniso


def build_structure_from_cif_with_thermal_properties(cif_path: str | Path) -> Structure:
    resolved = Path(cif_path).expanduser().resolve()
    parser = CifParser(str(resolved))
    structures = parser.parse_structures(primitive=False)
    if len(structures) == 0:
        raise ValueError(f"No structures parsed from CIF: {resolved}")

    structure = structures[0]
    cif_block = parser._cif.data[list(parser._cif.data.keys())[0]].data
    label_to_u_iso = _extract_label_to_u_iso(cif_block)
    label_to_u_aniso = _extract_label_to_u_aniso(cif_block)
    if label_to_u_iso or label_to_u_aniso:
        site_properties = {
            key: list(values)
            for key, values in structure.site_properties.items()
        }
        if label_to_u_iso:
            site_properties["u_iso"] = [
                float(label_to_u_iso.get(str(site.label), 0.0))
                for site in structure
            ]
        if label_to_u_aniso:
            site_properties["u_aniso"] = [
                label_to_u_aniso.get(str(site.label))
                for site in structure
            ]
        structure = structure.copy(site_properties=site_properties)
    return structure


def build_structure_from_cif(
    cif_path: str | Path,
    *,
    backend: str = "auto",
) -> Structure:
    resolved = Path(cif_path).expanduser().resolve()
    backend_key = str(backend).lower()
    if backend_key not in {"auto", "pymatgen"}:
        raise ValueError(f"unsupported CIF backend: {backend}")
    return build_structure_from_cif_with_thermal_properties(resolved)
