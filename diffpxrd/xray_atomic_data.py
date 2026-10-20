from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from pymatgen.core import Element

try:
    from . import _vendored_atmdata as _atmdata
except ImportError:
    import _vendored_atmdata as _atmdata

_AU = 2.80022e7
_C1 = 0.02721
_C_LIGHT = 137.0367
_ALG = (
    0.11846344252810,
    0.23931433524968,
    0.28444444444400,
    0.23931433524968,
    0.11846344252810,
)
_XLG = (
    0.04691007703067,
    0.23076534494716,
    0.5,
    0.76923465505284,
    0.95308992296933,
)


def _xsect_data_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "Xsect.dat"


def load_reference_atmdata() -> Any:
    return _atmdata


def _get_el_info(symbol: str, wavelength: float) -> dict[str, float]:
    fp, fpp, mu = _fpcalc(_get_xsection_coeffs(symbol), 12.397639 / float(wavelength))
    return {"fp": float(fp), "fpp": float(fpp), "mu": float(mu)}


def load_reference_element_api() -> Any:
    return SimpleNamespace(
        GetXsectionCoeff=_get_xsection_coeffs,
        FPcalc=_fpcalc,
        GetElInfo=lambda symbol, inst: _get_el_info(symbol, float(inst["Lam"][1])),
    )


def _element_symbol(value: Any) -> str:
    if hasattr(value, "element") and getattr(value.element, "symbol", None):
        return str(value.element.symbol)
    if hasattr(value, "symbol") and getattr(value, "symbol", None):
        return str(value.symbol)
    text = str(value).strip()
    if not text:
        return "H"
    symbol = text[0]
    if len(text) > 1 and text[1].islower():
        symbol += text[1]
    return symbol


def _format_integer_charge_label(symbol: str, oxidation_state: float | None) -> str | None:
    if oxidation_state is None:
        return None
    rounded = int(round(float(oxidation_state)))
    if abs(float(oxidation_state) - rounded) > 1.0e-8 or rounded == 0:
        return None
    sign = "+" if rounded > 0 else "-"
    return f"{symbol}{sign}{abs(rounded)}"


def canonicalize_species_label(species: Any) -> str:
    symbol = _element_symbol(species)
    oxidation_state = getattr(species, "oxi_state", None)
    charge_label = _format_integer_charge_label(symbol, oxidation_state)
    if charge_label is not None:
        return charge_label
    return symbol


def _candidate_species_labels(species: Any) -> list[str]:
    symbol = _element_symbol(species)
    candidates: list[str] = []
    exact = canonicalize_species_label(species)
    if exact:
        candidates.append(exact)
    raw_text = str(species).strip()
    if raw_text and raw_text not in candidates:
        candidates.append(raw_text)
    if symbol not in candidates:
        candidates.append(symbol)
    if symbol in {"D", "T"} and "H" not in candidates:
        candidates.append("H")
    return candidates


@lru_cache(maxsize=None)
def _lookup_coeffs_from_label_cached(
    label: str,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, float, int, str]:
    table = _atmdata.XrayFF
    if label in table:
        entry = table[label]
        return (
            np.asarray(entry["fa"], dtype=dtype),
            np.asarray(entry["fb"], dtype=dtype),
            float(entry.get("fc", 0.0)),
            int(entry["Z"]),
            label,
        )
    raise KeyError(label)


def _lookup_coeffs_from_label(
    label: str,
    *,
    dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray, float, int, str]:
    return _lookup_coeffs_from_label_cached(label, np.dtype(dtype))


def lookup_xray_form_factor_coeffs(
    species: Any,
    *,
    dtype: np.dtype | type = np.float32,
) -> tuple[np.ndarray, np.ndarray, float, int, str]:
    """Resolve local form-factor coefficients in float32 or float64 precision."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be numpy.float32 or numpy.float64")
    last_error: Exception | None = None
    candidate_labels = _candidate_species_labels(species)
    for label in candidate_labels:
        try:
            return _lookup_coeffs_from_label(label, dtype=dtype)
        except Exception as exc:  # pragma: no cover - only used on rare lookup fallbacks
            last_error = exc
            continue
    raise KeyError(
        f"Unable to resolve x-ray form factor coefficients for {species!r}; "
        f"tried labels {candidate_labels}"
    ) from last_error


@lru_cache(maxsize=None)
def _get_xsection_coeffs(element_symbol: str) -> tuple[dict[str, Any], ...]:
    target_symbol = str(element_symbol).upper().ljust(2)
    orbitals: list[dict[str, Any]] = []
    with _xsect_data_path().open("r", encoding="utf-8") as xsect:
        while True:
            line = xsect.readline()
            if not line:
                break
            if line[:2] != target_symbol:
                continue
            record = line.rstrip("\n") + xsect.readline().rstrip("\n") + xsect.readline()
            orbital_name = record[9:14]
            payload = record[14:]
            if_be = int(payload[0])
            values = payload[1:].split()
            bind_en = float(values[0])
            orbital: dict[str, Any] = {
                "OrbName": orbital_name,
                "IfBe": if_be,
                "BindEn": bind_en,
                "BB": bind_en / _C1,
            }
            energy = [float(values[2 * index + 1]) for index in range(11)]
            xsect_values = [float(values[2 * index + 2]) for index in range(11)]
            orbital["XSecIP"] = [xsect_values[index + 5] / _AU for index in range(5)]
            if if_be == 0:
                orbital["SEdge"] = xsect_values[10] / _AU
                nval = 11
            else:
                orbital["ElEterm"] = xsect_values[10]
                del energy[10]
                del xsect_values[10]
                orbital["SEdge"] = 0.0
                nval = 10
            orbital["Nval"] = nval
            sorted_pairs = sorted(zip(energy, xsect_values, strict=False), key=lambda pair: pair[0])
            orbital["LEner"] = [math.log(value) for value, _ in sorted_pairs]
            orbital["LXSect"] = [math.log(value) if value > 0.0 else 0.0 for _, value in sorted_pairs]
            orbitals.append(orbital)
    return tuple(orbitals)


def _aitken(orbital: dict[str, Any], log_kev: float) -> float:
    nval = orbital["Nval"]
    j = nval - 1
    log_energy = orbital["LEner"]
    for index, energy in enumerate(log_energy):
        if energy <= log_kev:
            j = index
    if j > nval - 3:
        j = nval - 3
    t = [0.0] * 6
    log_xsect = orbital["LXSect"]
    for index in range(3):
        t[index] = log_xsect[index + j]
        t[index + 3] = log_energy[index + j] - log_kev
    t[1] = (t[0] * t[4] - t[1] * t[3]) / (log_energy[j + 1] - log_energy[j])
    t[2] = (t[0] * t[5] - t[2] * t[3]) / (log_energy[j + 2] - log_energy[j])
    t[2] = (t[1] * t[5] - t[2] * t[4]) / (log_energy[j + 2] - log_energy[j + 1])
    return t[2]


def _dgauss(orbital: dict[str, Any], cx: float, rx: float, isig: int) -> float:
    result = 0.0
    bb = orbital["BB"]
    bb_sq = bb**2
    rx_sq = rx**2
    xsect_ip = orbital["XSecIP"]
    for weight, abscissa, xs in zip(_ALG, _XLG, xsect_ip, strict=False):
        x_sq = abscissa**2
        if isig == 0:
            value = bb * (xs * (bb_sq / x_sq) - cx * rx_sq) / (rx_sq * x_sq - bb_sq)
        elif isig == 1:
            value = 0.5 * bb * bb_sq * xs / (math.sqrt(abscissa) * (rx_sq * x_sq - abscissa * bb_sq))
        elif isig == 2:
            temp = abscissa * x_sq * rx_sq - bb_sq / abscissa
            value = 2.0 * bb * (xs * bb_sq / (temp * x_sq**2) - (cx * rx_sq / temp))
        else:
            value = bb * bb_sq * (xs - orbital["SEdge"] * x_sq) / (rx_sq * x_sq**2 - x_sq * bb_sq)
        result += weight * value
    return result


def _fpcalc(orbitals: tuple[dict[str, Any], ...], kev: float) -> tuple[float, float, float]:
    fp = 0.0
    fpp = 0.0
    mu = 0.0
    log_kev = math.log(kev)
    rx = kev / _C1
    el_eterm = 0.0
    for orbital in orbitals:
        cx = 0.0
        bb = orbital["BB"]
        bind_en = orbital["BindEn"]
        if orbital["IfBe"] != 0:
            el_eterm = orbital["ElEterm"]
        if bind_en <= kev:
            cx = math.exp(_aitken(orbital, log_kev))
            mu += cx
            cx /= _AU
        corr = 0.0
        if orbital["IfBe"] == 0 and bind_en >= kev:
            cx = 0.0
            fpi = _dgauss(orbital, cx, rx, 3)
            corr = 0.5 * orbital["SEdge"] * bb**2 * math.log((rx - bb) / (-rx - bb)) / rx
        else:
            fpi = _dgauss(orbital, cx, rx, orbital["IfBe"])
            if cx != 0.0:
                corr = -0.5 * cx * rx * math.log((rx + bb) / (rx - bb))
        fp += (fpi + corr) * _C_LIGHT / (2.0 * math.pi**2)
        fpp += _C_LIGHT * cx * rx / (4.0 * math.pi)
    fp -= el_eterm
    return (fp, fpp, mu)


@lru_cache(maxsize=None)
def _lookup_xray_dispersion_terms_cached(
    atomic_number: int,
    wavelength: float,
) -> tuple[float, float]:
    symbol = Element.from_Z(int(atomic_number)).symbol
    fp, fpp, _mu = _fpcalc(_get_xsection_coeffs(symbol), 12.397639 / float(wavelength))
    return float(fp), float(fpp)


def lookup_xray_dispersion_terms(
    atomic_number: int,
    wavelength: float,
) -> tuple[float, float]:
    return _lookup_xray_dispersion_terms_cached(
        int(atomic_number),
        float(wavelength),
    )
