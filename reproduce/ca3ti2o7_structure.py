"""Mean-position displacement and TiO6 orientation descriptors."""

from __future__ import annotations

import numpy as np


def rotation_vector(row_rotation):
    rotation = row_rotation.T
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    if angle < 1e-10:
        return np.zeros(3)
    if abs(np.sin(angle)) < 1e-8:
        values, vectors = np.linalg.eig(rotation)
        axis = np.real(vectors[:, np.argmin(np.abs(values - 1))])
        axis /= np.linalg.norm(axis)
    else:
        axis = np.asarray([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                           rotation[1, 0] - rotation[0, 1]]) / (2 * np.sin(angle))
    return angle * axis


def octahedral_rotations(reference, current, cell, neighbors):
    rotations = []
    for titanium, oxygen in neighbors.items():
        first = reference[oxygen] - reference[titanium]
        second = current[oxygen] - current[titanium]
        first = (first - np.round(first)) @ cell
        second = (second - np.round(second)) @ cell
        first -= first.mean(0)
        second -= second.mean(0)
        u, _, vt = np.linalg.svd(first.T @ second)
        if np.linalg.det(u @ vt) < 0:
            u[:, -1] *= -1
        rotations.append(rotation_vector(u @ vt))
    return np.stack(rotations)


class Descriptors:
    def __init__(self, problem):
        d = problem.data
        self.reference, self.basis = d["acaa_fractional"], problem.basis
        titanium = np.flatnonzero(d["species"] == "Ti")
        oxygen = np.flatnonzero(d["species"] == "O")
        self.neighbors = {}
        for site in titanium:
            delta = d["neighbor_reference_fractional"][oxygen] - d["neighbor_reference_fractional"][site]
            vectors = (delta - np.round(delta)) @ d["neighbor_reference_lattice"]
            self.neighbors[int(site)] = oxygen[np.argsort(np.linalg.norm(vectors, axis=1))[:6]]
        candidates = np.flatnonzero(d["neutron_cycle"] == "heating")
        index = candidates[np.argmin(d["neutron_temperature"][candidates])]
        self.template = octahedral_rotations(
            self.reference, d["neutron_fractional"][index], np.diag(d["neutron_lattice"][index]), self.neighbors
        )[:, 2]
        if np.linalg.norm(self.template) <= 0:
            raise ValueError("zero neutron rotation template")

    def __call__(self, amplitudes, lengths):
        cell = np.diag(lengths)
        fractional = np.mod(self.reference + np.einsum("m,msd->sd", amplitudes, self.basis) @ np.linalg.inv(cell), 1.)
        delta = fractional - self.reference
        displacement = (delta - np.round(delta)) @ cell
        rotations = octahedral_rotations(self.reference, fractional, cell, self.neighbors)
        return {
            "structural_rms_from_acaa_angstrom": float(np.sqrt(np.mean(np.sum(displacement**2, axis=1)))),
            "tio6_inplane_tilt_change_from_acaa_deg": float(np.sqrt(np.mean(np.sum(rotations[:, :2]**2, axis=1))) * 180 / np.pi),
            "tio6_c_axis_coherent_rotation_deg": float(np.dot(rotations[:, 2], self.template / np.linalg.norm(self.template))
                                                       * 180 / np.pi / np.sqrt(len(rotations))),
        }


def pseudo_voigt(x, center, fwhm, eta):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-0.5 * ((x - center) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    gamma = fwhm / 2
    lorentzian = gamma / (np.pi * ((x - center)**2 + gamma**2))
    return (1 - eta) * gaussian + eta * lorentzian


def reference_doublet(parameters, x, observed, variance):
    """Fixed measured peak areas/centers/width, with only background refitted."""
    area_213, area_117, center_213, center_117, fwhm, eta = parameters
    peaks = area_213 * pseudo_voigt(x, center_213, fwhm, eta) + area_117 * pseudo_voigt(x, center_117, fwhm, eta)
    design = np.column_stack((np.ones_like(x), x - x.mean()))
    weight = 1 / np.sqrt(variance)
    coefficients, _, rank, _ = np.linalg.lstsq(design * weight[:, None], (observed - peaks) * weight, rcond=None)
    if rank != 2:
        raise ValueError("doublet background is rank deficient")
    return peaks + design @ coefficients
