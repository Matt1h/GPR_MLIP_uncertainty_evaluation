import os
from os.path import join
from typing import List, Dict
import numpy as np
import torch
from dscribe.descriptors import SOAP
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


from .nuclear_charge_dict import NUCLEAR_CHARGE


SPECIES = {value: key for key, value in NUCLEAR_CHARGE.items()}


def soap_atomistic(
        geometries,
        charges,
        periodic=False,
        r_cut=5.0,
        n_max=3,
        l_max=1,
        ignore_h=False,
        **kwargs,
):
    species = species_in_charges(charges)
    soap_ = SOAP(
        species=species,
        periodic=periodic,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        **kwargs,
    )

    ase_atoms_l = geometries_to_ase_atoms(geometries, charges)

    soap_descriptors = []
    if ignore_h:
        species_list = charges_to_species(charges)
        idx_no_h = [i for i, spec in enumerate(species_list) if spec != 'H']

        for ase_atoms in ase_atoms_l:
            soap_descriptors.append(soap_.create(ase_atoms, idx_no_h))
    else:
        for ase_atoms in ase_atoms_l:
            soap_descriptors.append(soap_.create(ase_atoms))

    soap_descriptors = torch.tensor(np.array(soap_descriptors))
    if torch.cuda.is_available():
        soap_descriptors = soap_descriptors.cuda()

    return soap_descriptors


def soap(geometries, charges, **kwargs):
    soap_descriptors = soap_atomistic(geometries, charges, **kwargs)

    n_samples = soap_descriptors.shape[0]
    n_atoms = soap_descriptors.shape[1]
    n_soap_entries = soap_descriptors.shape[2]

    soap_descriptors = soap_descriptors.reshape(n_samples, n_atoms * n_soap_entries)
    return soap_descriptors


def soap_global(
        geometries,
        charges,
        periodic=False,
        r_cut=5.0,
        n_max=3,
        l_max=1,
        average="outer",
        **kwargs,
):
    species = species_in_charges(charges)
    soap_ = SOAP(
        species=species,
        periodic=periodic,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average=average,
        **kwargs,
    )

    ase_atoms_l = geometries_to_ase_atoms(geometries, charges)

    soap_descriptors = []
    for ase_atoms in ase_atoms_l:
        soap_descriptors.append(soap_.create(ase_atoms))

    soap_descriptors = torch.tensor(np.array(soap_descriptors))
    return soap_descriptors


def charges_to_species(charges):
    species = []
    for charge in charges:
        species.append(SPECIES[int(charge)])
    return species


def species_in_charges(charges):
    charges_set = set([float(charge) for charge in charges])
    species = []
    for k, v in NUCLEAR_CHARGE.items():
        if v in charges_set:
            species.append(k)
    return species


def create_atoms_object(geometry: torch.Tensor, charges: torch.Tensor, properties: dict) -> Atoms:
    """Creates an ASE Atoms object from geometry, charges, and other properties."""
    n_atoms = int(len(geometry) / 3)

    atoms = Atoms(positions=geometry.reshape((n_atoms, 3)), numbers=charges)

    if properties:
        props = {k: v for k, v in properties.items()}
        atoms.calc = SinglePointCalculator(atoms, **props)

    return atoms


def geometries_to_ase_atoms(
        geometries: torch.Tensor,
        charges: torch.Tensor,
        **properties: Dict[str, np.ndarray]
) -> List[Atoms]:
    """Converts tensor of geometries to list with ASE atoms, handling optional properties like energies and forces."""
    ase_atoms_l = []
    for i, geometry in enumerate(geometries):
        props_for_atoms = {prop: properties[prop][i] for prop in properties}
        atoms = create_atoms_object(geometry, charges, props_for_atoms)
        ase_atoms_l.append(atoms)
    return ase_atoms_l