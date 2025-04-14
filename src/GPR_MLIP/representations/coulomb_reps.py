from typing import List
import typesentry
import torch

is_typed = typesentry.Config().is_type  # equivalent of isinstance()


def coulomb(x, charges, offset=0):
    if isinstance(x, torch.Tensor):
        return coulomb_(x, charges, offset=offset)
    elif is_typed(x, List[torch.Tensor]):
        x_new = []
        for geometry in x:
            geometry = coulomb_(geometry, charges, offset=offset)
            x_new.append(geometry)
        return torch.stack(x_new)
    else:
        raise TypeError(f'x has to be torch.Tensor or List[torch.Tensor], not {type(x)}')


def coulomb_atomistic(x, charges):

    if isinstance(x, torch.Tensor):
        return coulomb_atomistic_(x, charges)
    elif is_typed(x, List[torch.Tensor]):
        x_new = []
        for geometry in x:
            geometry = coulomb_atomistic_(geometry, charges)
            x_new.append(geometry)
        return torch.stack(x_new)
    else:
        raise TypeError(f'x has to be torch.Tensor or List[torch.Tensor], not {type(x)}')


def coulomb_cut(x, charges, offset=-1):
    return coulomb(x, charges, offset=offset)


def coulomb_pca(x,  eigen_vectors=None, center=None, **kwargs):
    if center is None:
        raise ValueError('No standard deviation provided.')
    if eigen_vectors is None:
        raise ValueError('No eigen vectors provided.')
    x_representation = coulomb_cut(x,  **kwargs)
    x_representation = x_representation - center
    x_pca = torch.matmul(x_representation, eigen_vectors.T)
    return x_pca



def coulomb_(x_, charges, offset=0):
    n_atoms = int(len(x_) / 3)

    coulomb_matrix = coulomb_atomistic_(x_, charges)
    tril_inds = torch.tril_indices(n_atoms, n_atoms, offset=offset)
    return coulomb_matrix[tril_inds[0], tril_inds[1]]


def coulomb_atomistic_(x_, charges_):
    n_atoms = int(len(x_) / 3)

    x_ = torch.reshape(x_, (n_atoms, 3))
    r_dist = torch.cdist(x_, x_, p=2)
    coulomb_matrix = torch.outer(charges_, charges_) * (r_dist ** -1)
    coulomb_matrix[range(n_atoms), range(n_atoms)] = \
        torch.tensor(0.5) * charges_ ** torch.tensor(2.4)

    return coulomb_matrix
