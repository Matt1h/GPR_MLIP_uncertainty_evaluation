from typing import Callable, Optional
import torch
import gpytorch


def process_in_batches(
        atomistic_kernel,
        representations1,
        representations2,
        batch_size,
):

    n_molecules1 = representations1.shape[0]
    n_molecules2 = representations2.shape[0]
    n_atoms = representations1.shape[1]
    n_entries_representation = representations1.shape[2]

    res = []
    for start_idx in range(0, n_molecules1, batch_size):
        end_idx = min(start_idx + batch_size, n_molecules1)

        # Select the subset of data for this batch
        batch_representations1 = representations1[start_idx:end_idx]
        batch_atomistic_representations1 = batch_representations1.reshape(
            (end_idx - start_idx) * n_atoms,
            n_entries_representation
        )

        # Perform the operation on the batch
        batch_result = torch.sum(
            atomistic_kernel(
                batch_atomistic_representations1,
                representations2
            ).evaluate().reshape(n_molecules2, (end_idx - start_idx), n_atoms, n_atoms),
            dim=[2, 3]
        ).T
        res.append(batch_result)
    res = torch.cat(res)
    return res


class AtomisticSumKernel1(gpytorch.kernels.Kernel):

    def __init__(
            self,
            atomistic_kernel,
            representation_fn: Callable,
            bs: Optional[int] = None,
            **kwargs
    ):
        super().__init__()

        self.representation_fn = representation_fn
        if 'ard_num_dims' in kwargs:
            self.ard_num_dims = kwargs['ard_num_dims']

        self.atomistic_kernel = atomistic_kernel(**kwargs)
        self.bs = bs

    def forward(self, x1, x2, diag=False, **params):

        if x1.shape[0] == 1:
            x1 = x1.squeeze(dim=0)
        if x2.shape[0] == 1:
            x2 = x2.squeeze(dim=0)

        representations1 = self.representation_fn(list(x1.cpu()))
        representations2 = self.representation_fn(list(x2.cpu()))

        if torch.cuda.is_available():
            representations1 = representations1.to(torch.device("cuda"))
            representations2 = representations2.to(torch.device("cuda"))

        n_molecules1 = representations1.shape[0]
        n_molecules2 = representations2.shape[0]
        n_atoms = representations1.shape[1]
        n_entries_representation = representations1.shape[2]

        if self.bs is None:
            atomistic_representations1 = representations1.reshape(n_molecules1 * n_atoms, n_entries_representation)
            res = torch.sum(
                self.atomistic_kernel(
                    atomistic_representations1,
                    representations2
                ).evaluate().reshape(n_molecules2, n_molecules1, n_atoms, n_atoms),
                dim=[2, 3]
            ).T
        else:
            res = process_in_batches(self.atomistic_kernel, representations1, representations2, self.bs)

        if diag:
            res = torch.diag(res)

        return res


# class AtomisticSumKernel2(gpytorch.kernels.Kernel):

#     def __init__(
#             self,
#             atomistic_kernel,
#             representation: Callable,
#             representation_kwargs: Optional[dict] = None,
#             cutoff: Optional[float] = None,
#             **kwargs
#     ):
#         super().__init__()

#         self.representation = representation
#         self.representation_kwargs = representation_kwargs or {}
#         self.cutoff = cutoff
#         self.n_atoms = len(self.representation_kwargs['charges'])

#         for i in range(self.n_atoms):
#             setattr(self, f'atomistic_kernel{i}', gpytorch.kernels.ScaleKernel(atomistic_kernel(**kwargs)))

#     def forward(self, x1, x2, **params):

#         if x1.shape[0] == 1:
#             x1 = x1.squeeze(dim=0)
#         if x2.shape[0] == 1:
#             x2 = x2.squeeze(dim=0)

#         representations1 = self.representation(list(x1), **self.representation_kwargs)
#         representations2 = self.representation(list(x2), **self.representation_kwargs)

#         if self.cutoff is not None:
#             representations1 = self.apply_cutoff(representations1, x1)
#             representations2 = self.apply_cutoff(representations2, x2)

#         atomistic_k = []
#         for i in range(self.n_atoms):
#             base_kernel = getattr(self, f'atomistic_kernel{i}')
#             atomistic_k.append(base_kernel(representations1[:, i, :], representations2[:, i, :]).evaluate())
#         res = torch.sum(torch.stack(atomistic_k), dim=0)

#         return res

#     def apply_cutoff(self, representations, x):
#         n_samples = len(x)

#         mask = torch.cdist(
#             x.reshape(n_samples, self.n_atoms, 3),
#             x.reshape(n_samples, self.n_atoms, 3)
#         ) > self.cutoff
#         representations[mask] = 0
#         return representations


# class FlareLikeKernel(gpytorch.kernels.Kernel):
#     def __init__(
#             self,
#             atomistic_kernel,
#             n_features,
#             **kwargs
#     ):
#         super().__init__()

#         self.n_features = n_features
#         for i in range(self.n_features):
#             setattr(self, f'atomistic_kernel{i}', gpytorch.kernels.ScaleKernel(atomistic_kernel(**kwargs)))

#     def forward(self, x1, x2, diag=False, **params):

#         if x1.shape[0] == 1:
#             x1 = x1.squeeze(dim=0)
#         if x2.shape[0] == 1:
#             x2 = x2.squeeze(dim=0)

#         res = torch.sum(
#             torch.stack(
#                 [
#                     getattr(self, f'atomistic_kernel{i}')(x1[:, i], x2[:, i]).evaluate()
#                     for i in range(self.n_features)
#                 ]
#             ), dim=0,
#         )

#         if diag:
#             res = torch.diag(res)

#         return res
