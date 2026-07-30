import unittest

import torch

from src.scene.deformation import ExplicitSparseDeformation, ExplicitSparseFDMDeformation


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by the deformation modules")
class SparseDeformationAnchorQueryTests(unittest.TestCase):
    def _initialize_topology(self, deformation, count=256):
        torch.manual_seed(7)
        means = torch.randn(count, 3, device="cuda") * 0.02
        anchor_ids = torch.arange(0, count, 16, device="cuda", dtype=torch.long)
        deformation.anchor_ids = anchor_ids
        if isinstance(deformation, ExplicitSparseFDMDeformation):
            deformation.means_def = torch.nn.Parameter(
                torch.randn(anchor_ids.numel(), 3, deformation.basis_num, device="cuda") * 0.001
            )
            deformation.rot_def = torch.nn.Parameter(
                torch.zeros(anchor_ids.numel(), 4, deformation.basis_num, device="cuda")
            )
        else:
            deformation.means_def = torch.nn.Parameter(
                torch.randn(anchor_ids.numel(), 3, device="cuda") * 0.001
            )
            deformation.rot_def = torch.nn.Parameter(
                torch.zeros(anchor_ids.numel(), 4, device="cuda")
            )
        deformation.init_topology(means, anchor_ids)
        return means, anchor_ids

    def test_sparse_targeted_means_match_full_interpolation(self):
        for deformation in (ExplicitSparseDeformation(), ExplicitSparseFDMDeformation()):
            with self.subTest(deformation=type(deformation).__name__):
                means, anchor_ids = self._initialize_topology(deformation)
                expected = deformation.get_deformed_means(means)[anchor_ids]
                actual = deformation.get_deformed_means_at_indices(means, anchor_ids)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))

    def test_anchor_flow_update_matches_legacy_full_update(self):
        for deformation in (ExplicitSparseDeformation(), ExplicitSparseFDMDeformation()):
            with self.subTest(deformation=type(deformation).__name__):
                means, anchor_ids = self._initialize_topology(deformation)
                torch.manual_seed(13)
                full_motion = torch.randn(means.shape[0], 3, 3, device="cuda") * 0.002
                full_weights = torch.rand(means.shape[0], 3, device="cuda")
                before = deformation.means_def.detach().clone()
                deformation.init_from_flow(full_motion, full_weights)
                legacy_result = deformation.means_def.detach().clone()
                deformation.means_def.data.copy_(before)
                deformation.init_anchors_from_flow(full_motion[anchor_ids], full_weights[anchor_ids])
                self.assertTrue(torch.allclose(deformation.means_def, legacy_result, atol=1e-7, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
