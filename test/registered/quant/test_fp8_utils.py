import unittest

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8_utils import (
    inverse_transform_scale_ue8m0,
    quant_weight_ue8m0,
    transform_scale_ue8m0,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=8, suite="stage-b-test-1-gpu-large")


class TestInverseTransformScaleUe8m0(CustomTestCase):
    def test_round_trip(self):
        for _ in range(100):
            weight_bf16 = torch.randn(
                # DeepSeek V3 kv_b_proj
                (32768, 512),
                dtype=torch.bfloat16,
                device="cuda",
            )

            weight_block_size = [128, 128]

            qweight, sf_fp32_original = quant_weight_ue8m0(
                weight_bf16, weight_block_size=weight_block_size
            )
            mn = qweight.shape[-2]

            sf_packed_original = transform_scale_ue8m0(sf_fp32_original, mn=mn)
            sf_fp32_recreated = inverse_transform_scale_ue8m0(sf_packed_original, mn=mn)

            sf_packed_recreated = transform_scale_ue8m0(sf_fp32_recreated, mn=mn)

            assert torch.all(
                sf_packed_original == sf_packed_recreated
            ), f"{sf_packed_original=} {sf_packed_recreated}"
            assert torch.all(
                sf_fp32_original == sf_fp32_recreated
            ), f"{sf_fp32_original=} {sf_fp32_recreated}"


class TestFp8ScaledMmAbstract(CustomTestCase):
    def test_accepts_out_argument(self):
        if not hasattr(fp8_utils, "_fp8_scaled_mm_abstract"):
            self.skipTest("CUDA fp8_scaled_mm fake implementation is not registered")

        mat_a = torch.empty((3, 4), device="meta", dtype=torch.float8_e4m3fn)
        mat_b = torch.empty((4, 5), device="meta", dtype=torch.float8_e4m3fn)
        scales_a = torch.empty((3, 1), device="meta", dtype=torch.float32)
        scales_b = torch.empty((1, 5), device="meta", dtype=torch.float32)
        out = torch.empty((3, 5), device="meta", dtype=torch.bfloat16)

        result = fp8_utils._fp8_scaled_mm_abstract(
            mat_a,
            mat_b,
            scales_a,
            scales_b,
            torch.bfloat16,
            None,
            out,
        )

        self.assertIs(result, out)
        self.assertEqual(result.shape, (3, 5))
        self.assertEqual(result.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
