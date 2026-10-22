import pytest
import torch
from scripts.backtrack_grpo_c import interpolate_aligned_heads


def test_head_backtracking_endpoints_and_alignment_guard():
    before = {'actor.0.weight':torch.ones(2), 'actor.9.weight':torch.zeros(2),
              'actor.9.bias':torch.zeros(3), 'logstd':torch.ones(3)}
    after = {k:v.clone() for k,v in before.items()}
    after['actor.9.weight'].fill_(4.)
    after['actor.9.bias'].fill_(2.)
    for alpha, expected in [(0.,before),(1.,after)]:
        result = interpolate_aligned_heads(before,after,alpha)
        for k in before:
            torch.testing.assert_close(result[k],expected[k],rtol=0,atol=0)
    result = interpolate_aligned_heads(before,after,.25)
    torch.testing.assert_close(result['actor.9.weight'],torch.ones(2))
    torch.testing.assert_close(result['actor.9.bias'],torch.full((3,),.5))
    after['actor.0.weight'].add_(1)
    with pytest.raises(ValueError,match='alignment mismatch'):
        interpolate_aligned_heads(before,after,.5)
