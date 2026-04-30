"""V2 entry point — composes the hierarchical-target diffusion task with the
DiT-style transformer denoiser via Python's MRO.

Picked apart, the v2 stack is::

    HierarchicalCodecDiffusion   # subclass of CodecDiffusion: target/score in coef space
            ▲
            │
    TransformerDenoiser          # subclass of CodecDiffusion: DiT denoiser
            ▲
            │
        V2Model                  # multiple-inheritance combiner

Both bases inherit from ``CodecDiffusion``. With the order
``(HierarchicalCodecDiffusion, TransformerDenoiser)``, MRO resolves
``step()``/``p_sample()`` to ``HierarchicalCodecDiffusion`` (coef-space
training) and ``forward()`` to ``TransformerDenoiser`` (the actual model).
"""

from task.hierarchical_diffusion import HierarchicalCodecDiffusion
from model.transformer_denoiser import TransformerDenoiser


class V2Model(HierarchicalCodecDiffusion, TransformerDenoiser):
    """The default v2 model class. Use via ``cfg.model.model.name = 'V2Model'``."""
    pass
