"""
VLM2Vec adapted OmniEmbed model.
Based on nvidia/omni-embed-nemotron-3b with custom forward logic to handle list inputs.
"""

from .modeling_omni_embed import OmniEmbedForConditionalGeneration

__all__ = ['OmniEmbedForConditionalGeneration']

