# Notices and model lineage

Authors: Trevor Martin–DataStreams Solutions; GPT-6 Astra (AI coauthor).
The AI assistant implemented and revised the learned connections and authored
reversible direct tensor edits under Trevor Martin's research direction.

This is an experimental native-text model extension, not a trained standalone
4B base or a release of a strong Go player. The public package is newly extracted
from the evaluated research code. Its CPU equivalence checks do not substitute
for a full-model GPU replay on the public loader.

Qwen/Qwen3.5-4B is by Alibaba Cloud, under Apache-2.0. See QWEN_LICENSE.
Download the separately hosted base at revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a. Base weights are not included.
The supplied native layer16 patch is a modified subset of those weights.

Moka by Million Software, Inc. is pinned to commit
3b249ec094c3f0fe0eb9b96144e25d65149954a9. Its MIT license and third-party
notices are in moka/. The donor is frozen. Its trunk representations, not
its policy outputs, feed the learned receiver. Upstream repository:
https://github.com/millionco/moka

Our connector code and learned connector weights are released under Apache-2.0.
The donor tensors embedded in bridge017.safetensors retain their MIT attribution.
No affiliation with or endorsement by Alibaba/Qwen or Million Software is claimed.

Native inference in the evaluated experiment disabled the published decision
LoRA and later choice residual. This package therefore does NOT silently attach
those decision-training weights. It reproduces that native architecture using
the base model, the retained native layer16 patch, the frozen donor, and bridge017.
The packed typed-decision route and its preservation guards are not included.
