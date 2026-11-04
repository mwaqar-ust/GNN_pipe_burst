# References and further reading

This tutorial develops one small graph neural network step by step. The references provide application context, graph-learning foundations and documentation for the hydraulic tools. The notebook is an educational example rather than a reproduction of a particular published experiment.

## A short literature introduction

1. **Passive transient localisation with graph learning.** Waqar, M., and Ghidaoui, M. S. (2026). *Passive Transient Source Localization in Water Distribution Network using Graph Neural Networks*. The 16th International Conference on Hydrodynamics (ICHD 2026), Macau, 14–18 April 2026. [HKUST publication record](https://researchportal.hkust.edu.hk/en/publications/passive-transient-source-localization-in-water-distribution-netwo/). This provides application context for learning source locations from network observations. The tutorial uses the supplied HKUST retained primary network and its own compact teaching model.

2. **Graph convolution and learned representations.** Kipf, T. N., and Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. International Conference on Learning Representations (ICLR). [Paper](https://arxiv.org/abs/1609.02907). This introduces a widely used approach to learning node representations from node features and local graph structure. The tutorial adapts neighbourhood aggregation to an event-level pipe and position output.

3. **Hydraulic transient simulation.** Xing, L., and Sela, L. (2020). *Transient simulations in water distribution networks: TSNet python package*. Advances in Engineering Software, **149**, 102884. [DOI](https://doi.org/10.1016/j.advengsoft.2020.102884). TSNet supplies the method-of-characteristics pressure histories used in the signal demonstrations and hydraulic tests.

## Software and data documentation

4. **TSNet examples and numerical model.** [Example applications](https://tsnet.readthedocs.io/en/latest/examples.html), [transient modelling framework](https://tsnet.readthedocs.io/en/latest/transient.html), and [original TSNet repository](https://github.com/glorialulu/TSNet). The supplied HKUST geometry and controlled hydraulic INP are retained in `networks/`, together with their source information. The pinned package version is recorded in `requirements.txt`; a documentation banner may refer to an earlier release.

5. **WNTR pipe insertion.** U.S. Environmental Protection Agency and Sandia National Laboratories, *Water Network Tool for Resilience*. [WNTR documentation](https://usepa.github.io/WNTR/) and [`wntr.morph.link.split_pipe`](https://usepa.github.io/WNTR/apidoc/wntr.morph.link.split_pipe.html). A temporary junction places the simulated burst inside an original pipe. The original pipe ID and position fraction remain the learning target.

Machine-readable entries are supplied in `references.bib`. `CITATION.cff` describes this tutorial itself. The conference reference is verified against the institutional record; no journal publication or DOI is inferred for that work.

Links checked on 4 November 2026.
