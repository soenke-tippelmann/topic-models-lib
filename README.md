# topic-models-lib

A library to train topic models. The core of this library was written during the Semantic Search Project at TU Berlin Summer Term 2017 as part of a bigger project. The code here is a refactored standalone version.

Currently only two topic models are supported:
1) Latent Dirichlet Allocation (LDA), which was introduced in [1].
2) The Dynamic Topic Model (DTM), which was introduced in [2].

For usage information please see the [examples directory](src/main/scala/de/tml/examples).

# Resources

[1] David M. Blei, Andrew Y. Ng, and Michael I. Jordan. 2003. Latent dirichlet allocation. J. Mach. Learn. Res. 3 (March 2003), 993-1022. 

[2] David M. Blei and John D. Lafferty. 2006. Dynamic topic models. In Proceedings of the 23rd international conference on Machine learning (ICML '06). ACM, New York, NY, USA, 113-120. DOI: https://doi.org/10.1145/1143844.1143859 
