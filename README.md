fastreg
============
implementation of "fastreg: fast non-rigid registration via accelerated optimisation on the manifold of diffeomorphisms", accepted for the 26th conference on information processing in medical imaging (ipmi 2019)

paper: [arxiv](https://arxiv.org/abs/1903.01905)

dependencies
-----------
* boost
* itk
* nvidia gpu with cuda >=8.0.44
* opencv

building
-----------
```
source setup.sh -a
```

usage
-----------
images must be of type `float32` and labels of type `int16`

```
./build/bin/app ./params.ini <path_to_target> <path_to_target_labels> <path_to_source> <path_to_source_labels> <output_path>
```

reference
-----------
D. Grzech et al., “FastReg: Fast Non-Rigid Registration via Accelerated Optimisation on the Manifold of Diffeomorphisms”, 2019.

