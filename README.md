python train.py --multirun model.timestep_embedding=true,false
python train.py -m --config-name sweep_muon_masking

Good configs for 3-2 layer/head combo:

Masking
Muon
No bias

----

Benchmark instructions:

ncu --profile-from-start off \
    --nvtx \
    --nvtx-include "FullModelForward/" \
    --set full \
    -o full_model_trace \
    python gpu_bench.py
