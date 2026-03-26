rjob submit --gpu=8 --memory=1000000 -P 1 --cpu=128 --priority 8 --name=distill_min_kl \
--host-network=true \
-e DISTRIBUTED_JOB=true \
--charged-group=protfma_gpu --private-machine=group \
--custom-resources rdma/mlnx_shared=8 \
--custom-resources mellanox.com/mlnx_rdma=1 \
--mount=gpfs://gpfs2/ai4scifm-gpfs02:/mnt/shared-storage-gpfs2/ai4scifm-gpfs02 \
--image=registry.h.pjlab.org.cn/ailab/pytorch2.7.0-cuda12.8.1-py3.12-ubuntu24.04:v2 \
-- bash -c "
  CONDA_PATH=/mnt/shared-storage-gpfs2/ai4scifm-gpfs02/wuyixin/miniconda3
  . \${CONDA_PATH}/etc/profile.d/conda.sh && conda init && conda activate ssttyy

  cd /mnt/shared-storage-gpfs2/ai4scifm-gpfs02/wuyixin/code/sty-lmf
  export NCCL_TIMEOUT=18000
  export NCCL_DEBUG=INFO
  export HF_DATASETS_CACHE=/mnt/shared-storage-gpfs2/ai4scifm-gpfs02/wuyixin/data/dataset_cache
  export HF_DATASETS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export OMP_NUM_THREADS=8

  python -m llamafactory.cli train /mnt/shared-storage-gpfs2/ai4scifm-gpfs02/wuyixin/code/sty-lmf/distill_kl_min.yaml
  rm -r /mnt/shared-storage-gpfs2/ai4scifm-gpfs02/wuyixin/code/sty-lmf/saves/qwen3-4b/distill_kl_max/checkpoint-*/global_step*
"

rjob submit --gpu=8 --memory=1000000 -P 1 --cpu=128 --priority 8 --name=distill_max_kl \
--host-network=true \
-e DISTRIBUTED_JOB=true \
--charged-group=protfma_gpu --private-machine=group \
--custom-resources rdma/mlnx_shared=8 \
--custom-resources mellanox.com/mlnx_rdma=1 \
--mount=gpfs://gpfs2/ai4scifm-gpfs02:/mnt/shared-storage-gpfs2/ai4scifm-gpfs02 \
--image=registry.h.pjlab.org.cn/ailab/pytorch2.7.0-cuda12.8.1-py3.12-ubuntu24.04:v2 \
-- bash -c "
  CONDA_PATH=/mnt/shared-storage-gpfs2/ai4scifm-gpfs02/wuyixin/miniconda3
  . \${CONDA_PATH}/etc/profile.d/conda.sh && conda init && conda activate ssttyy

  cd /mnt/shared-storage-gpfs2/ai4scifm-gpfs02/wuyixin/code/sty-lmf
  export NCCL_TIMEOUT=18000
  export NCCL_DEBUG=INFO
  export HF_DATASETS_CACHE=/mnt/shared-storage-gpfs2/ai4scifm-gpfs02/wuyixin/data/dataset_cache
  export HF_DATASETS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export OMP_NUM_THREADS=8

  python -m llamafactory.cli train /mnt/shared-storage-gpfs2/ai4scifm-gpfs02/wuyixin/code/sty-lmf/distill_kl_max.yaml
  rm -r /mnt/shared-storage-gpfs2/ai4scifm-gpfs02/wuyixin/code/sty-lmf/saves/qwen3-4b/distill_kl_max/checkpoint-*/global_step*
"