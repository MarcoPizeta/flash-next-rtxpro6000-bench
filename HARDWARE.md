# Snapshot srv-lm — 04/09/2026 20:21

## GPU
name, driver_version, memory.total [MiB], power.limit [W], power.default_limit [W], temperature.gpu, clocks.max.sm [MHz], clocks.max.memory [MHz], pcie.link.gen.current, pcie.link.width.current
NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 595.84, 97887 MiB, 600.00 W, 600.00 W, 40, 3090 MHz, 14001 MHz, 5, 16

## CPU
    CPU(s):                                  48
    Model name:                              AMD Ryzen Threadripper 9960X 24-Cores
    Thread(s) per core:                      2
    CPU(s) scaling MHz:                      33%
    CPU max MHz:                             5489.7642
    CPU min MHz:                             1223.6230

## RAM
               total        used        free      shared  buff/cache   available
Mem:            60Gi       3.8Gi       8.7Gi       828Ki        48Gi        57Gi
    	Bank Locator: P0 CHANNEL E		Part Number: MTC20F1045S1RC48BA2           		Rank: 1		Configured Memory Speed: 4800 MT/s		Size: 32 GB
    	Bank Locator: P0 CHANNEL G		Part Number: MTC20F1045S1RC48BA2 JGCC      		Rank: 1		Configured Memory Speed: 4800 MT/s	

## Disco modelli
/dev/nvme0n1p2  915G  472G  398G  55% /
/dev/nvme0n1p2  915G  472G  398G  55% /
Samsung SSD 990 PRO 1TB

## OS / kernel
Ubuntu 26.04.1 LTS, 7.0.0-30-generic, docker 29.7.2,

## Immagini (digest)
    vllm/vllm-openai:qwen38-flash-next       vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
    ghcr.io/theroyallab/tabbyapi:cu13        ghcr.io/theroyallab/tabbyapi@sha256:90a01932dffcf7e1a93e231480d645a2e6cd02718e811d7ef5d31bb81734a1a2
    lmsysorg/sglang:dev-cu13                 lmsysorg/sglang@sha256:9ca30410d6280c09c8e6804525b8ba0ef598c456044e758819046f8ef3037020

## Versioni nei container
    vllm 0.1.dev20073+g8e685d198 torch 2.13.0+cu130 cuda 13.0

## Checkpoint (revision)
    primitive-ai mixed: adf14c6be32571ec8c8b5e85dbbdaed3800d8601
## Carico residuo
mon-cadvisor
pizeta-obb
open-webui
mon-gpu-exporter
portainer_agent
mon-prometheus
mon-grafana
mon-node-exporter
pizeta-ocr-v5
pizeta-ocr-ops

## tool-eval-bench
    tool-eval-bench 2.6.1.dev42+g6a98f0324
