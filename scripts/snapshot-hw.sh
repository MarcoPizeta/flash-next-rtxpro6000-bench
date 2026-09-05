#!/usr/bin/env bash
# Dichiarazione hardware/software: cio' che manca a tutti i report pubblicati.
OUT=/opt/flash-next/bench/results/HARDWARE.md
{
echo "# Snapshot srv-lm — $(date '+%d/%m/%Y %H:%M')"
echo; echo "## GPU"; nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,power.default_limit,temperature.gpu,clocks.max.sm,clocks.max.mem,pcie.link.gen.current,pcie.link.width.current --format=csv
echo; echo "## CPU"; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|MHz' | sed 's/^/    /'
echo; echo "## RAM"; free -h | head -2
sudo dmidecode -t memory | grep -E 'Bank Locator|^\s+Size:|Configured Memory Speed|Part Number|^\s+Rank' | paste - - - - - 2>/dev/null | grep -v 'No Module' | sed 's/^/    /'
echo; echo "## Disco modelli"; df -h /opt/hf-cache /opt/flash-next | tail -2
ROOTDEV=$(findmnt -no SOURCE /); lsblk -dno MODEL /dev/$(lsblk -no PKNAME "$ROOTDEV")
echo; echo "## OS / kernel"; . /etc/os-release; echo "$PRETTY_NAME, $(uname -r), docker $(docker --version | cut -d' ' -f3)"
echo; echo "## Immagini (digest)"
for i in vllm/vllm-openai:qwen38-flash-next ghcr.io/theroyallab/tabbyapi:cu13 lmsysorg/sglang:dev-cu13; do
  D=$(docker image inspect "$i" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo 'NON PRESENTE')
  printf '    %-40s %s\n' "$i" "$D"
done
echo; echo "## Versioni nei container"
docker run --rm --entrypoint python3 vllm/vllm-openai:qwen38-flash-next -c "import vllm,torch;print('    vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)" 2>/dev/null
docker run --rm --entrypoint python3 ghcr.io/theroyallab/tabbyapi:cu13 -c "import exllamav3,torch;print('    exllamav3',exllamav3.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)" 2>/dev/null
echo; echo "## Checkpoint (revision)"
cat /opt/hf-cache/hub/models--primitive-ai--Qwen3.8-Flash-Next-mixed-NVFP4-FP8/refs/main 2>/dev/null | sed 's/^/    primitive-ai mixed: /'
cat /opt/hf-cache/hub/models--primitive-ai--Qwen3.8-Flash-Next-PLE-quant/refs/main 2>/dev/null | sed 's/^/    PLE-quant: /'
grep -h '"commit_hash"' /opt/flash-next/exl3-4.05bpw/.cache/huggingface/download/*.metadata 2>/dev/null | head -1 | sed 's/^/    exl3 4.05bpw: /'
echo; echo "## Carico residuo"; docker ps --format '    {{.Names}}'
echo; echo "## tool-eval-bench"; "$HOME/.local/bin/tool-eval-bench" --version 2>/dev/null | head -1 | sed 's/^/    /'
} > "$OUT" 2>&1
cat "$OUT"
