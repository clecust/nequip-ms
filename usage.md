
nequip-train -cn  tutorial_ecemc_debug.yaml   hydra.run.dir=/home/giga/code/nequip-ms/run

nequip-compile /home/giga/code/nequip-ms/run/best.ckpt ./model.nequip.pth --device cuda  --mode torchscript 