
nequip-train -cn  tutorial_ecemc_debug.yaml   hydra.run.dir=/home/giga/code/nequip-ms/run

nequip-compile /home/giga/code/nequip-ms/run/best.ckpt ./model.nequip.pth --device cuda  --mode torchscript 


nequip-compile /home/giga/code/nequip-ms/run/best.ckpt ./modelcsoa1c1.nequip.pth --device cuda  --mode torchscript --a1 1.0 --coefficient 1.0 --atomic_numbers 1 6 8