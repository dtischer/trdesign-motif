mkdir -p output
../hallucination_grid/design.py --pdb=input/5ius_A.pdb --out=output/test2 --mask=25-35,A119-140,15-25,A63-82,0-15 --opt_rate=0.1 --opt_iter=200 --init_sd=0.1 --num=2 --feat_drop=0
