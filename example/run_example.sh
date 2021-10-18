mkdir -p output
../hallucination/design.py --pdb=input/5ius_A.pdb --out=output/test1 --mask_v2=25-35,A119-140,15-25,A63-82,0-15 --opt_rate=0.2 --opt_iter=300 --init_sd=0.01 --feat_drop=0.2 --num=2 
