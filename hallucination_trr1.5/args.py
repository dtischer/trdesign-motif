import argparse

def get_args_design():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("PDB", type=str, help="input PDB with binding site residues")
    parser.add_argument("DIR", type=str, help="path for saving predictions")

    parser.add_argument('--len', type=int, dest='len', default=100,  help='sequence length')
    parser.add_argument('--nseq', type=int, dest='nseq', default=1, help='number of hallucinated sequences to be added to the MSA')
    parser.add_argument('--nrun', type=int, dest='nrun', default=10,  help='number of models to generate')
    parser.add_argument('--nmin', type=int, dest='nmin', default=100, help='number of minimization steps')

    parser.add_argument('--w_sat', type=float, dest='ws', default=1.0, help='weight for the satisfaction term')
    parser.add_argument('--w_consist', type=float, dest='wc', default=1.0, help='weight for the consistency term')
    parser.add_argument('--w_clash', type=float, dest='wcl', default=1.0, help='weight for the clash term')
    #parser.add_argument('--w_cav', type=float, dest='wcav', default=0.1, help='weight for the cavity term')
    parser.add_argument('--drop', type=float, dest='drop', default=0.5, help='dropout rate for the input features')
    parser.add_argument('--step', type=float, dest='step', default=1.0, help='NGD minimization step size')

    parser.add_argument('--beta0', type=float, dest='b0', default=2.0, help='inverse temperature at the beginning of the minimization')
    parser.add_argument('--beta1', type=float, dest='b1', default=20.0, help='inverse temperature at the end of the minimization')

    parser.add_argument('--sample', dest='sample', action='store_true', help='perform NGD+sample')
    parser.set_defaults(sample=False)
    
    args = parser.parse_args()

    return args


def get_args_refine():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("NPZ", type=str, help="input NPZ")
    parser.add_argument("PDB", type=str, help="input PDB with bs residues")
    parser.add_argument("OUT", type=str, help="output (refined) NPZ")

    parser.add_argument('-r', "--replicas=", type=int, required=False, dest='NREP', default=8, help="number of replicas")
    parser.add_argument('-t0', "--TEMP0=", type=float, required=False, dest='T0', default=0.01, help="starting temperature")
    parser.add_argument('-t1', "--TEMP1=", type=float, required=False, dest='T1', default=0.001, help="final temperature")
    parser.add_argument("-n", "--nsteps=", type=int, required=False, dest='NSTEPS', default=1000, help='number of annealing steps')
    parser.add_argument('-m', "--mode=", type=int, required=False, dest='MODE', default=3, help="mixing mode")

    parser.add_argument('--w_kl', type=float, dest='wkl', default=1.0, help='weight for the hallucination term')
    parser.add_argument('--w_cce', type=float, dest='wcce', default=1.0, help='weight for the fixed-backbone term')
    parser.add_argument('--w_aa', type=float, dest='waa', default=0.0, help='weight for the aa composition biasing loss term')
    parser.add_argument('--w_clash', type=float, dest='wcl', default=1.0, help='weight for the clash term')

    parser.add_argument("--rm_aa=", type=str, required=False, dest='RM_AA', default="",
                        help="disable specific amino acids from being sampled (ex: 'C' or 'W,Y,F')")
    
    args = parser.parse_args()

    return args