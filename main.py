import argparse
import numpy as np
from em_util.io import read_yml, read_vol, write_h5
from em_util.eval import adapted_rand
from agglo import agglo_waterz, agglo_branch


def get_arguments():
    """
    Get command line arguments for converting ground truth segmentation to a graph of skeleton.

    Returns:
        argparse.Namespace: Parsed command line arguments including seg_path, seg_resolution, output_path, and num_thread.
    """
    parser = argparse.ArgumentParser(
        description="waterz for 3D agglomeration"
    )
    parser.add_argument(
        "-d",
        "--data",
        choices=['snemi', 'axonem-m', 'axonem-h', 'j0126'],
        help="dataset",
        default='snemi',
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=['waterz', 'branch'],
        help="method",
        default='branch',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="output file",
        default="",
    )   
    return parser.parse_args()



if __name__== "__main__":
    # sa agg
    # python main.py -d snemi - m branch
    args = get_arguments()
    conf = read_yml('data/data.yml')
    
    data = conf[args.data]    
    aff = read_vol(data['aff'])[0]
    gt = None if 'gt' not in data else read_vol(data['gt'])
    
    if args.method == "waterz":
        out = agglo_waterz(aff)
    elif args.method == "branch":
        out = agglo_branch(aff)        
         
    if gt is not None:
        if args.data == 'snemi':
            if isinstance(out, list):
                for i in range(len(out)):      
                    print(adapted_rand(out[i].astype(int), gt))
            else:
                print(adapted_rand(out.astype(int), gt))
             
    if args.output != '': 
        write_h5(args.output, out)         