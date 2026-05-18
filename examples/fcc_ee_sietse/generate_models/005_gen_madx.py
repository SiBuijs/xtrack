import argparse
import xtrack as xt

parser = argparse.ArgumentParser(description='Generate split dipoles for given configuration.')
parser.add_argument('-c', '--configuration', type=str, default='z',
                    help="Configuration name (default: 'z')")
args = parser.parse_args()
configuration = args.configuration

temp_folder = f'../lattices/{configuration}/_temp'

env = xt.load(f'{temp_folder}/env_with_split_dip_bpms_corrs.json')

madx_src = env.fccee_p_ring.to_madx_sequence('fccee_p_ring')

with open(temp_folder + '/lattice.madx', 'w') as fid:
    fid.write(madx_src)