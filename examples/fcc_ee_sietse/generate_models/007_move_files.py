import argparse
import xtrack as xt
import shutil

parser = argparse.ArgumentParser(description='Generate split dipoles for given configuration.')
parser.add_argument('-c', '--configuration', type=str, default='z',
                    help="Configuration name (default: 'z')")
args = parser.parse_args()
configuration = args.configuration

temp_folder = f'../lattices/{configuration}/_temp'

# cp env with split dipoles, bpms, corrs
shutil.copyfile(f'{temp_folder}/env_final.json',
                f'../lattices/{configuration}/fccee_{configuration}.json')

# cp env with split dipoles, bpms, corrs
shutil.copyfile(f'{temp_folder}/env_final_merged_dipoles.json',
                f'../lattices/{configuration}/fccee_{configuration}_merged_dipoles.json')

# Copy python lattice file and strengths
shutil.copyfile(f'{temp_folder}/lattice_final.py',
                f'../lattices/{configuration}/fccee_{configuration}_lattice.py')
shutil.copyfile(f'{temp_folder}/strengths.py',
                f'../lattices/{configuration}/fccee_{configuration}_strengths.py')

# Copy madx file
shutil.copyfile(f'{temp_folder}/lattice.madx',
                f'../lattices/{configuration}/fccee_{configuration}.madx')



