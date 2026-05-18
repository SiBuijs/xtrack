import argparse
import xtrack as xt
import math

parser = argparse.ArgumentParser(description='Generate split dipoles for given configuration.')
parser.add_argument('-c', '--configuration', type=str, default='z',
                    help="Configuration name (default: 'z')")
args = parser.parse_args()
configuration = args.configuration

temp_folder = f'../lattices/{configuration}/_temp'

env = xt.load(f'{temp_folder}/env_no_bpms_merged_quads.json')
line = env.fccee_p_ring
ip_marker = xt.Marker()
print("Inserting relevant markers...")
# Insert injection marker
tt = line.get_table()
ind_l = list(tt['name']).index('qf13i')+1
ind_r = list(tt['name']).index('qf13j')
s_l = tt['s'][ind_l]
s_r = tt['s'][ind_r]
line.insert('septum',ip_marker,at=0.5*(s_l+s_r))
# Insert markers FF
ip_names_left = ['ip.7','ip.1','ip.3','ip.5']
ip_names = ['ip.0','ip.2','ip.4','ip.6']
ip_survey_names = ['ipa','ipd','ipg','ipj']
#for thing in line.element_names:
#    if thing.startswith('ip'):
#        #print(thing)
for ii,thing in enumerate(ip_survey_names):
    line.insert(thing,ip_marker,at=0,from_=ip_names[ii])    
    ind = line.element_names.index(ip_names[ii])
    beyond_scrab=False
    for  mm in range(10000):
        ind_temp=ind+mm
        if line.element_names[ind_temp].startswith("scrab"):
            beyond_scrab=True
        if beyond_scrab==True and (isinstance(line.element_dict[line.element_names[ind_temp]],xt.RBend) or isinstance(line.element_dict[line.element_names[ind_temp]],xt.Bend)):
            #print(line.element_names[ind_temp])
            line.insert('end_straight_start_ds_'+thing,ip_marker,at=0,from_=line.element_names[ind_temp],from_anchor='start')
            #line.insert('start_ds',ip_marker,at=0,from_=line.element_names[ind_temp])
            beyond_scrab=False
        if line.element_names[ind_temp].startswith("qd1am"):
            #print(line.element_names[ind_temp])
            line.insert('end_ds_start_arc_'+thing,ip_marker,at=0,from_=line.element_names[ind_temp],from_anchor='start')
            break
    ind = line.element_names.index(ip_names_left[ii])
    beyond_scrab=False
    for  mm in range(10000):
        ind_temp=ind-mm
        if line.element_names[ind_temp].startswith("scrab"):
            beyond_scrab=True
        if beyond_scrab==True and (isinstance(line.element_dict[line.element_names[ind_temp-1]],xt.RBend) or isinstance(line.element_dict[line.element_names[ind_temp-1]],xt.Bend)):
            #print(line.element_names[ind_temp-1])
            line.insert('end_ds_start_straight_'+thing,ip_marker,at=0,from_=line.element_names[ind_temp-1],from_anchor='end')
            beyond_scrab=False
            #line.insert('start_ds',ip_marker,at=0,from_=line.element_names[ind_temp])
        if line.element_names[ind_temp-1].startswith("qd1am"):
            #print(line.element_names[ind_temp-1])
            line.insert('end_arc_start_ds_'+thing,ip_marker,at=0,from_=line.element_names[ind_temp-1],from_anchor='end')
            break
    #line.remove(ip_names[ii])
    #line.remove(ip_names_left[ii])
    #del env[ip_names[ii]]
    #del env[ip_names_left[ii]]
# Insert marker Straights
ele_names = ['qf13i','qd18c.0','qd18f.0','qd18d.0']
survey_names=['ipb','ipf','iph','ipl']
ip_marker = xt.Marker()
for ii,thing in enumerate(ele_names):
    #print("\n",thing,ii)
    ind = line.element_names.index(ele_names[ii])
    temp_ip = survey_names[ii]
    #print("\n",thing,ii,ind)
    placed=False
    for  mm in range(10000):
        ind_temp=ind+mm
        if placed==False and (line.element_names[ind_temp].startswith('dl') or line.element_names[ind_temp].startswith('ds')) and (isinstance(line.element_dict[line.element_names[ind_temp]],xt.RBend) or isinstance(line.element_dict[line.element_names[ind_temp]],xt.Bend)):
            #print(line.element_names[ind_temp])
            line.insert('end_straight_start_ds_'+temp_ip,ip_marker,at=0,from_=line.element_names[ind_temp],from_anchor='start')
            placed=True
        if line.element_names[ind_temp].startswith("qd1am"):
            #print(line.element_names[ind_temp])
            line.insert('end_ds_start_arc_'+temp_ip,ip_marker,at=0,from_=line.element_names[ind_temp],from_anchor='start')
            break
    placed=False
    for  mm in range(10000):
        ind_temp=ind-mm 
        if placed==False and  (line.element_names[ind_temp-1].startswith('dl') or line.element_names[ind_temp-1].startswith('ds')) and  (isinstance(line.element_dict[line.element_names[ind_temp-1]],xt.RBend) or isinstance(line.element_dict[line.element_names[ind_temp-1]],xt.Bend)):
            #print(line.element_names[ind_temp-1])
            line.insert('end_ds_start_straight_'+temp_ip,ip_marker,at=0,from_=line.element_names[ind_temp-1],from_anchor='end')
            placed=True
            #line.insert('start_ds',ip_marker,at=0,from_=line.element_names[ind_temp])
        if line.element_names[ind_temp-1].startswith("qd1am"):
            line.insert('end_arc_start_ds_'+temp_ip,ip_marker,at=0,from_=line.element_names[ind_temp-1],from_anchor='end')
            #print(line.element_names[ind_temp-1])
            break
print("done.")
env.to_json(f'{temp_folder}/env_no_bpms_merged_quads_markers.json')
