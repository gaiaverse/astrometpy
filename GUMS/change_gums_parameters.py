import sys
sys.path.append('../')
import h5py, numpy as np, astromet, astropy, scipy

gums_file = '/data/vault/asfe2/Conferences/EDR3_workshop/gums_sample.h'
with h5py.File(gums_file, 'r') as f:
    gums = {}
    for key in f.keys():
        gums[key] = f[key][...]

# Transform variables
gums['parallax'] = 1e3/gums['barycentric_distance']
gums['period'] = gums.pop('orbit_period')/astromet.T # years
gums['l'] = 10**(0.4*(gums['primary_mag_g']-gums['secondary_mag_g']))
gums['q'] = gums['secondary_mass']/gums['primary_mass']
gums['a'] = gums.pop('semimajor_axis')
gums['e'] = gums.pop('eccentricity')
gums['vtheta'] = np.arccos(-1+2*np.random.rand(gums['system_id'].size))#gums['periastron_argument']
gums['vphi'] = 2*np.pi*np.random.rand(gums['system_id'].size)#gums['longitude_ascending_node']
gums['vomega'] = 2*np.pi*np.random.rand(gums['system_id'].size)#gums['inclination']
gums['tperi'] = gums['periastron_date']

gums['phot_g_mean_mag'] = np.where(gums['binary'], -2.5*np.log10(10**(-gums['primary_mag_g']/2.5) + 10**(-gums['secondary_mag_g']/2.5)),
                                                   gums['primary_mag_g'])

max_proj_sep = gums['a']*gums['parallax']*(1+gums['e'])*np.cos(gums['vtheta'])
gums['unresolved']=(gums['binary']==True) & (max_proj_sep<180) & (gums['period']<30) # unresolved binaries

gums['flipped'] = (gums['l']>1)
gums['l'][gums['flipped']] = 1/gums['l'][gums['flipped']]
gums['q'][gums['flipped']] = 1/gums['q'][gums['flipped']]

# Save
new_file = '/data/vault/asfe2/Conferences/EDR3_workshop/gums_sample_reparameterised.h'
with h5py.File(new_file, 'w') as hf:
    for key in gums.keys():
        print(key)
        hf.create_dataset(key, data=gums[key])
