import sys, os, numpy as np

sys.path.append('../')
import tqdm, astromet, h5py, astropy
from multiprocessing import Pool

# Load data


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

gums['phot_g_mean_mag'] = np.where(gums['binary'], np.log10(10**(-gums['primary_mag_g']/2.5) + 10**(-gums['secondary_mag_g']/2.5)),
                                                   gums['primary_mag_g'])

max_proj_sep = gums['a']*gums['parallax']*(1+gums['e'])*np.cos(gums['vtheta'])
gums['unresolved']=np.flatnonzero((gums['binary']==True) & (max_proj_sep<180) & (gums['period']<30)) # unresolved binaries

gums['flipped'] = np.flatnonzero(gums['l']>1)
gums['l'][gums['flipped']] = 1/gums['l'][gums['flipped']]
gums['q'][gums['flipped']] = 1/gums['q'][gums['flipped']]


# Load scanning law
import scanninglaw.times
from scanninglaw.source import Source
from scanninglaw.config import config
config['data_dir'] = '/data/asfe2/Projects/testscanninglaw'
dr3_sl=scanninglaw.times.Times(version='dr3_nominal')

# Run fits
results = {}
def fit_object(isource):

    params=astromet.params()

    #print(gums['ra'].shape, gums['ra'][isource][0], type(gums['ra'][isource]))

    for key in ['ra','dec','pmra','pmdec','parallax']:
        setattr(params, key, gums[key][isource])
    if gums['binary'][isource]:
        # Binaries
        for key in ['period','l','q','a','e',
                  'vtheta','vphi','vomega','tperi']:
            setattr(params, key, gums[key][isource])
    else:
        # Single sources - no binary motion
        setattr(params, 'a', 0)

    #c = Source(params.ra,params.dec,unit='deg',frame='icrs')
    c = Source(float(params.ra),float(params.dec),unit='deg',frame='icrs')
    sl=dr3_sl(c, return_times=True, return_angles=True)
    ts=2010+np.hstack(sl['times']).flatten()/365.25
    sort=np.argsort(ts)
    ts=ts[sort].astype(float)
    phis=np.hstack(sl['angles']).flatten()[sort].astype(float)

    trueRacs,trueDecs=astromet.track(ts,params)

    # Need to change this to total magnitude
    al_err = astromet.sigma_ast(gums['phot_g_mean_mag'][isource])
    t_obs,x_obs,phi_obs,rac_obs,dec_obs=astromet.mock_obs(ts,phis,trueRacs,trueDecs,err=al_err)

    fitresults=astromet.fit(t_obs,x_obs,phi_obs,al_err,params.ra,params.dec)
    gaia_output=astromet.gaia_results(fitresults)

    gaia_output['system_id'] = gums['system_id'][isource]
    gaia_output['phot_g_mean_mag'] = gums['phot_g_mean_mag'][isource]
    gaia_output['flipped'] = gums['flipped'][isource]

    # global results
    # for key in gaia_output.keys():
    #     try: results[key].append(gaia_output[key])
    #     except KeyError: results[key] = [gaia_output[key]]

    return gaia_output


isources = np.argwhere(gums['unresolved'])[:1000,0]

# def fit_object(isource):
#     print(isource)
#     global results
#     try: results['a'].append(isource)
#     except KeyError: results['a'] = [isource]
#     results['b'] = 20
#results = {'system_id':[], 'phot_g_mean_mag':[]}

# Parallel
# with Pool(2) as pool:
#     pool_output = tqdm.tqdm(pool.imap(fit_object, isources), total=len(isources))
# for gaia_output in pool_output:
#     for key in gaia_output.keys():
#         try: results[key].append(gaia_output[key])
#         except KeyError: results[key] = [gaia_output[key]]


# Serial
for isource in tqdm.tqdm(isources, total=len(isources)):
    gaia_output = fit_object(isource)
    for key in gaia_output.keys():
        try: results[key].append(gaia_output[key])
        except KeyError: results[key] = [gaia_output[key]]





# Save results
save_file = '/data/vault/asfe2/Conferences/EDR3_workshop/gums_fits.h'
with h5py.File(save_file, 'w') as f:
    for key in results.keys():
        f.create_dataset(key, data=results[key])






# binaries=np.flatnonzero(gums['binary']==True)
# pllxs=1000/gums['barycentric_distance'] # mas
# semis=gums['semimajor_axis'] # au
# eccs=gums['eccentricity']
# # randomly generating viewing angles because I got too confused by the argument of pericentre
# vthetas=np.arccos(-1+2*np.random.rand(pllxs.size)) # rad
# vphis=2*np.pi*np.random.rand(pllxs.size) # rad
# vomegas=2*np.pi*np.random.rand(pllxs.size) # rad
# periods=gums['orbit_period']/astromet.T # years
# tperis=2016+gums['periastron_date']/astromet.T
# tot_mags=-2.5*np.log10(10**(-0.4*gums['primary_mag_g'])+10**(-0.4*gums['secondary_mag_g']))
# ls=10**(0.4*(gums['primary_mag_g']-gums['secondary_mag_g']))
# qs=gums['secondary_mass']/gums['primary_mass']
# misordered=np.flatnonzero(ls>1)
# ls[misordered]=1/ls[misordered]
# qs[misordered]=1/qs[misordered]
# max_proj_sep=semis*pllxs*(1+eccs)*np.cos(vthetas)
# ubins=np.flatnonzero((gums['binary']==True) & (max_proj_sep<180) & (periods<30)) # unresolved binaries
# #rbins=np.flatnonzero((gums['binary']==True) & (max_proj_sep>180)) # (partially) resolved binaries
# uras=gums['ra'][ubins]
# udecs=gums['dec'][ubins]
# upmras=gums['pmra'][ubins]
# upmdecs=gums['pmdec'][ubins]
# upllxs=pllxs[ubins]
# uperiods=periods[ubins]
# uas=semis[ubins]
# ues=eccs[ubins]
# uls=ls[ubins]
# uqs=qs[ubins]
# utperis=tperis[ubins]
# uvthetas=vthetas[ubins]
# uvphis=vphis[ubins]
# uvomegas=vomegas[ubins]
# umags=tot_mags[ubins]
