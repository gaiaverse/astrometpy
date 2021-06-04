import sys, os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

sys.path.append('../')
import tqdm, astromet, h5py, astropy, scipy, numpy as np
import astropy.units as u
from multiprocessing import Pool

# Load data
gums_file = '/data/vault/asfe2/Conferences/EDR3_workshop/gums_sample_reparameterised.h'
with h5py.File(gums_file, 'r') as f:
    gums = {}
    for key in f.keys():
        gums[key] = f[key][...]


# Get Earth barycenter
times = np.linspace(2014.6, 2017.5,100000)
print('Getting earth barycenter coordinates...')
pos_earth = astropy.coordinates.get_body_barycentric('earth', astropy.time.Time(times, format='jyear'), ephemeris="de430")
pos_earth =  np.array([pos_earth.x.to(u.AU) / u.AU,
                     pos_earth.y.to(u.AU) / u.AU,
                     pos_earth.z.to(u.AU) / u.AU]).T
pos_earth_interp = scipy.interpolate.interp1d(times, pos_earth.T, bounds_error=False, fill_value=0.)


# Load scanning law
import scanninglaw.times
from scanninglaw.source import Source
from scanninglaw.config import config
config['data_dir'] = '/data/asfe2/Projects/testscanninglaw'
dr3_sl=scanninglaw.times.Times(version='dr3_nominal')




# Run fits
results = {}
def fit_object(isource, return_dict=False):

    params=astromet.params()

    for key in ['ra','dec','pmdec','parallax']:
        setattr(params, key, gums[key][isource])
        params.pmrac = gums['pmra'][isource]
    if gums['binary'][isource]:
        # Binaries
        for key in ['period','l','q','a','e',
                  'vtheta','vphi','vomega','tperi']:
            setattr(params, key, gums[key][isource])
    else:
        # Single sources - no binary motion
        setattr(params, 'a', 0)

    if params.e==0:
        params.e+=1e-10

    c = Source(float(params.ra),float(params.dec),unit='deg',frame='icrs')
    sl=dr3_sl(c, return_times=True, return_angles=True)
    # sl = {'times': [np.array([2016,2016.5,2017]), np.array([2016,2016.5,2017])],
    #       'angles': [np.array([10.,50.,70.]), np.array([10.,50.,70.])]}

    ts=2010+np.hstack(sl['times']).flatten()/365.25
    sort=np.argsort(ts)
    ts=ts[sort].astype(float)
    phis=np.hstack(sl['angles']).flatten()[sort].astype(float)

    trueRacs,trueDecs=astromet.track(ts,params,earth_barycenter=pos_earth_interp)

    al_err = astromet.sigma_ast(gums['phot_g_mean_mag'][isource])
    t_obs,x_obs,phi_obs,rac_obs,dec_obs=astromet.mock_obs(ts,phis,trueRacs,trueDecs,err=al_err)

    # EDR3
    fitresults=astromet.fit(t_obs,x_obs,phi_obs,al_err,params.ra,params.dec, earth_barycenter=pos_earth_interp)
    gaiaedr3_output=astromet.gaia_results(fitresults)

    # DR2
    subset = t_obs<2016.391467761807
    fitresults=astromet.fit(t_obs[subset],x_obs[subset],phi_obs[subset],al_err,params.ra,params.dec, earth_barycenter=pos_earth_interp)
    gaiadr2_output=astromet.gaia_results(fitresults)

    gaia_output = {}
    gaia_output['system_id'] = gums['system_id'][isource]
    for key in gaiadr2_output.keys():
        gaia_output[key+'_dr2'] = gaiadr2_output[key]
        gaia_output[key+'_edr3'] = gaiaedr3_output[key]

    # global results
    # for key in gaia_output.keys():
    #     try: results[key].append(gaia_output[key])
    #     except KeyError: results[key] = [gaia_output[key]]

    if return_dict: return gaia_output

    output = []
    for key in gaia_keys:
        output.append(gaia_output[key])

    return output


isources = np.argwhere(gums['unresolved']|~gums['binary'])[:,0]

gaia_keys = fit_object(isources[0], return_dict=True).keys()
print(gaia_keys)

# def fit_object(isource):
#     print(isource)
#     global results
#     try: results['a'].append(isource)
#     except KeyError: results['a'] = [isource]
#     results['b'] = 20
#results = {'system_id':[], 'phot_g_mean_mag':[]}

# Parallel
with Pool(40) as pool:
    pool_output = list(tqdm.tqdm(pool.imap(fit_object, isources), total=len(isources)))
for gaia_output in pool_output:
    for i, key in enumerate(gaia_keys):
        try: results[key].append(gaia_output[i])
        except KeyError: results[key] = [gaia_output[i]]


# Serial
# for isource in tqdm.tqdm(isources, total=len(isources)):
#     gaia_output = fit_object(isource)
#     for key in gaia_output.keys():
#         try: results[key].append(gaia_output[key])
#         except KeyError: results[key] = [gaia_output[key]]





# Save results
save_file = '/data/vault/asfe2/Conferences/EDR3_workshop/gums_fits_unresolved_dr2&3.h'
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
