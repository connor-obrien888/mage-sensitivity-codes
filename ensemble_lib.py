import os, sys
import pandas as pd
import numpy as np
import h5py
from astropy.time import Time #For converting MJD to datetime
import kaipy.kaiH5 as kh5
import kaipy.gamera.magsphere as msph
import glob
import time
import matplotlib.pyplot as plt
import xarray as xr
import bezpy
import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from IPython.display import HTML
mpl.rc('animation', html='html5')
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely

ISOFMT = '%Y-%m-%dT%H:%M:%S.%f'

def extract_indices(rundir, runid, deltab_extension = ".deltab.h5"):
    '''
    Extracts SuperMAGE geomagnetic indices from dB results of a given MAGE run.
    Note that calcdb.x must be run on the MAGE results before using this tool.

    Parameters
    ----------
    rundir : str
        Path to MAGE run results, e.g. "/glade/derecho/scratch/cobrien/doctor-patient-day-prime/"
    runid : str
        Run ID stored at rundir, e.g. "msphere"

    Returns
    -------
    data : DataFrame
        Pandas DataFrame containing the extracted SuperMAGE indices.
    '''
    dbFile = os.path.join(rundir, runid+deltab_extension)

    db = msph.GamsphPipe(rundir, runid, doFast=False)
    gsph = msph.GamsphPipe(rundir,runid,doFast=False)

    nSteps = len(db.MJDs)

    SMR = []
    SMR06 = []
    SMR12 = []
    SMR18 = []
    SMR00 = []
    SML = []
    SMU = []
    SME = []
    CPCPN = []
    CPCPS = []

    for i in range(nSteps-1):
        SMR.append(kh5.PullAtt(dbFile,"SMR" ,i))
        SMR06.append(kh5.PullAtt(dbFile,"SMR_06",i))
        SMR12.append(kh5.PullAtt(dbFile,"SMR_12",i))
        SMR18.append(kh5.PullAtt(dbFile,"SMR_18",i))
        SMR00.append(kh5.PullAtt(dbFile,"SMR_00",i))
        SML.append(kh5.PullAtt(dbFile,"SML" ,i))
        SMU.append(kh5.PullAtt(dbFile,"SMU" ,i))
        SME.append(kh5.PullAtt(dbFile,"SME" ,i))

        cpcp = gsph.GetCPCP(i)
        CPCPN.append(cpcp[0])
        CPCPS.append(cpcp[1])


    data = pd.DataFrame(np.array([SMR, SMR06, SMR12, SMR18, SMR00, SML, SMU, SME, CPCPN, CPCPS]).T, columns = ['SMR', 'SMR06', 'SMR12', 'SMR18', 'SMR00', 'SML', 'SMU', 'SME', 'CPCPN', 'CPCPS'])
    data['Epoch'] = Time(db.MJDs[:-1], format = 'mjd').to_datetime()
    data['Epoch'] = pd.to_datetime(data['Epoch'], utc = True)
    return data

def extract_perturbations(rundir, runid, deltab_extension = ".deltab.h5"):
    '''
    Extracts magnetic perturbation components from dB results of given MAGE run.
    Returns timesteps of seven parameters on a geographic lat/lon grid.
    Parameters returned are, in order: 
    - `dBn`, the magnetic-northward perturbation
    - `dBp`, the phi GEO coordinate perturbation, called Y in GIC extraction function read_mage
    - `dBr`, the radial GEO coordinate perturbation
    - `dBt`, the theta GEO coordinate perturbation, called X in GIC extraction function read_mage
    - `smlat`, the latitude of this cell in SM coordinates
    - `smlon`, the longitude of this cell in SM coordinates
    - `|dB/dt|`, the naive time derivative of the perturbations

    If the file read includes the contributions from magnetospheric, ionospheric, and field-aligned currents separately, the parameters returned are instead:
    ['dBn', 'dBp', 'dBpF', 'dBpI', 'dBpM', 'dBr', 'dBrF', 'dBrI', 'dBrM', 'dBt', 'dBtF', 'dBtI', 'dBtM', 'dbJ', 'dbJF', 'dbJI', 'dbJM', 'smlat', 'smlon', '|dB/dt|']
    Which are the same as above, but with M, I, and F denoting the magnetospheric, ionospheric, and field-aligned current contributions.
    'dBJM', etc. are the current densities from each source via Ampere's law in each cell.

    Parameters
    ----------
    rundir : str
        Path to MAGE run results, e.g. "/glade/derecho/scratch/cobrien/doctor-patient-day-prime/"
    runid : str
        Run ID stored at rundir, e.g. "msphere"
    detlab_extension : str
        Extension for deltab file to be read, default ".deltab.h5".

    Returns
    -------
    data : ndarray
        numpy array containing the extracted magnetic perturbations.
        Shape [timesteps, latitudes, longitudes, parameters].
    info : kaipy.kaiH5.H5Info object
        HDF info as extracted by kaiH5. Useful for plotting. TODO: ditch this and just use xarray for data
    '''
    filepath = os.path.join(rundir, runid+deltab_extension)

    info = kh5.H5Info(filepath) # Get HDF info for timesteps, MJDs, etc.

    with h5py.File(filepath, "r") as file:
        data = np.empty((len(info.steps), file[f"/{info.stepStrs[0]}/smlat"].shape[1], file[f"/{info.stepStrs[0]}/smlat"].shape[2],len(list(file[f"/{info.stepStrs[0]}"]))+1))
        for idx, step in enumerate(info.stepStrs):
            for jdx, key in enumerate(file[f"/{step}"]):
                data[idx, :, :, jdx] = file[f"/{step}/{key}"]
        # print(list(file[f"/{step}"]))
        dofatio = True if (len(list(file[f"/{info.stepStrs[0]}"])) == 19) else False

    if dofatio:
        rind, pind, tind = 5, 1, 9 # Indices in data corresponding to each dB component (for |dB/dt|)
    else:
        rind, pind, tind = 2, 1, 3

    # Now we calculate dB/dt
    for i in range(data.shape[0]):
        if (i == 0) | (i == data.shape[0]-1): #Skip the first and last entries because we can't do their time derivative
            continue
        d1 = 0.5*(data[i+1, :, : , rind] - data[i-1, :, : , rind]) # dBr/dt (radial GEO)
        d2 = 0.5*(data[i+1, :, : , tind] - data[i-1, :, : , tind]) # dBt(heta)/dt (theta GEO)
        d3 = 0.5*(data[i+1, :, : , pind] - data[i-1, :, : , pind]) # dBp/dt (phi GEO)
        data[i, :, : , -1] = np.sqrt(d1**2.0 + d2**2.0 + d3**2.0)
    
    return data, info

def extract_solarwind(rundir, swfile = "bcwind.h5"):
    '''
    Extracts solar wind data from MAGE input file.

    Parameters
    ----------
    rundir : str
        Path to MAGE run results, e.g. "/glade/derecho/scratch/cobrien/doctor-patient-day-prime/"
    
    swfile : str, optional
        Solar wind file to read, default "bcwind.h5"

    Returns
    -------
    data : DataFrame
        Pandas DataFrame containing the extracted solar wind data.
    '''
    swFile = os.path.join(rundir, "bcwind.h5")

    vars = kh5.getRootVars(swFile)

    data = pd.DataFrame()

    for var in vars:
        data[var] = kh5.PullVar(swFile, var)
    
    data['Epoch'] = Time(data['MJD'], format = 'mjd').to_datetime()
    data['Epoch'] = pd.to_datetime(data['Epoch'], utc = True)
    return data

def bin_indices(data, n_bins = 24, lat_lim = [40, 80]):
    '''
    Bins data returned by extract_perturbations() by MLT and calculates SuperMAG indices in those bins.
    Parameters returned are, in order: 
    - `SMU`, the "upper" SuperMAG auroral index
    - `SML`, the "lower" SuperMAG auroral index
    - `SME`, the SuperMAG auroral index (SMU - SML)
    - `|dB/dt|`, the maximum dB/dt in each MLT bin
    - `MLT`, the magnetic local time center of each bin (to debug plots)
    
    Parameters
    ----------
    data : ndarray
        Data array extracted from MAGE deltab file by extract_perturbations()
    n_bins : int, optional
        Number of MLT bins from 0000 to 2359. Default 24.
    lat_lim : list of float, optional
        Latitude extent of bins [low, high]. 
        If comparing to SuperMAG directly, do not change from default!
        Default [40, 80].

    Returns
    -------
    im_arr : ndarray
        numpy array containing the binned indices.
        Shape [timesteps, bins, parameters].
    '''
    smlatind, smlonind, dbnind, dbdtind = -3, -2, 0, -1 # Cleverly, these are the same whether we have a fatio file or not
    bins = np.linspace(0, 24, n_bins+1) # MLT bin edges
    im_arr = np.empty((len(data), n_bins, 5))
    mlt = smlon_to_mlt(data[:,:,:,smlonind])
    for i in range(n_bins):
        lat_bool = (data[:,:,:,smlatind] >= lat_lim[0])&(data[:,:,:,smlatind] <= lat_lim[1])
        if i == 0:
            mlt_bool = (mlt >= bins[i])&(mlt <= bins[i+1])|(mlt >= bins[i-2])&(mlt <= bins[i-1]) #The first bin straddles midnight
        else:
            mlt_bool = (mlt >= bins[i-1])&(mlt <= bins[i+1]) # Other bins are easier

        dbn_ma = np.where(mlt_bool&lat_bool, data[:,:,:,dbnind], np.nan) # Mask out cells outside specified lat/lon range
        dbdt_ma = np.where(mlt_bool&lat_bool, data[:,:,:,dbdtind], np.nan)

        im_arr[:, i, 0] = np.nanmax(np.nanmax(dbn_ma, axis = 1), axis = 1) # Max dBn in the bin (SMU)
        im_arr[:, i, 1] = np.nanmin(np.nanmin(dbn_ma, axis = 1), axis = 1) # Min dBn in the bin (SML)
        im_arr[:, i, 3] = np.nanmax(np.nanmax(dbdt_ma, axis = 1), axis = 1) # Max dB/dt in the bin

    im_arr[:,:,2] = im_arr[:,:,0] - im_arr[:,:,1] # Calculate SME
    im_arr[:,:,4] = bins[:-1] # MLT bins

    return im_arr

def smlon_to_mlt(smlon):
    '''
    Utility function to convert SM longitude (degrees 0-360) to Magnetic Local Time (0 - 24)
    '''
    return ((smlon + 180.0) % 360.0) * (12.0 / 180.0)

def bin_real(data, n_bins = 24, lat_lim = [40, 80]):
    '''
    Bins data returned by SuperMAG API function SuperMAGGetData() in MLT bins
    Parameters returned are, in order: 
    - `SMU`, the "upper" SuperMAG auroral index
    - `SML`, the "lower" SuperMAG auroral index
    - `SME`, the SuperMAG auroral index (SMU - SML)
    - `|dB/dt|`, the maximum dB/dt in each MLT bin

    
    Parameters
    ----------
    data : DataFrame
        Data array returned by SuperMAG API function SuperMAGGetData()
    n_bins : int, optional
        Number of MLT bins from 0000 to 2359. Default 24.
    lat_lim : list of float, optional
        Latitude extent of bins [low, high]. 
        If comparing to SuperMAG directly, do not change from default!
        Default [40, 80].

    Returns
    -------
    im_arr : ndarray
        numpy array containing the binned indices.
        Shape [timesteps, bins, parameters].
    '''
    bins = np.linspace(0, 24, n_bins+1) # MLT bin edges
    time_range = pd.date_range(data['tval'].min(), data['tval'].max()+pd.Timedelta(minutes=1), freq = '1min')
    im_arr = np.empty((len(time_range)-1, n_bins, 4))
    for i in range(n_bins):
        for j in range(len(time_range)-1):
            lat_bool = ((90-data['mcolat']) >= lat_lim[0])&((90-data['mcolat']) <= lat_lim[1])
            if i == 0:
                mlt_bool = (data['mlt'] >= bins[i])&(data['mlt'] <= bins[i+1])|(data['mlt'] >= bins[i-2])&(data['mlt'] <= bins[i-1]) #The first bin straddles midnight
            else:
                mlt_bool = (data['mlt'] >= bins[i-1])&(data['mlt'] <= bins[i+1]) # Other bins are easier
            time_bool = (data['tval'] >= time_range[j])&(data['tval'] <= time_range[j+1])

            im_arr[j, i, 0] = np.nanmax(data.loc[mlt_bool&lat_bool&time_bool, 'N.nez']) # Max dBn in the bin (SMU)
            im_arr[j, i, 1] = np.nanmin(data.loc[mlt_bool&lat_bool&time_bool, 'N.nez']) # Min dBn in the bin (SML)
            im_arr[j, i, 3] = np.nanmax(data.loc[mlt_bool&lat_bool&time_bool, 'dBdt']) # Max dBdt in the bin (SML)

    im_arr[:,:,2] = im_arr[:,:,0] - im_arr[:,:,1] # Calculate SME

    return im_arr

def MJD2UT(mjd):
    """ If given single value, will return single datetime.datetime
    If given list, will return list of datetime.datetimes
    """
    UT = Time(mjd, format='mjd').isot
    if type(UT) == str:
        return pd.to_datetime(UT,format=ISOFMT)
    else:
        return [pd.to_datetime(UT[n],format=ISOFMT) for n in range(len(UT))]


def read_mage(fname):
    """Read the given MAGE deltab h5 file"""
    # The file of magnetic field data
    f = h5py.File(fname, "r")
    
    ntimes = len([x for x in f.keys() if "Step" in x])
    # Create the names manually to avoid sorting on non-zero padded strings
    steps = [f"Step#{i}" for i in range(ntimes)]

    # Calculate datetimes from the attributes
    times = [MJD2UT(f[step].attrs["MJD"]) for step in steps]

    # Set up the arrays
    lons = np.rad2deg(f["Phicc"][0, ...]).ravel()
    lons[lons > 180] -= 360
    lats = 90 - np.rad2deg(f["Thetacc"][0, ...]).ravel()
    
    dB = np.empty((ntimes, len(lons), 3))
    for i, step in enumerate(steps):
        dB[i, :, 0] = f[step]["dBr"][0, ...].ravel()  # Radial
        dB[i, :, 1] = -f[step]["dBt"][0, ...].ravel()  # Theta [colatitude] (X)
        dB[i, :, 2] = f[step]["dBp"][0, ...].ravel()  # Phi (Y)

    ds = xr.Dataset(coords={"longitude": ("site", lons),
                          "latitude": ("site", lats),
                          "component": ["R", "T", "P"],
                          "time": times},
                  data_vars={"B": (("time", "site", "component"), dB)})
    return ds

def read_impedance_file(fname, header_lines = 13):
    """
    Reads impedance file generated by bezpy for continental US
    """
    model_sites = {}

    with open(fname, 'r') as f:
        # 10 lines of header... Can this change?
        # YES! 0.25 degree files have 13
        for i in range(header_lines):
            f.readline()

        for line in f:
            elements = line.split()
            period = float(elements[0])
            name = elements[1]
            lat = float(elements[2])
            lon = float(elements[3])
            component = elements[7]
            val = float(elements[8]) + float(elements[9])*1j

            if name not in model_sites:
                site = bezpy.mt.Site3d(name)
                model_sites[name] = site
                site.latitude = lat
                site.longitude = lon
                # 45 periods in the dataset
                site.periods = np.zeros(45)
                site.Z = np.zeros((4, 45), dtype=complex)
                old_period = 0.
                period_counter = -1

            if period != old_period:
                period_counter += 1
                old_period = period
                site.periods[period_counter] = period

            if component == 'ZXX':
                loc = 0
            elif component == 'ZXY':
                loc = 1
            elif component == 'ZYX':
                loc = 2
            elif component == 'ZYY':
                loc = 3

            site.Z[loc,period_counter] = val
            
    # Calculate resistivity before returning
    [site.calc_resisitivity() for site in model_sites.values()]
        
    return model_sites

def create_impedance_mapping(impedance_sites, deltab_ds):
    """Create a mapping from impedance sites to magnetic field location"""
    impedance_xys = np.array([(x.longitude, x.latitude) for x in impedance_sites])
    # Store a mapping from k -> (i, j)
    # impedance site location (k) -> magnetic field location (i, j)
    # NOTE: The values are shifted by 1/8 degree due to cell centered vs edges
    #       between the two datasets
    mapping_dict = {}
    for k in range(len(impedance_xys)):
        #arr_i = np.argmin(abs(impedance_xys[k, 0] - ds["longitude"].to_numpy()) + abs(impedance_xys[k, 1] - ds["latitude"].to_numpy()))
        arr_i = np.argmin(abs(impedance_xys[k, 0] - deltab_ds["longitude"].values) + abs(impedance_xys[k, 1] - deltab_ds["latitude"].values))
        mapping_dict[k] = arr_i
    return mapping_dict

def calc_e(mapping_dict, b_ds, model_sites_sorted):
    """Calculate the electric field at the impedance locations."""
    b_model = b_ds["B"].values
    # Create an empty Electric field dataset that we can index into
    ntimes = b_model.shape[0]
    # We actually want B/E to be on the impedance site grid, not the original model grid
    b = np.zeros((ntimes, len(model_sites_sorted), 2))
    e = np.zeros_like(b)

    lons = []
    lats = []
    for site_num, site in enumerate(model_sites_sorted):
        lons.append(site.longitude)
        lats.append(site.latitude)
        # Pull out the magnetic field components for this site
        i = mapping_dict[site_num]
        b[:, site_num, 0], b[:, site_num, 1] = b_model[:, i, 1], b_model[:, i, 2]
        e[:, site_num, 0], e[:, site_num, 1] = site.convolve_fft(b[:, site_num, 0], b[:, site_num, 1], dt=60)
        
    # Create a new dataset
    ds = xr.Dataset(coords={"longitude": ("site", lons),
                          "latitude": ("site", lats),
                          "component": ["T", "P"],
                          "time": b_ds["time"]},  # Same as model times
                  data_vars={"B": (("time", "site", "component"), b),
                             "E": (("time", "site", "component"), e)})
    return ds

def load_bezpy_cache(bezpyDataPath, bezpyFilename):
    """
    Loads cached bezpy impedance data, then count and sort the grid of impedance sites
    
    :param bezpyDataPath: Path to cached bezpy data.
    :param bezpyFilename: Name of cached bezpy impedance file
    """
    # bezpyDataPath = '/glade/campaign/hao/msphere/gamshare/bezpy-data'
    # bezpyDataPath = '/glade/u/home/cobrien/ensemble-analysis/data/bezpy-data'
    # bezpyFilename = 'USA_impedance_gridded_45per_0.25x0.25-SP2.dat'
    fname = os.path.join(bezpyDataPath,bezpyFilename)
    model_sites_all = read_impedance_file(fname)

    model_sites = {} #Sort the impedance grid points by lat/lon
    for name in model_sites_all:
        site = model_sites_all[name]
        model_sites[name] = site
    del model_sites_all
    model_sites_sorted = sorted(model_sites.values(), key=lambda x: (x.longitude, x.latitude))

    site_xys = np.array([(x.longitude, x.latitude) for x in model_sites_sorted])
    return model_sites_sorted, site_xys

def get_transmission_lines(bezpyDataPath, transmission_file = "transmission_line_objects.pkl", site_xys = None):
    """
    Load power transmission line location/resistivity file
    
    :param bezpyDataPath:  Path to cached bezpy data.
    :param transmission_file: Filename of pickled tansmission line objects
    :param site_xys: Locations of resistivity model grid. Optional, since unused if loading pickled data. If not loading pickled data and not specified, will fail.
    """
    try:
        return pd.read_pickle(os.path.join(bezpyDataPath,transmission_file))
    except FileNotFoundError:
        # Only recalculate if the weights haven't been filled before
        print("No transmission line pickle files found, loading the data"
              "and calculating values manually which can be slow")
        df = gpd.read_file(os.path.join(bezpyDataPath,"Electric_Power_Transmission_Lines.shp"))
        # Change all MultiLineString into LineString objects by grabbing the first line
        # Will miss a few coordinates, but should be OK as an approximation
        df.loc[df["geometry"].apply(lambda x: x.geometryType()) == "MultiLineString","geometry"] = \
            df.loc[df["geometry"].apply(lambda x: x.geometryType()) == "MultiLineString","geometry"].apply(lambda x: x[0])

        # Get rid of erroneous 1MV and low power line voltages
        df = df[(df["VOLTAGE"] < 1000) & (df["VOLTAGE"] >= 200)]

        df["obj"] = df.apply(bezpy.tl.TransmissionLine, axis=1)
        df["length"] = df.obj.apply(lambda x: x.length)

        print("Starting delaunay calculations which can take ~20 minutes")
        t1 = time.time()
        df.obj.apply(lambda x: x.set_delaunay_weights(site_xys, use_gnomic=False))
        print(f"Done filling interpolation weights: {time.time() - t1} s".format())
        # Create the pickle object so we can load faster next time
        df.to_pickle(transmission_file)
        return df
    
def calc_voltages(df_lines, ds_E):
    """Calculate the voltages along the lines for the given electric field time-series"""
    ntimes = len(ds_E["time"])
    n_trans_lines = len(df_lines)
    arr_delaunay = np.zeros(shape=(ntimes, n_trans_lines))
    e_fields = ds_E["E"].values
    # Iterate over all transmission lines to do the integration
    for i, t_line in enumerate(df_lines.obj):
        arr_delaunay[:, i] = t_line.calc_voltages(e_fields, how='delaunay')
    
    # We need to add a new coordinate, the "line" which corresponds to the index
    # in the line dataframe
    ds_E = ds_E.assign_coords({"line": np.arange(n_trans_lines)})
    # gic_proxy = V / R = V / (resistivity * line_length)
    rho_line = 0.1  # Ohms/km (Grigsby 2007)
    ds_E = ds_E.assign({"V": (("time", "line"), arr_delaunay),
                        "I": (("time", "line"), arr_delaunay / ((rho_line * df_lines["length"].values)[np.newaxis, :]))})
    return ds_E

def generate_gic(mageDataPath, mageRunID, bezpyDataPath, deltab_extension = "deltab", bezpyFilename = 'USA_impedance_gridded_45per_0.25x0.25-SP2.dat', transmission_file = "transmission_line_objects.pkl"):
    """
    Generates GIC estimates from MAGE simulation stored at mageDataPath
    
    :param mageDataPath: Path to a stored MAGE simulation run. Must have a deltab.h5 file
    :param mageRunID: RunID of stored MAGE simulation.
    :param bezpyDataPath: Path to cached bezpy data for impedance and powrer line locations/resistivity
    """
    model_sites_sorted, site_xys = load_bezpy_cache(bezpyDataPath, bezpyFilename)
    transmission_df = get_transmission_lines(bezpyDataPath, transmission_file = transmission_file, site_xys = site_xys)
    metadata_dict = {
        'model_sites_sorted' : model_sites_sorted,
        'site_xys' : site_xys,
        'transmission_df' : transmission_df,
    }
    try:
        mage_ds = xr.load_dataset(os.path.join(mageDataPath,"mage_data.nc"))
    except FileNotFoundError:
        # Create the data and save it
        mage_ds = read_mage(os.path.join(mageDataPath,f"{mageRunID}.{deltab_extension}.h5"))
        mage_mapping = create_impedance_mapping(model_sites_sorted, mage_ds)
        mage_ds = calc_e(mage_mapping, mage_ds, model_sites_sorted)
        mage_ds = calc_voltages(transmission_df, mage_ds)
        mage_ds.to_netcdf(os.path.join(mageDataPath,"mage_data.nc"))
    return mage_ds, metadata_dict

def prep_plot_objects():
    scale = '10m'
    land = cfeature.NaturalEarthFeature('physical', 'land', scale,
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])
    coast = cfeature.NaturalEarthFeature(category='physical', scale=scale,
                                        edgecolor='k',
                                        facecolor='none', name='coastline')
    ocean = cfeature.NaturalEarthFeature(
            category='physical',
            name='ocean',
            scale=scale,
            facecolor='gray')
    rivers = cfeature.NaturalEarthFeature(
            category='physical',
            name='rivers_lake_centerlines',
            scale=scale,
            facecolor=cfeature.COLORS['water'],
            edgecolor='face')
    lakes = cfeature.NaturalEarthFeature(
            category='physical',
            name='lakes',
            scale=scale,
            facecolor=cfeature.COLORS['water'],
            edgecolor='face')

    states = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale=scale,
            facecolor='none',
            edgecolor='k')
    countries = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale=scale,
            facecolor='none',
            edgecolor='k')
    feature_dict = dict(
        land = land,
        coast = coast,
        ocean = ocean,
        rivers = rivers,
        lakes = lakes,
        states = states,
        countries = countries,
    )
    return feature_dict

def setup_axes(ax, bbox, feature_dict, oceancolor = cfeature.COLORS["water"], statealpha = 0.8, countryalpha = 1.0):
    """Setup an axes with the proper background, features, and extent."""
    ax.set_extent(bbox, ccrs.PlateCarree())
#     ax.add_feature(coast)
    ax.set_facecolor(oceancolor)
    ax.add_feature(cfeature.LAND, color=(0.8, 0.8, 0.8, 1))
    ax.add_feature(feature_dict['states'], alpha = statealpha)
    ax.add_feature(feature_dict['countries'], alpha = countryalpha)

# Set up the equations
def calc_line_width(x, log_scale=False, line_length_bounds = [10, 1000], line_width_bounds = [0.25, 2]):
    if log_scale:
        line_width = (np.log10(x) - np.log10(line_length_bounds[0]))/ \
        (np.log10(line_length_bounds[1]) - np.log10(line_length_bounds[0])) * \
        (line_width_bounds[1] - line_width_bounds[0]) + line_width_bounds[0]
    else:
        line_width = (x - line_length_bounds[0])/ \
        (line_length_bounds[1] - line_length_bounds[0]) * \
        (line_width_bounds[1] - line_width_bounds[0]) + line_width_bounds[0]
    return np.clip(line_width, a_min=line_width_bounds[0], a_max=line_width_bounds[1])

# bbox = (np.min(site_xys[:, 0]), np.max(site_xys[:, 0]),
#         np.min(site_xys[:, 1]), np.max(site_xys[:, 1]))

def init_grid(site_xys):
    """
    Initialize the plotting grid.
    
    :param site_xys: X and Y (lat/lon) positions of impedance grid
    """
    gridx, gridy = np.meshgrid(sorted(np.unique(site_xys[:, 0])),  # lons, lats
                           sorted(np.unique(site_xys[:, 1])))
    # Generate a list of index transfer functions to save and reuse later
    idxs = np.zeros(len(site_xys), dtype=int)
    for i in range(len(site_xys)):
        idxs[i] = np.argwhere((site_xys[i, 0] == gridx.ravel())
                          & (site_xys[i, 1] == gridy.ravel()))[0][0]
    return (gridx, gridy, idxs)

def fill_mesh(data, grid):
    """Fills the mesh with the data in the proper locations"""
    #NOTE: Requires "global" gridx.
    output = np.full(grid[0].ravel().size, np.nan, dtype=data.dtype)
    output[grid[2]] = data
    return output.reshape(grid[1].shape)

def plot_quantity(ax, ds, transmission_df, grid, t=0, alpha=0.5, quantity="B",
                  B_norm = mpl.colors.Normalize(0, 300),
                  B_cmap = mpl.colormaps["viridis"],
                  E_norm = mpl.colors.Normalize(0, 1000),
                  E_cmap = mpl.colormaps["viridis"],
                  I_norm = mpl.colors.Normalize(0, 3),
                  I_cmap = mpl.colormaps["plasma"],
                  ):
    """Plots the requested quantity from the dataset on the axes"""
    proj_data = ccrs.PlateCarree()
    line_coordinates = [np.array([[coord[0], coord[1]] for coord in linestring.coords]) for linestring in transmission_df['geometry']]
    line_widths = calc_line_width(transmission_df["length"], log_scale=True)
    if quantity == "I":
        # We need to add a LineCollection
        coll = mpl.collections.LineCollection(line_coordinates)
        coll.set_array(np.abs(ds["I"].values[t, :]))
        coll.set_cmap(I_cmap)
        coll.set_norm(I_norm)
        coll.set_transform(proj_data)
        coll.set_linewidths(line_widths)
        ax.add_collection(coll)
        return coll

    elif quantity == "B":
        norm = B_norm
        cmap = B_cmap
    elif quantity == "E":
        norm = E_norm
        cmap = E_cmap

    arr = ds[quantity].values
    arr = np.sqrt(arr[t, :, 0]**2 + arr[t, :, 1]**2)
    mesh = ax.pcolormesh(grid[0], grid[1], fill_mesh(arr, grid),transform=proj_data, cmap=cmap, norm=norm, alpha=alpha)
    return mesh

def plot_fft(deltab_data, t, deltab_index = 0, savename = None, nlon = 180, nlat = 90, lonres = 2, latres = 2):
    fs = np.fft.fft2(deltab_data[t, :, :, deltab_index], [nlon, nlat])
    psd = np.abs(np.fft.fftshift(fs))**2
    psd_fold = (psd[int(psd.shape[0]/2):, int(psd.shape[1]/2):] + # Top right
            np.flip(psd[int(psd.shape[0]/2):, :int(psd.shape[1]/2)], axis = 1) + # Bottom right mirrored vertically
            np.flip(psd[:int(psd.shape[0]/2), int(psd.shape[1]/2):], axis = 0) + # Top left flipped horizontally
            np.flip(np.flip(psd[:int(psd.shape[0]/2), :int(psd.shape[1]/2)], axis = 1), axis = 0)) # Bottom left flipped horizontally and vertically


    #NOTE: The original units/spacing of entries is 222.4km (2 degree on Earth's surface). 
    #      Thus the frequency space units are in cycles/111.2km.

    # Make the gridspec to put hists on top and right of figure, and colorbar below
    fig = plt.figure(figsize=(8.5, 8.5))
    gs_sup = mpl.gridspec.GridSpec(2, 1, figure = fig, height_ratios = [1, 0.05])
    gs = mpl.gridspec.GridSpecFromSubplotSpec(2, 2, height_ratios = [1, 3], width_ratios = [3, 1], hspace = 0, wspace = 0, subplot_spec = gs_sup[0])
    gs_cb = mpl.gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios = [2, 1], subplot_spec = gs_sup[1])

    imax = fig.add_subplot(gs[1, 0]) # Image of 2D PSD
    lonhistax = fig.add_subplot(gs[0, 0]) # Longitude-summed PSD hist
    lathistax = fig.add_subplot(gs[1, 1]) # Latitiude-summed PSD hist
    cbax = fig.add_subplot(gs_cb[0]) # Colorbar

    x_grid, y_grid = np.meshgrid(np.linspace(0, np.fft.fftshift(np.fft.fftfreq(nlon, d = lonres)).max(), psd_fold.shape[0]+1), np.linspace(0, np.fft.fftshift(np.fft.fftfreq(nlat, d = latres)).max(), psd_fold.shape[1]+1))
    im = imax.pcolormesh(
        x_grid, y_grid, psd_fold.T,
        norm = mpl.colors.LogNorm(vmin = 1e0, vmax = 1e12),
    )
    imax.set_xlabel(r"Longitudinal Scale Size ($\circ^{-1}$)")
    imax.set_ylabel(r"Latitiudinal Scale Size ($\circ^{-1}$)")
    lonhistax.step(np.linspace(0, np.fft.fftshift(np.fft.fftfreq(nlon, d = lonres)).max(), psd_fold.shape[0]), 
                np.sum(psd_fold, axis = 1), color = 'k')
    lonhistax.set_xticks([])
    lonhistax.set_xlim(0, np.fft.fftshift(np.fft.fftfreq(nlon, d = lonres)).max())
    lonhistax.set_yscale('log')

    lonhistax.set_ylabel(f'Summed PSD\n'+r'($nT^2/s^2/\circ^{-1}$)')
    lathistax.step(np.sum(psd_fold, axis = 0),
                np.linspace(0, np.fft.fftshift(np.fft.fftfreq(nlat, d = latres)).max(), psd_fold.shape[1]), color = 'k')
    lathistax.set_yticks([])
    lathistax.set_ylim(0, np.fft.fftshift(np.fft.fftfreq(nlon, d = lonres)).max())
    lathistax.set_xscale('log')
    lathistax.set_xlabel(f'Summed PSD\n'+r'($nT^2/s^2/\circ^{-1}$)')
    plt.colorbar(im, cax = cbax, label = r'PSD ($nT^2/s^2/\circ^{-1}$)', orientation = 'horizontal')

    plt.suptitle(f"Frame {t}")
    if savename is not None:
        plt.savefig(savename, bbox_inches = 'tight')

def fft_movie(deltab_data, t0 = 0, t1 = None, deltab_index = 0, savedir = "vidfft", nlon = 180, nlat = 90, lonres = 2, latres = 2):
    if t1 is None:
        trange = np.arange(t0, len(deltab_data))
    else:
        trange = np.arange(t0, t1)
    for t in trange:
        savename = savedir + "/fftvid.%04d.png"%(t)
        plot_fft(deltab_data, t, deltab_index=deltab_index, savename=savename, nlon=nlon, nlat=nlat, lonres=lonres, latres=latres)
        plt.close()

def update_quantity(coll, ds, gridx, gridy, 
                    t=0, quantity="B"):
    if quantity == "I":
        # Line collection, so just set_array with the new data
        coll.set_array(np.abs(ds[quantity].values[t, :]))
        return
    arr = ds[quantity].values
    arr = np.sqrt(arr[t, :, 0]**2 + arr[t, :, 1]**2)
    coll.set_array(fill_mesh(arr, gridx, gridy).ravel())

def lonlat_tf(lon, lat, nlat = 90, nlon = 180):
    '''
    Returns indices to query directly from MAGE deltab file to acquire particular lat/lon data.
    '''
    if lon < 0: # Unfurl longitude into monotonically increasing degrees
        lon_req = lon + 360
    else:
        lon_req = lon
    lon_idx = int(((nlon-1)/358) * (lon_req + 1))
    lat_idx = int(((lat/90)+1)*((nlat-1)/2))
    return lon_idx, lat_idx

def sh_atten(xarr, t_ind, lmax):
    '''
    Does a spherical harmonic attenuation to deltab file xarray.
    '''
    import pyshtools as pysh # TODO: Fix lazy optional dependency
    br_atten = pysh.SHCoeffs.from_least_squares(data = xarr['B'][t_ind, :, 0].data, latitude = xarr.latitude.data, longitude = xarr.longitude.data, weights = np.ones_like(xarr.latitude),lmax = lmax, units = 'nT').expand(lat = xarr.latitude.data, lon = xarr.longitude.data)
    bt_atten = pysh.SHCoeffs.from_least_squares(data = xarr['B'][t_ind, :, 1].data, latitude = xarr.latitude.data, longitude = xarr.longitude.data, weights = np.ones_like(xarr.latitude),lmax = lmax, units = 'nT').expand(lat = xarr.latitude.data, lon = xarr.longitude.data)
    bp_atten = pysh.SHCoeffs.from_least_squares(data = xarr['B'][t_ind, :, 2].data, latitude = xarr.latitude.data, longitude = xarr.longitude.data, weights = np.ones_like(xarr.latitude),lmax = lmax, units = 'nT').expand(lat = xarr.latitude.data, lon = xarr.longitude.data)
    xarr['B'][t_ind, :, 0] = br_atten
    xarr['B'][t_ind, :, 1] = bt_atten
    xarr['B'][t_ind, :, 2] = bp_atten
    return xarr

def fft_atten(xarr, f_ind, lonres = 2, latres = 2):
    '''
    Removes all longitudinal spatial frequencies from deltab file smaller than the f_ind'th frequency.
    Requires the lat/lon resolution of the deltab file to be specified since it is annoying to infer it from the xarray object.
    '''
    xarr_copy = xarr.copy(deep = True)
    nlon = len(np.unique(xarr_copy['B']['longitude'])) # Number of longitudes in grid
    nlat = len(np.unique(xarr_copy['B']['latitude']))
    lon_freq = np.fft.fftfreq(nlon, d = lonres) # Precompute the frequencies we decompose into
    #Precompute the attenuation factor that cuts the requested fluctuations:
    atten_fac = np.ones_like(lon_freq)
    atten_fac = np.where(np.abs(lon_freq)>=lon_freq[f_ind + 1], 0, atten_fac)
    for i in range(len(xarr_copy['B']['time'])): # Loop over all times the old fashioned way
        for j in range(3): # Loop over each component ezpz
            fs_atten = np.fft.fft2(xarr_copy['B'][i, :, j].data.reshape((nlon, nlat)), [nlon, nlat]) # Take the 2D fft of that component
            for k in range(nlon): # Apply the attenuation factor
                fs_atten[k, :] *= atten_fac[k]
            xarr_copy['B'][i, :, j] = np.fft.ifft2(fs_atten, [nlon, nlat]).ravel()
    return xarr_copy



def EKL(vx, vy, vz, by, bz):
    '''
    Returns Kan-Lee electric field in mV/m given solar wind V and B.
    Could work for magnetosheath, but the Kan-Lee E field was not developed for the sheath so it is not recommended.

    Parameters
    ----------
    vx : float, array-like
        Solar wind GSE X velocity in km/s
    vy : float, array-like
        Solar wind GSE Y velocity in km/s
    vz : float, array-like
        Solar wind GSE Z velocity in km/s
    by : float, array-like
        Interplanetary magnetic field GSM Y component in nT
    bz : float, array-like
        Interplanetary magnetic field GSM Z component in nT

    Returns
    -------
    ekl : float, array-like
        Kan-Lee electric field in mV/m
    '''
    ekl = 0.001*np.sqrt(vx**2+vy**2+vz**2)*np.sqrt(by**2+bz**2)*(np.sin(np.arctan2(by, bz)/2))**2
    return ekl

def r_quick(n, vx, vy, vz, bx, by, bz):
    """
    Calculates R_quick (mV/m) from solar wind n (cm^-3), V (km/s), and B (nT).
    See Borovsky and Birn 2013 for derivation. (doi.org/10.1002/2013JA019193)

    """
    v = np.sqrt(vx**2 + vy**2 + vz**2)  # Flow velocity magnitude (km/s)
    b = np.sqrt(bx**2 + by**2 + bz**2)  # IMF magnitude (nT)
    theta = np.arctan2(by, bz)  # IMF clock angle (radians)
    ma = (v * (n**0.5) / b) * 0.045846  # Alfven mach number
    c = (2.44e-4 + (1 + 1.38 * np.log(ma)) ** (-6)) ** (
        -1 / 6
    )  # Bow shock compression ratio
    beta = (ma / 6) ** 1.92  # Magnetosheath plasma beta
    r_q = (
        0.4
        * (np.sin(theta / 2) ** 2)
        * (c ** (-1 / 2))
        * (n ** (1 / 2))
        * (v**2)
        * ((1 + beta) ** (-3 / 4))
        * 4.5846e-5
    )  # Reconnection rate (mV/m)
    return r_q
