# +
#!/usr/bin/env python

""" Methods for reading 0D (scalar), 1D (vector) and 2D (array) variables 
from CloudSat data files.  These methods will work with data files that are
compliant with HDF-EOS file specifications.
"""

# --------------------------------------------------------------------------
# Methods for reading variables from CloudSat files.
#
# Copyright (c) 2015 Norman Wood
#
# This file is part of the free software CloudSat_tools:  you can use,
# redistribute and/or modify it under the terms of the BSD 3-Clause
# License.
#
# The software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# BSD 3-Clause License for more details
#
# You should have received a copy of the BSD 3-Clause License in the file
# LICENSE distributed along with this software.  If not, see
# <https://spdx.org/licenses/BSD-3-Clause.html>.
# --------------------------------------------------------------------------


"""
Examples of use:

Scalar variable
###############

import pyhdf.HDF
import CloudSat_tools.read_var_eos

f_ptr = pyhdf.HDF.HDF(filename, pyhdf.HDF.HC.READ)
var = CloudSat_tools.read_var_eos.get_0D_var(f_ptr, varname)
f_ptr.close()


1D variable
###########

import pyhdf.HDF
import CloudSat_tools.read_var_eos

f_ptr = pyhdf.HDF.HDF(filename, pyhdf.HDF.HC.READ)
var = CloudSat_tools.read_var_eos.get_1D_var(f_ptr, varname)
f_ptr.close()

This can be used to read 1D variables (like 'Latitude'
and 'Longitude') and also attributes (like 'Latitude.factor')

1D variables are returned in the shape [N_profiles, 1]

2D variable
###########

import pyhdf.SD
import pyhdf.HDF
import CloudSat_tools.read_var_eos

f_SD_ptr = pyhdf.SD.SD(filename, pyhdf.SD.SDC.READ)
f_VD_ptr = pyhdf.HDF.HDF(filename, pyhdf.HDF.HC.READ)
var = CloudSat_tools.read_var_eos.get_2D_var(f_SD_ptr, f_VD_ptr, varname)
f_VD_ptr.close()
f_SD_ptr.end()

2D variables are returned in the shape [N_profiles, N_bins]

MOD06 3D variable
#################

import pyhdf.SD
import pyhdf.HDF
import CloudSat_tools.read_var_eos

f_SD_ptr = pyhdf.SD.SD(filename, pyhdf.SD.SDC.READ)
f_VD_ptr = pyhdf.HDF.HDF(filename, pyhdf.HDF.HC.READ)
var = CloudSat_tools.read_var_eos.get_3D_mod06_var(f_SD_ptr, f_VD_ptr, varname)
f_VD_ptr.close()
f_SD_ptr.end()

The MOD06-5KM-AUX datasets include 3D variables that are dimensioned differently than
variables in the standard CloudSat data products and whose scale factors and offsets
are handled differently.  The dimensions are (N_bands, N_profiles, N_MODIS_granule_indices).
N_MODIS_granule_indices is usually 1 for MOD06-5KM-AUX datasets.  The scale factors and offsets
are contained in VD variables with names varname_add_offset and varname_scale factor.  These
scale factor and offset values vary depending on the particular MODIS granule from which
the varname variable values were obtained and so are 1D arrays.  The index needed to select
the appropriate array elements is given in the MODIS_granule_index variable.

"""

# +

import numpy
import warnings
import datetime
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pyhdf.HDF
    import pyhdf.VS
    import pyhdf.SD


# -

def get_profile_times(file_VS_ptr):
    time_offset_seconds = get_1D_var(file_VS_ptr, 'Profile_time')[:, 0]
    UTC_start_time_seconds = get_0D_var(file_VS_ptr, 'UTC_start')
    # Construct the absolute times.  Get 00Z of the granule doy
    start_time_string = get_0D_var(file_VS_ptr, 'start_time')
    YYYYmmdd = start_time_string[0:8]
    base_time = datetime.datetime.strptime(YYYYmmdd, '%Y%m%d')
    UTC_start_offset = datetime.timedelta(seconds=UTC_start_time_seconds)
    profile_times = numpy.array(
        [base_time + UTC_start_offset + datetime.timedelta(seconds=x) for x in time_offset_seconds])
    return profile_times


def get_0D_var(file_VS_ptr, varname):
    vs = file_VS_ptr.vstart()
    var = vs.attach(varname)
    tmp = var.read(1)
    var_value = tmp[0][0]
    # Check for factor and offset for this variable
    factor_label = "%s.factor" % (varname,)
    factor_ref_no = vs.find(factor_label)
    offset_label = "%s.offset" % (varname,)
    offset_ref_no = vs.find(offset_label)

    if offset_ref_no is not 0:
        # Get the offset value
        vd_tmp = vs.attach(offset_ref_no)
        offset = vd_tmp[0][0]
        var_value = var_value - offset
        vd_tmp.detach()
    if factor_ref_no is not 0:
        vd_tmp = vs.attach(factor_ref_no)
        factor = vd_tmp[0][0]
        var_value = var_value/factor

    var.detach()
    return(var_value)


def get_1D_var(file_VS_ptr, varname, dtype=None):
    vs = file_VS_ptr.vstart()
    var = vs.attach(varname)
    var_info = var.inquire()
    var_nRecs = var_info[0]
    tmp = var.read(var_nRecs)
    var_values = numpy.array(tmp)

    # Check for factor and offset for this variable
    factor_label = "%s.factor" % (varname,)
    factor_ref_no = vs.find(factor_label)
    offset_label = "%s.offset" % (varname,)
    offset_ref_no = vs.find(offset_label)

    if offset_ref_no is not 0:
        # Get the offset value
        vd_tmp = vs.attach(offset_ref_no)
        offset = vd_tmp[0][0]
        var_values = var_values - offset
        vd_tmp.detach()
    if factor_ref_no is not 0:
        vd_tmp = vs.attach(factor_ref_no)
        factor = vd_tmp[0][0]
        var_values = var_values/factor

    var.detach()
    vs.end()

    if dtype is not None:
        # Try to convert to the requested datatype
        var_values = var_values.astype(dtype)
    return(var_values)


def get_2D_var(file_SD_ptr, file_VS_ptr, varname):
    var = file_SD_ptr.select(varname)
    var_values = var[:]

    # Check for factor and offset for this variable
    vs = file_VS_ptr.vstart()
    factor_label = "%s.factor" % (varname,)
    factor_ref_no = vs.find(factor_label)
    offset_label = "%s.offset" % (varname,)
    offset_ref_no = vs.find(offset_label)

    if offset_ref_no is not 0:
        # Get the offset value
        vd_tmp = vs.attach(offset_ref_no)
        offset = vd_tmp[0][0]
        var_values = var_values - offset
        vd_tmp.detach()
    if factor_ref_no is not 0:
        vd_tmp = vs.attach(factor_ref_no)
        factor = vd_tmp[0][0]
        var_values = var_values/factor

    vs.end()
    return(var_values)


def get_3D_mod06_var(file_SD_ptr, file_VS_ptr, varname):

    # Some MOD06 variables have an additional dimension, mod_1km, which indicates the
    # particular MODIS granule used to produce the collocated CloudSat-MODIS data.
    # Scale factors and offsets are allowed to vary as a function of this MODIS granule index,
    # so this needs to be taken into account when extracting these 3D data from the MODO6 files.

    # This was written for use with the Brightness_Temperature variable in particular.
    # Verify it works with other 3D MOD06 variables before using it with them.

    var = file_SD_ptr.select(varname)
    var_values = var[:]

    MODIS_granule_indices = file_SD_ptr.select('MODIS_granule_index')
    MODIS_granule_index_values = MODIS_granule_indices[:]

    # Get the valid range for MODIS_granule_index
    vs = file_VS_ptr.vstart()
    valid_range_label = 'MODIS_granule_index.valid_range'
    valid_range_ref_no = vs.find(valid_range_label)
    vd_tmp = vs.attach(valid_range_ref_no)
    MODIS_granule_index_range = vd_tmp[0][0]
    MODIS_granule_idx_min = MODIS_granule_index_range[0]
    MODIS_granule_idx_max = MODIS_granule_index_range[1]

    # Check for factor and offset for this variable
    factor_label = "%s_scale_factor" % (varname,)
    factor_ref_no = vs.find(factor_label)
    if factor_ref_no != 0:
        # Get the factor value
        vd_tmp = vs.attach(factor_ref_no)
        factors = vd_tmp[:]
        vd_tmp.detach()

    offset_label = "%s_add_offset" % (varname,)
    offset_ref_no = vs.find(offset_label)
    if offset_ref_no is not 0:
        # Get the offset value
        vd_tmp = vs.attach(offset_ref_no)
        offsets = vd_tmp[:]
        vd_tmp.detach()

    # Create a variable to store the results
    results = numpy.empty(var_values.shape, dtype=numpy.float64)

    # Process each group of MODIS_granule_index values
    for granule_idx in range(MODIS_granule_idx_min, MODIS_granule_idx_max+1):
        # Select the var values that have this granule_idx
        granule_mask = numpy.equal(
            MODIS_granule_index_values[:, 0], granule_idx)
        granule_selection = numpy.nonzero(granule_mask)
        if numpy.any(granule_mask):
            factor = factors[granule_idx-1][0]
            offset = offsets[granule_idx-1][0]
            valid_mask = numpy.logical_not(numpy.equal(
                var_values[:, granule_mask, :], -32767))
            tmp = numpy.copy(
                var_values[:, granule_mask, :].astype(numpy.float64))
            # Apply factor and offset only to valid values
            tmp[valid_mask] = (tmp[valid_mask] - offset)*factor
            results[:, granule_mask, :] = tmp

    vs.end()
    return(results)


def secs_to_HHMMSS(utc_seconds):
    int_utc_seconds = utc_seconds.astype('int32')
    hh = int_utc_seconds/3600
    int_utc_seconds_remainder = int_utc_seconds - 3600*hh
    mm = int_utc_seconds_remainder/60
    int_utc_seconds_remainder = int_utc_seconds_remainder - 60*mm
    ss = int_utc_seconds_remainder
    return (hh, mm, ss)
