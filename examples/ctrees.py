"""Convert CTREES output to Millenium.

A control script to be used with `taoconvert` to convert CTREES output binary data
into HDF5 input for TAO.
"""

import re, os
import numpy as np
import pandas as pd
import tao

import gzip
import math 
# from collections import deque
from IPython.core.debugger import Tracer

# ### Taken from: http://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python
# def find_nearest_index(array, values):
#	idx = (np.abs(array-values)).argmin()
#	return idx   

_islistlike = lambda l: hasattr(l, '__iter__')
a2z = lambda a: 1./a - 1.
z2a = lambda z: 1./(1.+z)
opener = lambda filename,mode: gzip.open(filename,mode) if filename.endswith('.gz') else open(filename,mode)

def generate_filename(filebase):
        if filebase.endswith('.gz'):
                gzip_file = filebase
                uncompressed_file = filebase[:-3]
        else:
                gzip_file  = filebase+'.gz'
                uncompressed_file  = filebase
        
        if os.path.isfile(gzip_file):
                return gzip_file
        else:
                if os.path.isfile(uncompressed_file):
                        return uncompressed_file
                else:
                        return None
                

def read_file_possibly_gzipped(file,max_chunks):

        if not hasattr(file,'__read__'):
		try:
			f = open(file,'r')
		except IOError:
                        f = gzip.open(file+'.gz','rb')
		except:
			raise tao.ConversionError('Could not open file. Neither as ascii or gz')
	
	try:
		chunks = f.read(max_chunks)
	except EOFError:
		## not enough bytes. Just read-in the entire file then
		chunks = f.read()
        except:
                raise tao.ConversionError('Could not read in from the file')
		
	f.close
	return chunks

		
class BaseParseFields():
    def __init__(self, header, fields=None):
	if len(header)==0:
	    if all([isinstance(f, int) for f in fields]):
		self._usecols = fields
		self._formats = [float]*len(fields)
		self._names = ['f%d'%f for f in fields]
	    else:
		raise ValueError('header is empty, so fields must be a list '\
				     'of int.')
	else:
	    header_s = map(self._name_strip, header)
	    if fields is None or len(fields)==0:
		self._names = header
		names_s = header_s
		self._usecols = range(len(names_s))
	    else:
		if not _islistlike(fields):
		    fields = [fields]
		self._names = [header[f] if isinstance(f, int) else str(f) \
				   for f in fields]
		names_s = map(self._name_strip, self._names)
		wrong_fields = filter(bool, [str(f) if s not in header_s \
						 else '' for s, f in zip(names_s, fields)])
		if len(wrong_fields):
		    raise ValueError('The following field(s) are not available'\
					 ': %s.\nAvailable fields: %s.'%(\
			    ', '.join(wrong_fields), ', '.join(header)))
		self._usecols = map(header_s.index, names_s)
	    self._formats = map(self._get_format, names_s)

    def get_src_dtype(self):
            return np.dtype({'names':self._names, \
                                     'formats':self._formats})
            
    def parse_line(self, l):
	items = l.split()
	return tuple([c(items[i]) for i, c in \
			  zip(self._usecols, self._formats)])

    def pack(self, X):
	return np.array(X, np.dtype({'names':self._names, \
					 'formats':self._formats}))

    def _name_strip(self, s):
	return self._re_name_strip.sub('', s).lower()

    def _get_format(self, s):
	return float if self._re_formats.search(s) is None else int

    _re_name_strip = re.compile('\W|_')
    _re_formats = re.compile('^phantom$|^mmp$|id$|^num|num$')


class BaseDirectory:
    def __init__(self, dir_path='.'):	
	self.dir_path = os.path.expanduser(dir_path)

	#get file_index
	files = os.listdir(self.dir_path)
	matches = filter(lambda m: m is not None, \
			     map(self._re_filename.match, files))
	if len(matches) == 0:
	    raise ValueError('cannot find matching files in this directory: %s.'%(self.dir_path))
	indices = np.array(map(self._get_file_index, matches))
	s = indices.argsort()
	self.files = [matches[i].group() for i in s]
	self.file_indices = indices[s]

	#get header and header_info
	header_info_list = []
	with open('%s/%s'%(self.dir_path, self.files[0]), 'r') as f:
	    for l in f:
		if l[0] == '#':
		    header_info_list.append(l)
		else:
		    break
	if len(header_info_list):
	    self.header_info = ''.join(header_info_list)
	    self.header = [self._re_header_remove.sub('', s) for s in \
			       header_info_list[0][1:].split()]
	else:
	    self.header_info = ''
	    self.header = []

	self._ParseFields = self._Class_ParseFields(self.header, \
							self._default_fields)

    def _load(self, index, exact_index=False, additional_fields=[]):
	p = self._get_ParseFields(additional_fields)
	fn = '%s/%s'%(self.dir_path, self.get_filename(index, exact_index))
	with open(fn, 'r') as f:
	    l = '#'
	    while l[0] == '#':
		try:
		    l = f.next()
		except StopIteration:
		    return p.pack([])
	    X = [p.parse_line(l)]
	    for l in f:
		X.append(p.parse_line(l))
	return p.pack(X)

    def _get_file_index(self, match):
	return match.group()

    def get_filename(self, index, exact_index=False):
	if exact_index:
	    i = self.file_indices.searchsorted(index)
	    if self.file_indices[i] != index:
		raise ValueError('Cannot find the exact index %s.'%(str(index)))
	else:
	    i = np.argmin(np.fabs(self.file_indices - index))
	return self.files[i]

    def _get_ParseFields(self, additional_fields):
	if not _islistlike(additional_fields) or len(additional_fields)==0:
	    return self._ParseFields
	else:
	    return self._Class_ParseFields(self.header, \
					       self._default_fields + list(additional_fields))

    _re_filename = re.compile('.+')
    _re_header_remove = re.compile('')
    _Class_ParseFields = BaseParseFields
    _default_fields = []
    load = _load


class TreesDir(BaseDirectory):
    _re_filename = re.compile('^tree_\d+_\d+_\d+.dat$')
    _re_header_remove = re.compile('\(\d+\)$')
    _default_fields = ['scale', 'id', 'pid','num_prog', 'upid', 'mvir', 'rvir', \
                                           'x', 'y', 'z', 'vmax']
    
    def load(self, tree_root_id, additional_fields=[]):
	p = self._get_ParseFields(additional_fields)
	tree_root_id_str = str(tree_root_id)
	location_file = generate_filename(self.dir_path + '/locations.dat.gz')
        
	if location_file is not None:
	    with opener(location_file, 'rb') as f:
		f.readline()
		for l in f:
		    items = l.split()
		    if items[0] == tree_root_id_str:
			break
		else:
		    raise ValueError("Cannot find this tree_root_id: %d."%(\
			    tree_root_id))
	    tree_file = generate_filename('%s/%s.gz'%(self.dir_path, items[-1]))
            if tree_file is None:
                    raise IOError('Could not find tree file {}'.format(tree_file))

	    with opener(tree_file, 'rb') as f:
		f.seek(int(items[2]))
		X = []
		for l in f:
		    if l[0] == '#': break
		    X.append(p.parse_line(l))
	else:
	    for fn in self.files:

                tree_file = generate_filename('%s/%s.gz'%(self.dir_path, fn))
                if tree_file is None:
                        raise IOError('Could not find tree file {}'.format(tree_file))

		with opener(tree_file, 'r') as f:
		    l = '#'
		    while l[0] == '#':
			try:
			    l = f.next()
			except StopIteration:
			    raise ValueError("Cannot find this tree_root_id: %d."%(\
				    tree_root_id))
		    num_trees = int(l)
		    for l in f:
			if l[0] == '#' and l.split()[-1] == tree_root_id_str:
			    break #found tree_root_id
		    else:
			continue #not in this file, check the next one
		    X = []
		    for l in f:
			if l[0] == '#': break
			X.append(p.parse_line(l))
		    break #because tree_root_id has found
	    else:
		raise ValueError("Cannot find this tree_root_id: %d."%(\
			tree_root_id))
	return p.pack(X)

    def get_src_dtype(self,additional_fields=[]):
            """
            Since the consistent tree data-type might change, we parse
            the tree_0_0_0.dat file and create a custom data-type that
            represents all the fields
            """
            p = self._get_ParseFields(additional_fields)
            return p.get_src_dtype()
    
    
class CTREESConverter(tao.Converter):
    """Subclasses tao.Converter to perform CTREES output conversion."""

    @classmethod
    def add_arguments(cls, parser):
	"""Adds extra arguments required for CTREES conversion.

	Extra arguments required for conversion are:
	  1. The location of the CTREES output trees.
	  2. The simulation box size.
	  3. The list of expansion factors (scales.txt).
	  4. The CTREES parameters file.
	"""

	parser.add_argument('--trees-dir', default='.',
			    help='location of CTREES trees')
	parser.add_argument('--a-list', help='CTREES scales.txt file')
	parser.add_argument('--parameters', help='CTREES parameter file (merger_tree.cfg)')
        parser.add_argument('--binary',help='Optional binary output in LHaloTrees format (disabled by default)')

    def get_simulation_data(self):
	"""Extract simulation data.

	Extracts the simulation data from the CTREES parameter file and
	returns a dictionary containing the values.
	"""

	if self.args.parameters:
            par = open(self.args.parameters, 'r').read()
            hubble = re.search(r'h0\s*=\s*(\d*\.?\d*)', par, re.I).group(1)
            omega_m = re.search(r'Om\s*=\s*(\d*\.?\d*)', par, re.I).group(1)
            omega_l = re.search(r'Ol\s*=\s*(\d*\.?\d*)', par, re.I).group(1)
            box_size = re.search(r'BOX_WIDTH\s*=\s*(\d*\.?\d*)', par, re.I).group(1)

        else:
            if not self.args.trees_dir:
                raise tao.ConversionError('Must specify either the CTREES config or the trees location')
                
            ## read in the first chunk bytes. (DO NOT read in entire file)
            chunk_guess = 5000
            first_tree = read_file_possibly_gzipped(self.args.trees_dir+'/tree_0_0_0.dat',chunk_guess)
            hubble = re.search(r'h0\s*=\s*(\d*\.?\d*)', first_tree, re.I).group(1)
            omega_m = re.search(r'Omega_M\s*=\s*(\d*\.?\d*)', first_tree, re.I).group(1)
            omega_l = re.search(r'Omega_L\s*=\s*(\d*\.?\d*)', first_tree, re.I).group(1)
            box_size = re.search(r'Full\s+box\s+size\s+=\s+(\d*\.?\d*)', first_tree, re.I).group(1)
			
	return {
	    'box_size': box_size,
	    'hubble': hubble,
	    'omega_m': omega_m,
	    'omega_l': omega_l,
	}

    def get_snapshot_redshifts(self):
	"""Parse and convert the expansion factors.

	Uses the expansion factors to calculate snapshot redshifts. Returns
	a list of redshifts in order of snapshots.
	"""

	if not self.args.a_list:
	    raise tao.ConversionError('Must specify a filename for the a-list')
	redshifts = []
	with open(self.args.a_list, 'r') as file:
	    for line in file:
		items = line.split()
                ## scales.txt contains snapshot number (items[0]) and scale factor (items[1])
		redshifts.append(1.0/float(items[1]) - 1.0)
	return redshifts


		
    def get_mapping_table(self):
	"""Returns a mapping from Millenium fields to CTREES fields."""

	return {
	    'posx': 'x',
	    'posy': 'y',
	    'posz': 'z',
	    'velx': 'vx',
	    'vely': 'vy',
	    'velz': 'vz',
	    'snapnum':'Snap_num',
            'Mvir':'mvir',
            'spinx':'Jx',
            'spiny':'Jy',
            'spinz':'Jz',
            'M_Mean200':'M200b',
            'M_TopHat':'M200c',
            'veldisp':'vrms',
	}

    def get_extra_fields(self):
	"""Returns a list of CTREES fields and types to include."""

	return [
	]

    
    def map_descendant(self, tree):
	"""Calculate the CTREES structure.

	The descendants are already set -> just figure out the
	indices involved. 
	"""
	descs = np.empty(len(tree), np.int32)
	descs.fill(-1)

	## desc_id is the relevant field
	ind = (np.where(tree['descid'] != -1))[0]
	for ii in xrange(len(ind)):
	    index	= ind[ii]
	    descid	= tree['descid'][index]
	    desc_scale	= tree['desc_scale'][index]
            ### The scale comparison is not required for Ctrees since ids are unique
            ### across the entire simulation. However, might as well keep it since that
            ### might change in the future.
	    desc_loc	= (np.where(tree['id'] == descid and tree['scale'] == desc_scale))[0]
	    desc[index] = desc_loc
	
	return descs


    def map_FirstProgenitor(self, tree):
        """
        FirstProgenitor crosses scale factors. 
        """

        first_prog = np.empty(len(tree), np.int32)
        first_prog.fill(-1)
        unique_descids = np.unique(tree['descid'])
        for uniq_descid in unique_descids:
            ind = (np.where(tree['descid'] == uniq_descid))[0]
            max_mass_ind = np.argmax(tree['mvir'][ind])

            desc_loc = (np.where(tree['id'] == uniq_descid))[0]
            first_prog[desc_loc] = ind[max_mass_ind]
            
        return first_prog

    def map_NextProgenitor(self,tree):
        """
        NextProgenitor might not cross scale-factors
        """
        next_prog = np.empty(len(tree), np.int32)
        next_prog.fill(-1)
        unique_descids = np.unique(tree['descid'])
        for uniq_descid in unique_descids:
            ind = (np.where(tree['descid'] == uniq_descid))[0]
            if len(ind) > 1:
                ### sort in descending order of mass
                sorted_mass_ind = (np.argsort(tree['mvir'][ind]))[::-1]

                lhs_inds = ind[sorted_mass_ind]
                rhs_inds = np.roll(lhs_inds,-1)
                
                next_prog[lhs_inds] = rhs_inds

                ## now fix the last one so that the last progenitor
                ## does not point back to the first progenitor
                next_prog[lhs_inds[-1]] = -1
                    
        return next_prog

    def map_FirstHaloInFOFgroup(self,tree):
            
        first_halo = np.empty(len(tree),np.int32)
        first_halo.fill(-1)
        host_inds = (np.where(tree['pid'] == -1))[0]
        if len(host_inds) > 0:
            ## Host halos points to themselves
            first_halo[host_inds] = host_inds
            
            ## Now set the subhalos
            hostids = tree['id'][host_inds]
            for ii,hostid in enumerate(hostids):
                ind_subs = (np.where(tree['pid'] == hostid))[0]
                if len(ind_subs) > 0:
                    first_halo[ind_subs] = host_inds[ii]
                    
            return first_halo


        return None
        
    def map_NextHaloInFOFgroup(self,tree):
        next_halo = np.empty(len(tree),np.int32)
        next_halo.fill(-1)
        host_inds = (np.where(tree['pid'] == -1))[0]
        if len(host_inds) > 0:
            hostids = tree['id'][host_inds]
            for ii,hostid in enumerate(hostids):
                ind_subs = (np.where(tree['pid'] == hostid))[0]
                if len(ind_subs) > 0:
                    ### at least the host has to be set,
                    ### and then all of the subs. However,
                    ### we will need to fix the last subhalo --> since
                    ### np.roll will attempt to point the last subhalo to the host
                        
                    ### Note that ind_subs is used directly since
                    ### those inds come directly from the tree
                    lhs_inds = np.hstack((host_inds[ii],ind_subs))
                    rhs_inds = np.roll(lhs_inds,-1)

                    ## check that assumptions are valid
                    assert np.max(lhs_inds) < len(tree),'LHS inds is wrong (can not be bigger than length of tree)'
                    assert np.max(rhs_inds) < len(tree),'RHS inds is wrong (can not be bigger than length of tree)'

                    ## now set the entire array
                    next_halo[lhs_inds] = rhs_inds

                    ## fix the last subhalo issue.
                    ## There is guaranteed to be at least
                    ## two elements in the lhs_inds array (host, + at least 1 sub)
                    next_halo[lhs_inds[-1]] = -1

            
            return next_halo

        return None

    def iterate_trees(self):
	"""
        Iterate over individual trees in CTREES.
        However, choices have to be made about flybys
        and all (sub)-subhalos have to be re-assigned to
        their host halos.
        """
        
        new_tree = TreesDir(self.args.trees_dir)
        additional_fields = ['vx','vy','vz','Jx','Jy','Jz','Snap_num','M200b','M200c','vrms','descid','desc_scale']
        src_dtype = new_tree.get_src_dtype(additional_fields)
        print src_dtype

        ### Load the forests into a dictionary.

        ### First, generate filenames that might be gzipped
        forests_filename = generate_filename(self.args.trees_dir+'/forests.list')
        if forests_filename is None:
            raise IOError('Could not find forests file {}'.format(forests_filename))


        ### create an empty forests dictionary. I find this
        ### syntax better than "forests = {}" -> much more obvious
        ### that forests is an empty python dictionary.
        ### forests will be a dictionary with keys = forest_ids
        ### and values is a list of of tree_root_ids. 
        forests = dict()
        with opener(forests_filename,'rb') as f:
            for line in f:
                ### the first line will have the header
                if line[0] == '#': continue
                tree_root_id, forest_id = map(int,line.split())

                ## if the forest_id does not exists already, then
                ## create an empty list, and then append tree_root_id
                forests.setdefault(forest_id,[]).append(tree_root_id)

        ## only loop over unique forests. 
        ## iteritems will return key, value but we 
        ## do not care about forest_ids -> hence the "_".
        for _,tree_ids in forests.iteritems():
            ntrees = len(tree_ids)
            full_tree = np.empty((1,),dtype=src_dtype)
            counts  = np.zeros(ntrees,dtype=np.int64)
            offsets = np.zeros(ntrees,dtype=np.int64)
            old_size = 0
            for itree,tree_id in enumerate(tree_ids):
                tree = new_tree.load(tree_id,additional_fields)
                
                ### Fix subs of subs
                try:
                    tree['pid'] = tree['upid']
                    ### numpy raises ValueError when key is absent !!!
                    ### Catch both, in case numpy changes behaviour in the future
                except (KeyError,ValueError) as e:
                        pass

                ## terrible hack to allow ascending order
                tree['scale'] *= -1.0
                tree['mvir'] *= -1.0
                
                ## things are already somewhat ordered -> so quicksort might 
                ## show worst-case scaling of O(N^2). Use either heapsort or mergersort
                tree.sort(order=('scale','descid','mvir'),kind='mergesort')
                
                ## now restore the original values
                tree['scale'] *= -1.0
                tree['mvir'] *= -1.0
			
                ## change angular momentum to LHaloTree convention
                inv_mass = 1.0/tree['mvir']
                for field in ['Jx','Jy','Jz']:
                    ### avoid the divide for 3 fields and multiply with reciprocal instead
                    ### However, the extra memory can be avoided, if needed (at the expense of speed)
                    tree[field] *= inv_mass

                ## convert mass units to 10^10 Msun/h
                for field in ['mvir','M200b','M200c']:
                    tree[field] *= 1e-10
			

                counts[itree]  = len(tree)
                offsets[itree] = old_size
                full_tree.resize(old_size + len(tree))

                ## python uses 0:N syntax for assign N variables (unlike C where it would 0:N-1)
                full_tree[old_size:old_size + len(tree)] = tree


                
            ### Still need to fix flybys but this can only be done
            ### after the entire forest has been loaded. 
                
            yield full_tree
                
                
                                                                                          

                                                                        
