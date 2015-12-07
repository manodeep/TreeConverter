"""Convert CTREES output to Millenium.

A control script to be used with `taoconvert` to convert CTREES output binary data
into HDF5 input for TAO.
"""

import re, os, time
import numpy as np
import pandas as pd
import tempfile
import errno
import tao

# from gzopen import gzopen
import gzip
import math 
from IPython.core.debugger import Tracer
import struct
import progressbar

def isWritable(path):
    try:
        testfile = tempfile.TemporaryFile(dir = path)
        testfile.close()
    except OSError as e:
        if e.errno == errno.EACCES:  # 13
            return False
        e.filename = path
        raise
    return True


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

    ## File not found
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
    _default_fields = ['scale', 'id', 'pid','num_prog', 'mvir', 'rvir', \
                                           'x', 'y', 'z', 'vmax']
    # @profile
    def load(self, tree_root_id, additional_fields=[],locations_dict=None):
	p = self._get_ParseFields(additional_fields)
	tree_root_id_str = str(tree_root_id)
	location_file = generate_filename(self.dir_path + '/locations.dat')
        
	if location_file is not None or locations_dict is not None:

            ### Have the locations dictionary
            if locations_dict is not None:
                offset,tree_file = locations_dict[tree_root_id]
                tree_file = generate_filename('%s/%s.gz'%(self.dir_path, tree_file))
            else:
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
                offset = int(items[2])
                
            if tree_file is None:
                raise IOError('Could not find tree file for tree root id = {}'.format(tree_root_id))

	    with opener(tree_file, 'rb') as f:

                ### Note that f might be GzipFile
                ### However, offsets are defined wrt uncompressed
                ### sizes -> following line will work as expected
		f.seek(offset)
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
    
    
class CTREESConverter():
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
        # parser.add_argument('--binary',help='Optional binary output in LHaloTrees format (disabled by default)')

    def get_simulation_data(self,trees_dir=None):
	"""Extract simulation data.

	Extracts the simulation data from the CTREES parameter file and
	returns a dictionary containing the values.
	"""

        if trees_dir is None:
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

        else:
            
            ## read in the first chunk bytes. (DO NOT read in entire file)
            chunk_guess = 5000
            first_tree = read_file_possibly_gzipped(trees_dir+'/tree_0_0_0.dat',chunk_guess)
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

    def get_snapshot_redshifts(self,scales_file=None):
	"""Parse and convert the expansion factors.

	Uses the expansion factors to calculate snapshot redshifts. Returns
	a list of redshifts in order of snapshots.
	"""

        
        if scales_file is None:
            if not self.args.a_list:
                raise tao.ConversionError('Must specify a filename for the a-list')
            else:
                scales_file = self.args.a_list
                
        redshifts = []
	with open(scales_file, 'r') as file:
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
            'vmax':'vmax',
	}

    def get_extra_fields(self):
	"""Returns a list of CTREES fields and types to include."""

	return [
	]

    def map_Len(self,tree):
        part_mass = 1.28235e9
        inv_part_mass = 1.0/part_mass
        length = np.array(np.rint(tree['mvir']*inv_part_mass), dtype=np.int32)
        return length
        

    def map_descendant(self,tree):
	"""Calculate the CTREES structure.

	The descendants are already set -> just figure out the
	indices involved. 
	"""
	descs = np.empty(len(tree), np.int32)
	descs.fill(-1)

	## desc_id is the relevant field
	ind = (np.where(tree['descid'] != -1))[0]
	for index in ind:
	    descid	= tree['descid'][index]
	    desc_scale	= tree['desc_scale'][index]
            ### The scale comparison is not required for Ctrees since ids are unique
            ### across the entire simulation. However, might as well keep it since that
            ### might change in the future.
	    desc_loc	= (np.where(tree['id'] == descid))[0]
            assert len(desc_loc) == 1,'There should be exactly one descendant'
            assert tree['scale'][desc_loc] == desc_scale,'ID is assumed to be unique! '
	    descs[index] = desc_loc
	
	return descs

    # @profile
    def map_FirstProgenitor(self,tree):
        """
        FirstProgenitor crosses scale factors. 
        """

        first_prog = np.empty(len(tree), np.int32)
        first_prog.fill(-1)
        unique_descids  = np.unique(tree['descid'])
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

    def lookup_fof_halo(self, pid, tree):
        orig_pid = pid
        loc = (np.where(tree['id'] == pid))[0]
        while tree['pid'][loc] != -1:
            pid = tree['pid'][loc]
            loc = (np.where(tree['id'] == pid))[0]

        return loc

    def map_FirstHaloInFOFgroup(self,tree):
            
        first_halo = np.empty(len(tree),np.int32)
        first_halo.fill(-1)
        host_inds = (np.where(tree['pid'] == -1))[0]
        if len(host_inds) > 0:
            ## Host halos points to themselves
            first_halo[host_inds] = host_inds
            
            ## Now set the subhalos
            hostids = tree['id'][host_inds]
            for host_loc,hostid in zip(host_inds,hostids):
                ind_subs = (np.where(tree['pid'] == hostid))[0]
                if len(ind_subs) > 0:
                    first_halo[ind_subs] = host_loc
            
                
            if min(first_halo) < 0:
                Tracer()()

            return first_halo
        print("tree = {}".format(tree))
        print("returning none. expect code to break")
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

    def load_forests(self,trees_dir='./'):
        
        ### First, generate filenames that might be gzipped
        forests_filename = generate_filename(trees_dir+'/forests.list')
        if forests_filename is None:
            raise IOError('Could not find forests file {}'.format(trees_dir+'/forests.list'))


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


        return forests

    
    def get_forest_sizes(self,trees_dir='./',forests=None,locations=None):

        if forests is None:
            forests = load_forests(trees_dir)

        if locations is None:
            locations = load_locations(trees_dir)

        tree_sizes = dict()

        ## This is an embarrassingly parallel loop
        for tree_root_id in locations.keys:
            offset,filename = locations[tree_root_id]
            tree_filename = generate_filename(trees_dir+'/'+filename)
            if tree_filename is None:
                raise IOError('Could not find tree file'.format(trees_dir+'/'+filename))

            with opener(tree_filename,'rb') as f:
                f.seek(offset)
                ## skip the #tree line
                f.next()

                count=0L
                ## now loop until the next #tree
                for line in f:
                    if '#tree' in line:
                        break
                    count+=1
                    
                tree_sizes[tree_root_id] = count

        forest_sizes = dict()
        for forest_id in forests:
            tree_ids = forests[forest_id]
            size = sum(tree_sizes[tree_id] for tree_id in tree_ids)
            forest_sizes[forest_id] = size

        return forest_sizes,tree_sizes

    def get_halotypes_for_tree_root(self,trees_dir='./',forests=None,locations=None):
        if forests is None:
            forests = load_forests(trees_dir)

        if locations is None:
            locations = self.load_locations(trees_dir)

        treeparser = TreesDir(trees_dir)
	p = treeparser._get_ParseFields([])

        tree_types  = dict()
        ## This is an embarrassingly parallel loop
        for tree_root_id,(offset,filename) in locations.items():
            tree_filename = generate_filename(trees_dir+'/'+filename)
            if tree_filename is None:
                raise IOError('Could not find tree file'.format(trees_dir+'/'+filename))


            with opener(tree_filename,'rb') as f:
                f.seek(offset)
                ## skip the #tree line
                f.next()

                ## now read the first output for this tree root
                line = f.next()
                try:
                    X = p.pack(p.parse_line(line))
                    tree_types[tree_root_id] = 1 if X['pid'] == -1 else 0
                except:
                    Tracer()()


        nforests = len(forests)
        isolated_forest = []
        for _,tree_ids in forests.items():
            try:
                num_fofs_max_scale = sum(tree_types[tree_id] for tree_id in tree_ids)
                isolated_forest.append(False if (num_fofs_max_scale > 1) else True)
            except:
                Tracer()()

        return tree_types,np.array(isolated_forest,dtype=bool)

    
    def load_locations(self,trees_dir='./'):
        locations_filename = generate_filename(trees_dir+'/locations.dat')
        if locations_filename is None:
            raise IOError('Could not find locations file'.format(trees_dir+'/locations.dat'))
                         
        locations = dict()
        with opener(locations_filename,'rb') as f:
            for line in f:
                if line[0] == '#': continue

                try:
                    items = line.split()
                    tree_root_id = int(items[0])
                    fileid       = int(items[1])
                    offset       = int(items[2])
                    filename = items[3]
                except:
                    print("Parse error: items = {}".format(items))
                    raise
                    
                locations.setdefault(tree_root_id,[]).append(offset)
                locations[tree_root_id].append(filename)

        return locations

    def get_MainBranch(iterable, get_num_prog):
        item = iter(iterable)
        q = deque([(item.next(), True)])
        X = []
        while len(q):
            i, i_mb = q.popleft()
            X.append(i_mb)
            n = get_num_prog(i)
            prog_mb = [i_mb] + [False]*(n-1) if n else []
            q.extend([(item.next(), mb) for mb in prog_mb])

        return np.array(X)
                    

    # @profile
    def iterate_trees(self,trees_dir='./',forests=None,locations=None):
	"""
        Iterate over individual trees in CTREES.
        However, choices have to be made about flybys
        and all (sub)-subhalos have to be re-assigned to
        their host halos.
        """
        
        new_tree = TreesDir(trees_dir)
        additional_fields = ['vx','vy','vz','Jx','Jy','Jz','Snap_num','M200b','M200c','vrms','descid','desc_scale']
        src_dtype = new_tree.get_src_dtype(additional_fields)
        print src_dtype

        if forests is None:
            ### Load the forests into a dictionary.
            forests = self.load_forests(trees_dir)

        if locations is None:
            locations = self.load_locations(trees_dir)
            
        ## only loop over unique forests. 
        ## items() will return key, value but we 
        ## do not care about forest_ids -> hence the "_".
        for _,tree_ids in forests.items():
            ntrees = len(tree_ids)
            full_tree = np.empty((1,),dtype=src_dtype)
            counts  = np.zeros(ntrees,dtype=np.int64)
            offsets = np.zeros(ntrees,dtype=np.int64)
            curr_size = 0

            # print("Forest = {} trees = {}".format(_,tree_ids))
            
            for itree,tree_id in enumerate(tree_ids):
                tree = new_tree.load(tree_id,additional_fields,locations)

                ## terrible hack to allow ascending order
                tree['scale'] *= -1.0
                tree['mvir'] *= -1.0
                
                ## things are already somewhat ordered -> so quicksort might 
                ## show worst-case scaling of O(N^2). Use either heapsort or mergersort
                tree.sort(order=('scale','id','descid'),kind='mergesort')
                
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
                offsets[itree] = curr_size
                full_tree.resize(curr_size + len(tree))

                # print("loaded tree = {} curr_size = {} nhalos = {}".format(tree_id,curr_size,len(tree)))

                
                ## python uses 0:N syntax for assign N variables (unlike C where it would 0:N-1)
                full_tree[curr_size:curr_size + len(tree)] = tree
                curr_size += len(tree)

            ### Fix subs of subs
            for ii,pid in enumerate(full_tree['pid']):
                if pid == -1: continue
                
                hostpid = pid
                fof_loc = self.lookup_fof_halo(hostpid,full_tree)
                full_tree['pid'][ii] = full_tree['id'][fof_loc]

            ### All the pre-processing has been done -> return the tree (really, all trees in the forest)
            yield full_tree

def construct_filename_without_duplicate_slash(dirname,filename):
    if dirname.endswith('/'):
        return dirname + filename
    else:
        return dirname + '/' + filename

            
# @profile
def LHaloTreeWriter(trees_dir,output_dir,comm=None):

    if comm is None:
        rank = 0

        
    cls = CTREESConverter()
    mapping_table = cls.get_mapping_table()

    forests = cls.load_forests(trees_dir)
    num_forests = len(forests)

    locations = cls.load_locations(trees_dir)
    
    ## Find out how many trees do we have that do not have flybys
    ## and reset num_forests to that value. WARNING: NEEDS TO BE FIXED
    ## after converter is verified
    
    t0 = time.time()
    isolated_file = construct_filename_without_duplicate_slash(output_dir,'lhalotree_isolated_forests.npy')
    if not os.path.isfile(isolated_file):
        _,isolated_forests = cls.get_halotypes_for_tree_root(trees_dir,forests,locations)
        np.save(isolated_file,isolated_forests)
    else:
        isolated_forests = np.load(isolated_file)
        assert len(isolated_forests) == num_forests, "Isolated forests has length = %r compared to number of forests = %r " %(len(isolated_forests),num_forests)
    print("Time taken to get isolated forests count = {:12.4f}".format(time.time()-t0))

    ### Resetting num_forests
    num_isolated_forests = len((np.where(isolated_forests == True))[0])
    assert num_isolated_forests <= num_forests,'Number of isolated forests can be at most the total number of forests %r,%r' % (num_isolated_forests,num_forests)
    output_file = construct_filename_without_duplicate_slash(output_dir,'lhalotree_test.bin')
    output_file = output_file + '.' + str(rank)

    if rank == 0:
        print("Writing {} trees to {}".format(num_forests,output_file))
        bar = progressbar.ProgressBar(max_value=num_forests)

    print("Resetting number of forests from {} to isolated forests = {}".format(num_forests,num_isolated_forests))
    ### reset num_forests
    num_forests = num_isolated_forests

    max_num_elements_buffer = 100000
    output_buffer = np.empty(max_num_elements_buffer,dtype=output_dtype)
    curr_size_output_buffer = 0
    
    with open(output_file,'wb') as f:
        totntrees = struct.pack('<i4',0)
        totnhalos = struct.pack('<i4',0)

        ## write the place holders for now
        f.write(totntrees)
        f.write(totnhalos)

        ## write place-holders for treeNhalos
        treeNhalos = struct.pack('%si' % num_forests, *([0]*num_forests))
        f.write(treeNhalos)

        treeNhalos = []
        totnhalos  = 0
        totntrees  = 0

        tree_generator = cls.iterate_trees(trees_dir,forests)
        numflybys = []
        for ii,src_tree in enumerate(tree_generator):
            if isolated_forests[ii] == False:
                continue

            Nhalos = len(src_tree)
            treeNhalos.append(Nhalos)
            totnhalos += Nhalos
            totntrees += 1
            
            src_dtype = src_tree.dtype            
            dst_tree = np.empty_like(src_tree,dtype=output_dtype)

            ## copy fields
            for name in dst_tree.dtype.names:
                if name in src_dtype.names:
                    dst_tree[name] = src_tree[name]

                try:
                    src_field_name = mapping_table[name]
                    dst_tree[name] = src_tree[src_field_name]
                    
                except KeyError:
                    pass

            ## Now, create all of the first/last/fof indices in dst_tree
            dst_tree['Descendant']          = cls.map_descendant(src_tree)
            dst_tree['FirstProgenitor']     = cls.map_FirstProgenitor(src_tree)
            dst_tree['NextProgenitor']      = cls.map_NextProgenitor(src_tree)
            dst_tree['FirstHaloInFOFgroup'] = cls.map_FirstHaloInFOFgroup(src_tree)
            dst_tree['NextHaloInFOFgroup']  = cls.map_NextHaloInFOFgroup(src_tree)
            dst_tree['Len']                 = cls.map_Len(src_tree)


            ### Still need to fix flybys but this can only be done
            ### after the entire forest has been loaded. The only
            ### problem happens when there are multiple roots (FOFs) at z=0
            max_scale = np.max(src_tree['scale'])

            ## ind gives the number of FOF groups at z=0
            ind = (np.where((src_tree['scale'] == max_scale) & (src_tree['pid'] == -1)))[0]
            numflybys.append(len(ind))
            if len(ind) > 1:

                for ifof,ifof_loc in enumerate(ind):

                    ## if this is the last FOF, then nothing
                    ## remains to be done
                    if ifof == len(ind)-1: break

                    last_sub = ifof_loc
                    while last_sub != -1:
                        last_sub = dst_tree['NextHaloInFOFgroup'][last_sub]

                    ## Point the NextHaloInFOFgroup to the next
                    ## FOF halo. Note that we shouldn't reach this stage

                    assert ifof+1 < len(ind), 'Seeking past FOF indices array'
                    dst_tree['NextHaloInFOFgroup'][last_sub] = ind[ifof+1]

            assert Nhalos == len(dst_tree),'len(dst_tree) must be the same as Nhalos'
            #### Instead of writing out directly, add to the output buffer
            #### and then intermittently write out the output buffer
            if (curr_size_output_buffer + Nhalos) >=  max_num_elements_buffer:
                output_buffer[0:curr_size_output_buffer].tofile(f)
                curr_size_output_buffer = 0

            ### Is the output buffer big enough to hold this tree?
            ### Resize if that is not the case.
            if Nhalos > max_num_elements_buffer:
                output_buffer.resize(Nhalos)
                max_num_elements_buffer = Nhalos

            assert curr_size_output_buffer + Nhalos <= max_num_elements_buffer,"Array overflow will happen in output buffer"
            output_buffer[curr_size_output_buffer:curr_size_output_buffer+Nhalos] = dst_tree.copy()
            curr_size_output_buffer += Nhalos
            
            if rank==0:
                bar.update(ii)


        ## Is there any data in the output_buffer -> write that to disk
        if curr_size_output_buffer > 0:
            output_buffer[0:curr_size_output_buffer].tofile(f)
            curr_size_output_buffer = 0
            
        assert curr_size_output_buffer == 0, 'Must have written all bytes in output buffer'
        print("len(treeNhalos) = {} num_forests = {}".format(len(treeNhalos),num_forests))
        assert len(treeNhalos) == num_forests, "TreeNhalos has %r forests instead of %r " % (len(treeNhalos),num_forests)
        treeNhalos = np.array(treeNhalos)
        
        ## rewind to the beginning of the output file
        f.seek(0)
        f.write(struct.pack('<i4',totntrees))
        f.write(struct.pack('<i4',totnhalos))
        f.write(memoryview(treeNhalos).tobytes())

    if rank == 0:
        ## this would be the spot to sum up all the numflybys arrays over mpi
        ##
        ##Tracer()()
        ## np.histogram(numflybys)
        scales_file = construct_filename_without_duplicate_slash(trees_dir,'../outputs/scales.txt')
        scales = []
        with open(scales_file,'r') as f:
            for line in f:
                items = line.split()
                a = float(items[1])
                scales.append(1)

        ##redshifts = cls.get_snapshot_redshifts(scales_file)
        scales = np.array(scales)
        a_listfile = construct_filename_without_duplicate_slash(output_dir,'a_list.txt')
        with open(a_listfile,'w') as f:
            f.write(scales)
    

if __name__ == '__main__':

    output_dtype = np.dtype([
            ('Descendant',         np.int32),
            ('FirstProgenitor',    np.int32),
            ('NextProgenitor' ,    np.int32),
            ('FirstHaloInFOFgroup',np.int32),
            ('NextHaloInFOFgroup', np.int32),
            ('Len',                np.int32),
            ('M_Mean200',          np.float32),
            ('Mvir',               np.float32),
            ('M_TopHat',           np.float32),
            ('posx',               np.float32),
            ('posy',               np.float32),
            ('posz',               np.float32),
            ('velx',               np.float32),
            ('vely',               np.float32),
            ('velz',               np.float32),
            ('veldisp',            np.float32),
            ('vmax',               np.float32),
            ('spinx',              np.float32),
            ('spiny',              np.float32),
            ('spinz',              np.float32),
            ('MostBoundID',        np.int64),
            ('snapnum',            np.int32),
            ('FileNr',             np.int32),
            ('SubhaloIndex',       np.int32),
            ('SubHalfMass',        np.float32),
            ],align=True)
    assert output_dtype.itemsize == 104,'Output datatype must be exactly equal to 104 bytes. Otherwise, C codes will read-in garbage'
    import argparse
    parser = argparse.ArgumentParser(description='Convert Consistent-Trees output to LHaloTrees binary format')
    parser.add_argument('--trees-dir','-t',help="the root directory containing the tree_*_*_*.dat files")
    parser.add_argument('--output-dir','-o',help="the output directory")
    args = parser.parse_args()

    if args.trees_dir:
        trees_dir = args.trees_dir
    else:
        trees_dir = "./"

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = trees_dir

    ## Check that the tree_0_0_0.dat file actually exists
    tree_file = generate_filename(trees_dir + '/tree_0_0_0.dat')
    if tree_file is None:
        print("\n\nERROR: Could not locate the tree_0_0_0.dat file in {}".format(trees_dir))
        print("exiting...")
        exit()

    if not isWritable(output_dir):
        print("\n\nERROR: output directory = {} is not writable".format(output_dir))
        print("exiting...")
        exit()
    
    print("\ntrees dir = {}\noutput dir = {}".format(trees_dir,output_dir))
    t0 = time.time()

    ## last argument is place-holder for rank for upcoming MPI capabilities
    LHaloTreeWriter(trees_dir,output_dir,None)
    print("\nDone converting..time taken = {}\n".format(time.time()-t0))

        
    
                                                                                          

                                                                        
