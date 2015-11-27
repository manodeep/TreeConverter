from collections import OrderedDict
from Module import Module
from .validators import *
from .generators import *
from .library import library

class MergerTree(Module):
    fields = OrderedDict([
            ('Descendant', {
                    'description':'Index of Descendant halo',
                    'type':np.int32,
                    }
             ),
            ('FirstProgenitor',{
                    'description':'Index for most massive progenitor halo',
                    'type':np.int32,
                    }
             ),
            ('NextProgenitor',{
                    'description': 'Index for next progenitor halo that points to the same descendant halo',
                    'type':np.int32,
                    }
             ),
            ('FirstHaloInFOFgroup',{
                    'description':'Index for the main halo within the FOF group',
                    'type':np.int32,
                    }
             ),
            ('NextHaloInFOFgroup',{
                    'description':'Index for the next halo within the FOF group',
                    'type':np.int32,
                    }
             ),
            ('Len',{
                    'description':'Number of particles in the halo',
                    'type':np.int32,
                    }
             ),
            ('M_Mean200',{
                    'description':'Mass of the halo (M_200b)',
                    'units':'Msun/h',
                    'type':np.float32,
                    }
             ),
            ('Mvir',{
                    'description':'Virial mass of the halo',
                    'units':'Msun/h',
                    'type':np.float32,
                    }
             ),
            ('M_TopHat',{
                    'description':'Halo mass',
                    'units':'Msun/h',
                    'type':np.float32,
                    }
             ),
            ('posx',{
                    'description':'X-position of the halo center',
                    'units':'Mpc/h',
                    'type':np.float32,
                    }
             ),
            ('posy',{
                    'description':'Y-position of the halo center',
                    'units':'Mpc/h',
                    'type':np.float32,
                    }
             ),
            ('posz',{
                    'description':'Z-position of the halo center',
                    'units':'Mpc/h',
                    'type':np.float32,
                    }
             ),
            ('velx',{
                    'description':'X-component of the halo velocity',
                    'units':'km/s',
                    'type':np.float32,
                    }
             ),
            ('vely',{
                    'description':'Y-component of the halo velocity',
                    'units':'km/s',
                    'type':np.float32,
                    }
             ),
            ('velz',{
                    'description':'Z-component of the halo velocity',
                    'units':'km/s',
                    'type':np.float32,
                    }
             ),
            ('veldisp',{
                    'description':'Velocity dispersion of the halo',
                    'units':'km/s',
                    'type':np.float32,
                    }
             ),
            ('vmax',{
                    'description':'Max. of the halo circular Velocity',
                    'units':'km/s',
                    'type':np.float32,
                    }
             ),
            ('spinx',{
                    'description':'X-component of the halo spin',
                    'type':np.float32,
                    }
             ),
            ('spiny',{
                    'description':'Y-component of the halo spin',
                    'type':np.float32,
                    }
             ),
            ('spinz',{
                    'description':'Z-component of the halo spin',
                    'type':np.float32,
                    }
             ),
            ('MostBoundID',{
                    'description':'Particle id for the most bound particle',
                    'type':np.int64,
                    }
             ),
            ('snapnum',{
                    'description':'Snapshot number of the halo',
                    'type':np.int32,
            }),
            ('FileNr',{
                    'description':'File number from the Peano-Hilbert key (Millenium specific)',
                    'type':np.int32,
                    }
             ),
            ('SubhaloIndex',{
                    'description':'Subhalo index',
                    'type':np.int32,
                    }
             ),
            ('SubHalfMass',{
                    'description':'Half-mass of the subhalo',
                    'type':np.float32,
                    }
             )
            ])

    generators = [
        GlobalIndices(),
    ]
    validators = [
        Required('posx', 'posy', 'posz',
                 'velx', 'vely', 'velz',
                 'spinx','spiny','spinz',
                 'snapnum','Mvir','veldisp'),
        OverLittleH('posx', 'posy', 'posz'),
        WithinRange(0.0, library['box_size'], 'posx', 'posy', 'posz'),
        WithinCRange(0, library['n_snapshots'], 'snapnum'),
        NonNegative('veldisp'),
#         NoNegative('SubhaloIndex','SubHalfMass','MostBoundID'),
    ]                            
