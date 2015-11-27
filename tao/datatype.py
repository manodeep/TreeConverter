import numpy as np

# datatype = np.dtype([
#     ('snapshot', np.int32),
#     ('type',     np.int32),
#     ('position_x', np.float32), ('position_y', np.float32), ('position_z', np.float32),
#     ('velocity_x', np.float32), ('velocity_y', np.float32), ('velocity_z', np.float32),
# ])

datatype = np.dtype([
        ('Descendant',np.int32,1),
        ('FirstProgenitor',np.int32,1),
        ('NextProgenitor',np.int32,1),
        ('FirstHaloInFOFgroup',np.int32,1),
        ('NextHaloInFOFgroup',np.int32,1),

        ('Len',np.int32,1),
        ('M_Mean200',np.float32,1),
        ('Mvir',np.float32,1),
        ('M_TopHat',np.float32,1),

        ('Pos',np.float32,3),
        ('Vel',np.float32,3),
        ('VelDisp',np.float32,1),
        ('Vmax',np.float32,1),
        ('Spin',np.float32,3),
        
        ('MostBoundID',np.int64,1),


        ('SnapNum',np.int32,1),
        ('FileNr',np.int32,1),
        ('SubhaloIndex',np.int32,1),
        ('SubHalfMass',np.float32,1)
        
        ])
