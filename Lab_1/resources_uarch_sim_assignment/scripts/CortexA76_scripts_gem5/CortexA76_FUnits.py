from m5.SimObject import SimObject
from m5.defines import buildEnv
from m5.params import *

from m5.objects.FuncUnit import *

class BranchUnit(FUDesc):
    opList = [ OpDesc(opClass='FloatCmp', opLat=2, pipelined=True),
               OpDesc(opClass='SimdCmp', opLat=2, pipelined=True),
               OpDesc(opClass='SimdFloatCmp', opLat=2, pipelined=True),
               OpDesc(opClass='SimdReduceCmp', opLat=2, pipelined=True),
               OpDesc(opClass='SimdFloatReduceCmp', opLat=2, pipelined=True),
               OpDesc(opClass='SimdPredAlu'),
               OpDesc(opClass='InstPrefetch')]
    count = 1

class IntALU(FUDesc):
    opList = [ OpDesc(opClass='IntAlu', opLat=1)]
    count = 2

class IntMACDiv(FUDesc):
    opList = [ OpDesc(opClass='IntAlu', opLat=1),
               OpDesc(opClass='IntMult', opLat=1),
               OpDesc(opClass='IntDiv', opLat=2, pipelined=True) ]
    count=1

class FP_SIMD_Unit(FUDesc):
    opList = [ OpDesc(opClass='FloatAdd', opLat=2, pipelined=True),
               OpDesc(opClass='FloatCvt', opLat=2, pipelined=True),
               OpDesc(opClass='FloatMult', opLat=2, pipelined=True),
               OpDesc(opClass='FloatMultAcc', opLat=3, pipelined=True),
               OpDesc(opClass='FloatMisc', opLat=2, pipelined=True),
               OpDesc(opClass='FloatDiv', opLat=4, pipelined=True),
               OpDesc(opClass='FloatSqrt', opLat=5, pipelined=True),
               OpDesc(opClass='SimdAdd'),
               OpDesc(opClass='SimdAddAcc'),
               OpDesc(opClass='SimdAlu'),
               OpDesc(opClass='SimdCvt'),
               OpDesc(opClass='SimdMisc'),
               OpDesc(opClass='SimdMult'),
               OpDesc(opClass='SimdMultAcc'),
               OpDesc(opClass='SimdShift'),
               OpDesc(opClass='SimdShiftAcc'),
               OpDesc(opClass='SimdDiv', opLat=4, pipelined=True),
               OpDesc(opClass='SimdSqrt', opLat=5, pipelined=True),
               OpDesc(opClass='SimdFloatAdd', opLat=2, pipelined=True),
               OpDesc(opClass='SimdFloatAlu', opLat=2, pipelined=True),
               OpDesc(opClass='SimdFloatCvt', opLat=2, pipelined=True),
               OpDesc(opClass='SimdFloatDiv', opLat=4, pipelined=True),
               OpDesc(opClass='SimdFloatMisc', opLat=2, pipelined=True),
               OpDesc(opClass='SimdFloatMult', opLat=2, pipelined=True),
               OpDesc(opClass='SimdFloatMultAcc', opLat=3, pipelined=True),
               OpDesc(opClass='SimdFloatSqrt', opLat=5, pipelined=True),
               OpDesc(opClass='SimdReduceAdd', opLat=8, pipelined=True),
               OpDesc(opClass='SimdReduceAlu', opLat=8, pipelined=True),
               OpDesc(opClass='SimdFloatReduceAdd', opLat=8, pipelined=True),
               OpDesc(opClass='SimdAes', opLat=20, pipelined=False),
               OpDesc(opClass='SimdAesMix', opLat=20, pipelined=False),
               OpDesc(opClass='SimdSha1Hash', opLat=20, pipelined=False),
               OpDesc(opClass='SimdSha1Hash2', opLat=20, pipelined=False),
               OpDesc(opClass='SimdSha256Hash', opLat=20, pipelined=False),
               OpDesc(opClass='SimdSha256Hash2', opLat=20, pipelined=False),
               OpDesc(opClass='SimdShaSigma2', opLat=20, pipelined=False),
               OpDesc(opClass='SimdShaSigma3', opLat=20, pipelined=False)]
    count = 2

class ReadPort(FUDesc):
    opList = [ OpDesc(opClass='MemRead'),
               OpDesc(opClass='FloatMemRead') ]
    count = 2

class WritePort(FUDesc):
    opList = [ OpDesc(opClass='MemRead'), 
              OpDesc(opClass='MemWrite')]
    count = 1

class IprPort(FUDesc):
    opList = [ OpDesc(opClass='IprAccess', opLat = 3, pipelined = False) ]
    count = 1


