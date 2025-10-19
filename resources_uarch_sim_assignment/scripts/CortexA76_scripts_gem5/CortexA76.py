# Import m5 library and all SimObjects
import m5
import os

from m5.objects import *
from CortexA76_caches import *
from CortexA76_FUnits import *
from CortexA76_FUPool import *

from optparse import OptionParser


parser = OptionParser()
# Workload options
parser.add_option('-c','--cmd',action="store", type="string",
                      default="",help="Workload to simulate in SE mode")
parser.add_option('-d','--cwd',action="store", type="string",
                      default=os.getcwd(),help="Working directory")
parser.add_option('-o','--options',action="store", type="string",
                      default="",help="Workload's command line options")

# Memory Options
parser.add_option('--l1i_size', type="string", default="64kB", help="L1 instruction cache size")
parser.add_option('--l1i_assoc', type="int", default=4, help="L1 instruction cache associativity")
parser.add_option('--l1i_lat', type="int", default=2, help="L1 instruction cache latency")
#parser.add_option('--itb_size', type="int", default=48, help="Instruction TLB size")
parser.add_option('--l1d_size', type="string", default="64kB", help="L1 data cache size")
parser.add_option('--l1d_assoc', type="int", default=4, help="L1 data cache associativity")
parser.add_option('--l1d_lat', type="int", default=4, help="L1 data cache latency")
#parser.add_option('--dtb_size', type="int", default=48, help="Data TLB size")
parser.add_option('--l2_size', type="string", default="256kB", help="Unified L2 cache size")
parser.add_option('--l2_assoc', type="int", default=8, help="Unified L2 cache associativity")
parser.add_option('--l2_lat', type="int", default=9, help="Unified L2 cache latency")
parser.add_option('--l3_size', type="string", default="2MB", help="Shared L3 cache size")
parser.add_option('--l3_assoc', type="int", default=16, help="Shared L3 cache associativity")
parser.add_option('--l3_lat', type="int", default=30, help="Shared L3 cache latency")

# CPU Options
parser.add_option('--fetch_width', type="int", default=4, help="CPU fetch width")
parser.add_option('--decode_width', type="int", default=4, help="CPU decode width")
parser.add_option('--rename_width', type="int", default=4, help="CPU rename width")
parser.add_option('--commit_width', type="int", default=4, help="CPU commith width")
parser.add_option('--dispatch_width', type="int", default=8, help="CPU dispatch width")
parser.add_option('--issue_width', type="int", default=8, help="CPU issue width")
parser.add_option('--wb_width', type="int", default=4, help="CPU write back width")

parser.add_option('--fb_entries', type="int", default=16, help="Number of fetch buffer entries")
parser.add_option('--fq_entries', type="int", default=16, help="Number of fetch queue entries")
parser.add_option('--iq_entries', type="int", default=16, help="Number of instruction queue entries")
parser.add_option('--rob_entries', type="int", default=128, help="Number of reorder buffer entries")
parser.add_option('--lq_entries', type="int", default=68, help="Number of load queue entries")
parser.add_option('--sq_entries', type="int", default=72, help="Number of store queue entries")
parser.add_option('--btb_entries', type="int", default=4096, help="Number of BTB entries")
parser.add_option('--ras_entries', type="int", default=16, help="Number of RAS entries")


parser.add_option('--num_fu_cmp', type="int", default=1, help="Number of execution units for compare/branch instructions")
parser.add_option('--num_fu_intALU', type="int", default=2, help="Number of execution units for integer ALU instructions")
parser.add_option('--num_fu_intDIVMUL', type="int", default=1, help="Number of execution units for integer Division and Multiplication instructions")
parser.add_option('--num_fu_FP_SIMD_ALU', type="int", default=2, help="Number of execution units for Floating-Point and SIMD instructions")
parser.add_option('--num_fu_read', type="int", default=2, help="Number of execution units for load instructions")
parser.add_option('--num_fu_write', type="int", default=1, help="Number of execution units for store instructions")

parser.add_option('--branch_predictor_type', type="int", default=10, help="Branch predictor type: 0 - BiModeBP, 1 - LTAGE, 2 - LocalBP, 3 - MultiperspectivePerceptron64KB, 4 - MultiperspectivePerceptron8KB, 5 - MultiperspectivePerceptronTAGE64KB, 6 - MultiperspectivePerceptronTAGE8KB, 7 - TAGE, 8 - TAGE_SC_L_64KB, 9 - TAGE_SC_L_8KB, 10 - TournamentBP")

(options, args) = parser.parse_args()

# Crate the system object. It will be the parent of all the other objects in our simulated system. 
# The System object contains a lot of functional (not timing-level) information, like the physical 
# memory ranges, the root clock domain, the root voltage domain, the kernel (in full-system simulation), etc
system = System()

# create a clock domain and set the clock frequency on that domain. 
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '2.1GHz'

# Create voltage domain for this clock domain.

system.clk_domain.voltage_domain = VoltageDomain()


# Set up how the memory will be simulated. We are going to use timing mode for the memory simulation. 
# You will almost always use timing mode for the memory simulation, except in special cases like fast-forwarding and restoring from a checkpoint. 

system.mem_mode = 'timing'

# Setup the memory range, CortexA76 processor has a capacity of 40-bit for addressing memory, this is 1TB
system.mem_ranges = [AddrRange('8GB')]

# The line size in Bytes for all cache memories 
system.cache_line_size = 64

# Create and setup a Out-of-Order CPU. 

system.cpu = DerivO3CPU()
# Connect the caches
system.cpu.icache = L1ICache(options)
system.cpu.dcache = L1DCache(options)
#system.cpu.dtb.size = options.dtb_size
#system.cpu.itb.size = options.itb_size

# Setup CPU features
system.cpu.cpu_id = 0
system.cpu.socket_id = 1
system.cpu.cacheLoadPorts = options.num_fu_read
system.cpu.cacheStorePorts = options.num_fu_write
# Setup branch predictor
if options.branch_predictor_type==0:
    system.cpu.branchPred = BiModeBP()
elif options.branch_predictor_type==1:
    system.cpu.branchPred = LTAGE()
elif options.branch_predictor_type==2:
    system.cpu.branchPred = LocalBP()
elif options.branch_predictor_type==3:
    system.cpu.branchPred = MultiperspectivePerceptron64KB()
elif options.branch_predictor_type==4:
    system.cpu.branchPred = MultiperspectivePerceptron8KB()
elif options.branch_predictor_type==5:
    system.cpu.branchPred = MultiperspectivePerceptronTAGE64KB()
elif options.branch_predictor_type==6:
    system.cpu.branchPred = MultiperspectivePerceptronTAGE8KB()
elif options.branch_predictor_type==7:
    system.cpu.branchPred = TAGE()
elif options.branch_predictor_type==8:
    system.cpu.branchPred = TAGE_SC_L_64KB()
elif options.branch_predictor_type==9:
    system.cpu.branchPred = TAGE_SC_L_8KB()
elif options.branch_predictor_type==10:
    system.cpu.branchPred = TournamentBP()
else:
    system.cpu.branchPred = TournamentBP()

system.cpu.branchPred.numThreads = 1
system.cpu.branchPred.btb.numEntries = options.btb_entries
system.cpu.branchPred.ras.numEntries = options.ras_entries

# Setup the width for pipelines stages
system.cpu.fetchWidth = options.fetch_width
system.cpu.decodeWidth = options.decode_width
system.cpu.renameWidth = options.rename_width
system.cpu.commitWidth = options.commit_width
system.cpu.dispatchWidth = options.dispatch_width
system.cpu.issueWidth = options.issue_width
system.cpu.wbWidth = options.wb_width
# Setup CPU internal queues
system.cpu.fetchBufferSize = options.fb_entries
system.cpu.fetchQueueSize = options.fq_entries
system.cpu.numIQEntries = options.iq_entries
system.cpu.numROBEntries = options.rob_entries
system.cpu.LQEntries = options.lq_entries
system.cpu.SQEntries = options.sq_entries
# Setup execution units configuration
system.cpu.fuPool = CortexA76_FUPool()
system.cpu.fuPool.FUList[0].count = options.num_fu_cmp # Compare unit
system.cpu.fuPool.FUList[1].count = options.num_fu_intALU # IntALU unit
system.cpu.fuPool.FUList[2].count = options.num_fu_intDIVMUL # IntMulDiv unit
system.cpu.fuPool.FUList[3].count = options.num_fu_FP_SIMD_ALU # FloatingPoint unit
system.cpu.fuPool.FUList[4].count = options.num_fu_read # Read unit
system.cpu.fuPool.FUList[5].count = options.num_fu_write # Read/Write unit

# Setup number of substage for each pipeline stage
system.cpu.fetchToDecodeDelay = 3
system.cpu.decodeToRenameDelay = 2
system.cpu.renameToIEWDelay = 1
system.cpu.issueToExecuteDelay = 2
system.cpu.iewToCommitDelay = 1

# Connect the cache ports on the CPU to it.

system.cpu.icache.connectCPU(system.cpu)
system.cpu.dcache.connectCPU(system.cpu)

# Create a bus to connect L1 caches and L2 cache
system.cpu.l2bus = L2XBar()

# Connect L1 caches the l2bus
system.cpu.icache.connectBus(system.cpu.l2bus)
system.cpu.dcache.connectBus(system.cpu.l2bus)

# Create L2 cache
system.cpu.l2cache = L2Cache(options)
# Connect L2 to l2bus at the CPU side
system.cpu.l2cache.connectCPUSideBus(system.cpu.l2bus)

# Create L3 cache
system.l3cache = L3Cache(options)
# Connect with L2 cache
system.l3cache.connectCPUSideCache(system.cpu.l2cache)


system.membus = SystemXBar()
system.l3cache.connectMemSideBus(system.membus)

# To create the system-wide memory bus:



# Next, we need to connect up a few other ports to make sure that our system will function correctly. 
# We need to create an I/O controller on the CPU and connect it to the memory bus. 
# Also, we need to connect a special port in the system up to the membus. This port is a functional-only port to allow the system to read and write memory.

system.cpu.createInterruptController()
#system.cpu.interrupts[0].pio = system.membus.mem_side_ports # (Only for X86)
#system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports # (Only for X86)
#system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports # (Only for X86)

system.system_port = system.membus.cpu_side_ports

# Next, we need to create a memory controller and connect it to the membus. 
# For this system, we'll use a simple DDR3 controller and it will be responsible for the entire memory range of our system.

system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# Create the process (another SimObject). Then we set the processes command to the command we want to run. 
# Set the CPU to use the process as it's workload, and finally create the functional execution contexts in the CPU.

binary = options.cmd
#for gem5 V21 and beyond, uncomment the following line
system.workload = SEWorkload.init_compatible(binary)

process = Process()
process.cmd = [binary]+options.options.split()
process.cwd = options.cwd
system.cpu.workload = process
system.cpu.createThreads()

# The final thing we need to do is instantiate the system and begin execution. First, we create the Root object. Then we instantiate the simulation. The instantiation process goes through all of the SimObjects we've created in python and creates the C++ equivalents.

root = Root(full_system = False, system = system)
m5.instantiate()

print("Beginning simulation!")
exit_event = m5.simulate()

print('Exiting @ tick {} because {}'
      .format(m5.curTick(), exit_event.getCause()))
