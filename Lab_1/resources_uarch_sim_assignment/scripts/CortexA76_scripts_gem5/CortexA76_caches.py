from m5.objects import Cache
from m5.objects.Prefetcher import *
from m5.objects.ReplacementPolicies import *

class L1Cache(Cache):
    assoc = 2
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 4
    tgts_per_mshr = 20
    
    def connectCPU(self, cpu):
        # need to define this in a base class!
        raise NotImplementedError

    def connectBus(self, bus):
        self.mem_side = bus.cpu_side_ports
    
    def __init__(self, options=None):
        super(L1Cache, self).__init__()
        pass
    
class L1ICache(L1Cache):
    size = '64kB'
    assoc = 4
    is_read_only = True
    prefetcher = StridePrefetcher()
    replacement_policy = TreePLRURP()
    
    def connectCPU(self, cpu):
        self.cpu_side = cpu.icache_port
        
    def __init__(self, options=None):
        super(L1ICache, self).__init__(options)
        self.size = options.l1i_size
        self.assoc = options.l1i_assoc
        self.tag_latency = options.l1i_lat
        self.data_latency = options.l1i_lat
        self.response_latency = options.l1i_lat

class L1DCache(L1Cache):
    size = '64kB'
    assoc = 4
    tag_latency = 4
    data_latency = 4
    response_latency = 4
    mshrs = 20
    write_buffers = 20
    prefetcher = StridePrefetcher()
    replacement_policy = TreePLRURP()
    #port_cpu_side_connection_count = 2
    #port_mem_side_connection_count = 1
    #prefetcher = 
    
    def connectCPU(self, cpu):
        self.cpu_side = cpu.dcache_port
    
    def __init__(self, options=None):
        super(L1DCache, self).__init__(options)
        self.size = options.l1d_size
        self.assoc = options.l1d_assoc
        self.tag_latency = options.l1d_lat
        self.data_latency = options.l1d_lat
        self.response_latency = options.l1d_lat
    
class L2Cache(Cache):
    size = '256kB'
    assoc = 8
    tag_latency = 9
    data_latency = 9
    response_latency = 9
    mshrs = 46
    tgts_per_mshr = 12
    write_buffers = 20
    prefetcher = StridePrefetcher()
    replacement_policy = TreePLRURP()
    
    def connectCPUSideBus(self, bus):
        self.cpu_side = bus.mem_side_ports

    def connectMemSideBus(self, bus):
        self.mem_side = bus.cpu_side_ports
    
    def __init__(self, options=None):
        super(L2Cache, self).__init__()
        self.size = options.l2_size
        self.assoc = options.l2_assoc
        self.tag_latency = options.l2_lat
        self.data_latency = options.l2_lat
        self.response_latency = options.l2_lat

class L3Cache(Cache):
    size = '2MB'
    assoc = 16
    tag_latency = 30
    data_latency = 30
    response_latency = 30
    mshrs = 64
    tgts_per_mshr = 4
    write_buffers = 4
    
    def connectCPUSideCache(self, cache):
        self.cpu_side = cache.mem_side

    def connectMemSideBus(self, bus):
        self.mem_side = bus.cpu_side_ports
    
    def __init__(self, options=None):
        super(L3Cache, self).__init__()
        self.size = options.l3_size
        self.assoc = options.l3_assoc
        self.tag_latency = options.l3_lat
        self.data_latency = options.l3_lat
        self.response_latency = options.l3_lat
