import xml.etree.cElementTree as ET
import sys
import json
import copy



def main():
    #Use this command to run script
    #python Program.py <statFileName> <configFileName> <mcpat-templateFile>

    command = "program.py stats.txt config.json mcpat-template.xml"

    #Must be 3 files in the input
    if len(sys.argv)!=4:
        print ("Input format is incorrect. Use this format: " + command)
        sys.exit(1)

    #Checking file type
    if sys.argv[1][-4:] != ".txt" or sys.argv[2][-5:] != ".json" or sys.argv[3][-4:] != ".xml":
        print ("ERROR: Please use appropriate file: " + command)
        sys.exit(1)

    #tree contains xml-templete file, mapping array for string from xml file to config or stat file, stats array contains all the value from stats file
    global tree, mapping, stats

    #parsing xml file into tree
    try:
        tree = ET.parse(sys.argv[3])
    except IOError:
        print ("******ERROR: Templete File not found******")
        sys.exit(1)

    mapping = {}
    stats = {}

    #First we will count no. of cores from config file
    countCores(sys.argv[2])
    #We will change xml format according to no. of cores, here we have also L2 cache in each core
    #changeXML()

    #read every value from stat file
    readStatsFile(sys.argv[1])
    #read value from config file then write into tree
    readWriteConfigValue(sys.argv[2])
    #write stat value to tree
    writeStatValue(sys.argv[3])

    #handle spaces with specific format in tree component
    indent(tree.getroot())

    #print tree into xml file
    file = sys.argv[2].split("/")[-1][:-5] + ".xml"
    tree.write(file)


def countCores(configFile):
    try:
        file = open(configFile)
    except IOError:
        print ("******ERROR: File not found or can't open config file******")
        sys.exit(1)

    configfile = json.load(file)

    #global variable we will use this elsewhere in the code
    global noCores

    #we are taking size of cpu array from config file
    noCores = len([configfile["system"]["cpu"]])
    #print configfile["system"]["cpu"]
    print ("noCores=", noCores)

    file.close()


def changeXML():
    #this will return root element
    root = tree.getroot()

    #subelement of root will be system
    systemElement = root.find("./component")

    #we are taking all the subelement of system
    core0 = tree.find("./component/component[1]")
    L1Directory0 = tree.find("./component/component[2]")
    L2Directory0 = tree.find("./component/component[3]")
    L2 = tree.find("./component/component[4]")
    L30 = tree.find("./component/component[5]")
    NoC0 = tree.find("./component/component[6]")
    mc = tree.find("./component/component[7]")
    niu = tree.find("./component/component[8]")
    pcie = tree.find("./component/component[9]")
    flashc = tree.find("./component/component[10]")

    #we need multicore in template-xml so first we are removing all the element of system
    for x in range(0,10):
        systemElement.remove(tree.find("./component/component[1]"))

    # system1 = copy.deepcopy(system)

    #first we copy core element then change id and name according to no-wise then append each core into system
    for no in range(0,noCores):
        core = copy.deepcopy(core0)
        core.attrib['id'] = "system.core"+str(no)
        core.attrib['name'] = "core"+str(no)

        #we also need to change id of sub component also
        for com in core.iter("component"):
            Id = com.attrib['id']
            IdArray = Id.split(".")
            Id = ""
            IdArray[1] = "core"+str(no)

            for e in range(0,len(IdArray)):
                Id += IdArray[e]
                Id += "."
            com.attrib['id'] = Id[:-1]

        systemElement.append(core)

    #append other component, if there are more than one make a loop here
    systemElement.append(L1Directory0)
    systemElement.append(L2Directory0)

    #In our case L2 cache is in each core...
    for no in range(0, noCores):
        L2 = copy.deepcopy(L2)
        L2.attrib['id'] = "system.L2"+str(no)
        L2.attrib['name'] = "L2"+str(no)
        systemElement.append(L2)

    #append other component, if there are more than one make a loop here
    systemElement.append(L30)
    systemElement.append(NoC0)
    systemElement.append(mc)
    systemElement.append(niu)
    systemElement.append(pcie)
    systemElement.append(flashc)

    #systemElement.append(system1)

    #tree.write("out.xml")



def readStatsFile(statFile):
    print ("Reading Stat File: " + statFile)

    try:
        File = open(statFile)
    except IOError:
        print ("******ERROR: File not found or can't open stat file******")
        sys.exit(1)

    #Ignoring line starting with "---"
    ignore = "---"
    count = 2

    #for each line in stat file
    for line in File:
        if not ignore in line:

            lineArray = line.split(" ")
            Name = lineArray[0]             #we got name from stat file
            val = ''

            for e in lineArray:
                try:
                    val = int(e)            #int value from each line
                except ValueError:
                    try:
                        val = float(e)      #float value from each line
                    except ValueError:
                        continue

            #print "%d Name: %s \tValue: %s" %(count,Name ,val)
            count += 1

            stats[Name] = val               #storing the value in stat array
    #print stats["system.cpu.commit.loads"]

    File.close()
    print ("Done")


def readWriteConfigValue(configFile):
    global config
    print ("Reading config File: " + configFile)

    try:
        file = open(configFile)
    except IOError:
        print ("******ERROR: File not found or can't open config file******")
        sys.exit(1)

    config = json.load(file)

    #This is parent-child mapping we need parent of any child of xml-tree we will use this
    parent_map = dict((c, p) for p in tree.iter() for c in p)
    print (parent_map)
    root = tree.getroot()

    #After getting value from config file if we have operation on the value that will goes here
    params = {}
    #This array contains default values that are not in config file but we are setting manually from this code
    defaultChangedConfigValue = {}

    defaultChangedConfigValue["system.number_of_cores"] = str(noCores)      #If you have homogeneous core make it 1
    defaultChangedConfigValue["system.number_of_L2s"] = "1"
    defaultChangedConfigValue["system.Private_L2"] = "1"                    #If you don't have L2 cache in each core make it 0 and make homo to 1
    defaultChangedConfigValue["system.homogeneous_cores"] = "0"             #we don't have homogeneous core otherwise make it 1
    defaultChangedConfigValue["system.homogeneous_L2s"] = "0"
    defaultChangedConfigValue["system.number_of_L3s"] = "1"                 #If there are more than one L3 cache then change it, also add loop in changeXML() and writeStatValue()
    defaultChangedConfigValue["system.mc.number_mcs"] = "1"                 #If more than one change it
    defaultChangedConfigValue["system.number_of_NoCs"] = "1"
    defaultChangedConfigValue["system.number_of_L1Directories"] = "1"
    defaultChangedConfigValue["system.number_of_L2Directories"] = "1"

    for no in range(0,noCores):
        defaultChangedConfigValue["system.cpu.icache.cache_policy"] = "1"
        defaultChangedConfigValue["system.cpu.icache.bank"] = "1"
        try:
            #Calculating throughput if we can't find the value in stat file make it 0
            defaultChangedConfigValue["system.cpu.icache.throughput"] = float(
                stats["system.cpu.icache.overall_accesses::total"]) / float(stats["sim_ticks"])
        except KeyError:
            defaultChangedConfigValue["system.cpu.icache.throughput"] = 0

        defaultChangedConfigValue["system.cpu.dcache.cache_policy"] = "1"
        defaultChangedConfigValue["system.cpu.dcache.bank"] = "1"
        try:
            defaultChangedConfigValue["system.cpu.dcache.throughput"] = float(stats["system.cpu" + str(
                no) + ".dcache.overall_accesses::total"]) / float(stats["sim_ticks"])
        except KeyError:
            defaultChangedConfigValue["system.cpu.dcache.throughput"] = 0



        defaultChangedConfigValue["system.cpu.branchPred.bank"] = "1"
        try:
            defaultChangedConfigValue["system.cpu.branchPred.throughput"] = float(stats["system.cpu.branchPred.lookups"])/float(stats["sim_ticks"])
        except KeyError:
            defaultChangedConfigValue["system.cpu.branchPred.throughput"] = 0

        #if defaultChangedConfigValue["system.cpu.branchPred.throughput"] == 0:
        #    defaultChangedConfigValue["system.cpu.branchPred.throughput"] = 1

        defaultChangedConfigValue["system.cpu.l2cache.cache_policy"] = "1"
        defaultChangedConfigValue["system.cpu.l2cache.bank"] = "1"
        try:
           defaultChangedConfigValue["system.cpu.l2cache.throughput"] = float(stats["system.cpu" + str(no) + ".l2cache.overall_accesses::total"]) / float(stats["sim_ticks"])
        except KeyError:
            defaultChangedConfigValue["system.cpu.l2cache.throughput"] = 1



    defaultChangedConfigValue["system.l3cache.cache_policy"] = "1"
    defaultChangedConfigValue["system.l3cache.bank"] = "1"

    #Setting default throughput to 1 for l3cache cache
    try:
        defaultChangedConfigValue["system.l3cache.throughput"] = float(stats["system.l3cache.overall_accesses::total"])/float(stats["sim_ticks"])
    except KeyError:
        defaultChangedConfigValue["system.l3cache.throughput"] = 1

    #Check for architecture type
    X86 = "Arm"
    archType = X86

    #We are setting INT_EXE and FP_EXE that will be use for calculating pipeline depth
    if archType[:3] == "X86":
        INT_EXE = 2
        FP_EXE = 8
    elif archType[:3] == "Arm":
        INT_EXE = 3
        FP_EXE = 7
    else:
        INT_EXE = 3
        FP_EXE = 6

    if X86[:3] == "Arm":
        X86 = "0"
    else:
        X86 = "1"

    for no in range(0,noCores):
        defaultChangedConfigValue["system.core" + str(no) + ".x86"] = X86


    #To calculate pipeline depth for each core
    for no in range(0,noCores):
        base = getConfValue("system.cpu.fetchToDecodeDelay") + getConfValue("system.cpu.decodeToRenameDelay") + getConfValue("system.cpu.renameToIEWDelay") + getConfValue("system.cpu.iewToCommitDelay")

        cToDecode = getConfValue("system.cpu.commitToDecodeDelay")
        cToFetch = getConfValue("system.cpu.commitToFetchDelay")
        cToIew = getConfValue("system.cpu.commitToIEWDelay")
        cToRename = getConfValue("system.cpu.commitToRenameDelay")

        maxBase = max(cToDecode, cToFetch, cToIew, cToRename)

        pipeline_depthValue = str(INT_EXE + base + maxBase) + "," + str(FP_EXE + base + maxBase)
        defaultChangedConfigValue["system.core"+str(no)+".pipeline_depth"] = pipeline_depthValue

    #Here we have mapping from template-xml file to name from config or stat file
    params["system.clk_domain.clock"] = 1000000/getConfValue("system.clk_domain.clock")
    print (params["system.clk_domain.clock"])
    
    mapping["system.number_of_cores"] = "system.number_of_cores"                        #we are not changing name if can't find in config or stat file
    mapping["system.number_of_L1Directories"] = "system.number_of_L1Directories"
    mapping["system.number_of_L2Directories"] = "system.number_of_L2Directories"
    mapping["system.number_of_L2s"] = "system.number_of_cores"
    mapping["system.Private_L2"] = "system.Private_L2"
    mapping["system.number_of_L3s"] = "system.number_of_L3s"
    mapping["system.number_of_NoCs"] = "system.number_of_NoCs"
    mapping["system.homogeneous_cores"] = "system.homogeneous_cores"
    mapping["system.homogeneous_L2s"] = "system.homogeneous_L2s"
    mapping["system.homogeneous_L1Directories"] = "default"
    mapping["system.homogeneous_L2Directories"] = "default"
    mapping["system.homogeneous_L3s"] = "default"
    mapping["system.homogeneous_ccs"] = "default"
    mapping["system.homogeneous_NoCs"] = "default"
    mapping["system.core_tech_node"] = "default"
    mapping["system.target_core_clockrate"] = "system.clk_domain.clock"
    mapping["system.temperature"] = "default"
    mapping["system.number_cache_levels"] = "default"
    mapping["system.interconnect_projection_type"] = "default"
    mapping["system.device_type"] = "default"
    mapping["system.longer_channel_device"] = "default"
    mapping["system.Embedded"] = "system.Embedded"
    if X86 == "0":
        defaultChangedConfigValue["system.Embedded"] = "1"
    else:
        defaultChangedConfigValue["system.Embedded"] = "0"

    mapping["system.power_gating"] = "default"
    mapping["system.opt_clockrate"] = "default"
    mapping["system.machine_bits"] = "default"
    mapping["system.virtual_address_width"] = "default"
    mapping["system.physical_address_width"] = "default"
    mapping["system.virtual_memory_page_size"] = "default"

    #For multi-core we are using loop for mapping
    for no in range(0,noCores):

        mapping["system.core"+str(no)+".clock_rate"] = "system.clk_domain.clock"
        mapping["system.core"+str(no)+".vdd"] = "default"
        mapping["system.core"+str(no)+".power_gating_vcc"] = "default"
        mapping["system.core"+str(no)+".opt_local"] = "default"
        mapping["system.core"+str(no)+".instruction_length"] = "default"
        mapping["system.core"+str(no)+".opcode_width"] = "default"
        mapping["system.core"+str(no)+".x86"] = "default"
        mapping["system.core"+str(no)+".micro_opcode_width"] = "default"
        mapping["system.core"+str(no)+".machine_type"] = "default"
        mapping["system.core"+str(no)+".number_hardware_threads"] = "system.cpu.numThreads"
        mapping["system.core"+str(no)+".fetch_width"] = "system.cpu.fetchWidth"
        mapping["system.core"+str(no)+".number_instruction_fetch_ports"] = "default"
        mapping["system.core"+str(no)+".decode_width"] = "system.cpu.decodeWidth"
        mapping["system.core"+str(no)+".issue_width"] = "system.cpu.issueWidth"
        mapping["system.core"+str(no)+".peak_issue_width"] = "default"
        mapping["system.core"+str(no)+".commit_width"] = "system.cpu.commitWidth"
        mapping["system.core"+str(no)+".fp_issue_width"] = "default"
        mapping["system.core"+str(no)+".prediction_width"] = "default"
        mapping["system.core"+str(no)+".pipelines_per_core"] = "default"
        mapping["system.core"+str(no)+".pipeline_depth"] = "system.core"+str(no)+".pipeline_depth"
        mapping["system.core"+str(no)+".ALU_per_core"] = "default"
        mapping["system.core"+str(no)+".MUL_per_core"] = "default"
        mapping["system.core"+str(no)+".FPU_per_core"] = "default"
        mapping["system.core"+str(no)+".instruction_buffer_size"] = "system.cpu.fetchBufferSize"
        mapping["system.core"+str(no)+".decoded_stream_buffer_size"] = "default"
        mapping["system.core"+str(no)+".instruction_window_scheme"] = "default"
        mapping["system.core"+str(no)+".instruction_window_size"] = "system.cpu.numIQEntries"
        params["system.cpu.numIQEntries"] = getConfValue("system.cpu.numIQEntries")/2

        mapping["system.core"+str(no)+".fp_instruction_window_size"] = "system.cpu.numIQEntries"
        params["system.cpu.numIQEntries"] = getConfValue("system.cpu.numIQEntries")/2

        mapping["system.core"+str(no)+".ROB_size"] = "system.cpu.numROBEntries"
        mapping["system.core"+str(no)+".archi_Regs_IRF_size"] = "default"
        mapping["system.core"+str(no)+".archi_Regs_FRF_size"] = "default"
        mapping["system.core"+str(no)+".phy_Regs_IRF_size"] = "system.cpu.numPhysIntRegs"
        mapping["system.core"+str(no)+".phy_Regs_FRF_size"] = "system.cpu.numPhysFloatRegs"
        mapping["system.core"+str(no)+".rename_scheme"] = "default"
        mapping["system.core"+str(no)+".checkpoint_depth"] = "default"
        mapping["system.core"+str(no)+".register_windows_size"] = "default"
        mapping["system.core"+str(no)+".LSU_order"] = "default"
        mapping["system.core"+str(no)+".store_buffer_size"] = "system.cpu.SQEntries"
        mapping["system.core"+str(no)+".load_buffer_size"] = "system.cpu.LQEntries"
        mapping["system.core"+str(no)+".memory_ports"] = "default"
        mapping["system.core"+str(no)+".RAS_size"] = "system.cpu.branchPred.ras.numEntries"
        mapping["system.core"+str(no)+".number_of_BPT"] = "default"
        #mapping["system.core"+str(no)+".predictor.local_predictor_size"] = "system.cpu.branchPred.localPredictorSize"
        #mapping["system.core"+str(no)+".predictor.local_predictor_entries"] = "system.cpu.branchPred.localHistoryTableSize"
        #mapping["system.core"+str(no)+".predictor.global_predictor_entries"] = "system.cpu.branchPred.globalPredictorSize"
        #mapping["system.core"+str(no)+".predictor.global_predictor_bits"] = "system.cpu.branchPred.globalCtrBits"
        #mapping["system.core"+str(no)+".predictor.chooser_predictor_entries"] = "system.cpu.branchPred.choicePredictorSize"
        #mapping["system.core"+str(no)+".predictor.chooser_predictor_bits"] = "system.cpu.branchPred.choiceCtrBits"
        mapping["system.core"+str(no)+".predictor.local_predictor_size"] = "default"
        mapping["system.core"+str(no)+".predictor.local_predictor_entries"] = "default"
        mapping["system.core"+str(no)+".predictor.global_predictor_entries"] = "default"
        mapping["system.core"+str(no)+".predictor.global_predictor_bits"] = "default"
        mapping["system.core"+str(no)+".predictor.chooser_predictor_entries"] = "default"
        mapping["system.core"+str(no)+".predictor.chooser_predictor_bits"] = "default"

        mapping["system.core"+str(no)+".itlb.number_entries"] = "system.cpu.mmu.itb.size"
        mapping["system.core"+str(no)+".icache.icache_config"] = "system.cpu" +".icache.size,system.cpu.icache.tags.block_size,system.cpu" +".icache.assoc,system.cpu" +".icache.bank,system.cpu" +".icache.throughput,system.cpu" +".icache.response_latency,system.cpu" +".icache.tags.block_size,system.cpu" +".icache.cache_policy"

        mapping["system.core"+str(no)+".icache.buffer_sizes"] = "system.cpu.icache.mshrs,system.cpu.icache.mshrs,system.cpu.icache.mshrs,system.cpu.icache.mshrs"
        mapping["system.core"+str(no)+".dtlb.number_entries"] = "system.cpu.mmu.dtb.size"
        mapping["system.core"+str(no)+".dcache.dcache_config"] = "system.cpu.dcache.size,system.cpu.dcache.tags.block_size,system.cpu.dcache.assoc,system.cpu.dcache.bank,system.cpu.dcache.throughput,system.cpu.dcache.response_latency,system.cpu.dcache.tags.block_size,system.cpu.dcache.cache_policy"
        mapping["system.core"+str(no)+".dcache.buffer_sizes"] = "system.cpu.dcache.mshrs,system.cpu.dcache.mshrs,system.cpu.dcache.mshrs,system.cpu.dcache.mshrs"
        mapping["system.core"+str(no)+".number_of_BTB"] = "default"
        mapping["system.core"+str(no)+".BTB.BTB_config"] = "system.cpu.branchPred.btb.numEntries,system.cpu.branchPred.btb.tagBits,system.cpu.branchPred.indirectBranchPred.indirectWays,system.cpu.branchPred.bank,system.cpu.branchPred.throughput,system.cpu.icache.response_latency"
        
    mapping["system.L2.L2_config"] = "system.cpu" + ".l2cache.size,system.cpu" + ".l2cache.tags.block_size,system.cpu" + ".l2cache.assoc,system.cpu" + ".l2cache.bank,system.cpu"+ ".l2cache.throughput,system.cpu" + ".l2cache.response_latency,system.cpu" + ".l2cache.tags.block_size,system.cpu" + ".l2cache.cache_policy"
    mapping["system.L2.buffer_sizes"] = "system.cpu.l2cache.mshrs,system.cpu.l2cache.mshrs,system.cpu.l2cache.mshrs,system.cpu.l2cache.mshrs"
    mapping["system.L2.clockrate"] = "system.clk_domain.clock"
    mapping["system.L2.ports"] = "default"
    mapping["system.L2.device_type"] = "default"
    mapping["system.L2.vdd"] = "default"
    mapping["system.L2.Merged_dir"] = "default"
    mapping["system.L2.power_gating_vcc"] = "default"


    #If more than 1, change the value
    for no in range (0,1):
        mapping["system.L1Directory"+str(no)+".Directory_type"] = "default"
        mapping["system.L1Directory"+str(no)+".Dir_config"] = "default"
        mapping["system.L1Directory"+str(no)+".buffer_sizes"] = "default"
        mapping["system.L1Directory"+str(no)+".clockrate"] = "system.clk_domain.clock"
        mapping["system.L1Directory"+str(no)+".ports"] = "default"
        mapping["system.L1Directory"+str(no)+".device_type"] = "default"
        mapping["system.L1Directory" + str(no) + ".vdd"] = "default"
        mapping["system.L1Directory" + str(no) + ".power_gating_vcc"] = "default"

        mapping["system.L2Directory"+str(no)+".Directory_type"] = "default"
        mapping["system.L2Directory"+str(no)+".Dir_config"] = "default"
        mapping["system.L2Directory"+str(no)+".buffer_sizes"] = "default"
        mapping["system.L2Directory"+str(no)+".clockrate"] = "system.clk_domain.clock"
        mapping["system.L2Directory"+str(no)+".ports"] = "default"
        mapping["system.L2Directory"+str(no)+".device_type"] = "default"
        mapping["system.L2Directory" + str(no) + ".vdd"] = "default"
        mapping["system.L2Directory" + str(no) + ".power_gating_vcc"] = "default"
        
    #If more than 1, change the value
    for no in range(0, 1):
        mapping["system.L3"+str(no)+".L3_config"] = "system.l3cache.size,system.l3cache.tags.block_size,system.l3cache.assoc,system.l3cache.bank,system.l3cache.throughput,system.l3cache.response_latency,system.l3cache.tags.block_size,system.l3cache.cache_policy"
        mapping["system.L3"+str(no)+".clockrate"] = "default"
        mapping["system.L3"+str(no)+".ports"] = "default"
        mapping["system.L3"+str(no)+".device_type"] = "default"
        mapping["system.L3"+str(no)+ ".vdd"] = "default"
        mapping["system.L3"+str(no)+".buffer_sizes"] = "system.l3cache.mshrs,system.l3cache.mshrs,system.l3cache.mshrs,system.l3cache.mshrs"
        mapping["system.L3"+str(no)+ ".Merged_dir"] = "default"
        mapping["system.L3"+str(no)+ ".power_gating_vcc"] = "default"
        
        mapping["system.NoC"+str(no)+".clockrate"] = "default"
        mapping["system.NoC"+str(no)+ ".vdd"] = "default"
        mapping["system.NoC"+str(no)+ ".power_gating_vcc"] = "default"
        mapping["system.NoC"+str(no)+".type"] = "default"
        mapping["system.NoC"+str(no)+".horizontal_nodes"] = "default"
        mapping["system.NoC"+str(no)+".vertical_nodes"] = "default"
        mapping["system.NoC"+str(no)+".has_global_link"] = "default"
        mapping["system.NoC"+str(no)+".link_throughput"] = "default"
        mapping["system.NoC"+str(no)+".link_latency"] = "default"
        mapping["system.NoC"+str(no)+".input_ports"] = "default"
        mapping["system.NoC"+str(no)+".output_ports"] = "default"
        mapping["system.NoC"+str(no)+".flit_bits"] = "default"
        mapping["system.NoC"+str(no)+".virtual_channel_per_port"] = "default"
        mapping["system.NoC"+str(no)+ ".input_buffer_entries_per_vc"] = "default"
        mapping["system.NoC"+str(no)+".chip_coverage"] = "default"
        mapping["system.NoC"+str(no)+".link_routing_over_percentage"] = "default"

    mapping["system.mc.type"] = "default"
    mapping["system.mc.vdd"] = "default"
    mapping["system.mc.power_gating_vcc"] = "default"
    mapping["system.mc.mc_clock"] = "system.clk_domain.clock"
    mapping["system.mc.peak_transfer_rate"] = "default"
    mapping["system.mc.block_size"] = "system.mem_ctrl.dram.write_buffer_size"
    mapping["system.mc.number_mcs"] = "system.mc.number_mcs"
    mapping["system.mc.memory_channels_per_mc"] = "default"
    mapping["system.mc.number_ranks"] = "system.mem_ctrl.dram.ranks_per_channel"
    mapping["system.mc.req_window_size_per_channel"] = "default"
    mapping["system.mc.IO_buffer_size_per_channel"] = "default"
    mapping["system.mc.databus_width"] = "default"
    mapping["system.mc.addressbus_width"] = "default"
    mapping["system.mc.withPHY"] = "default"
    mapping["system.niu.type"] = "default"
    mapping["system.niu.vdd"] = "default"
    mapping["system.niu.power_gating_vcc"] = "default"
    mapping["system.niu.clockrate"] = "default"
    mapping["system.niu.number_units"] = "default"
    mapping["system.pcie.type"] = "default"
    mapping["system.pcie.vdd"] = "default"
    mapping["system.pcie.power_gating_vcc"] = "default"
    mapping["system.pcie.withPHY"] = "default"
    mapping["system.pcie.clockrate"] = "default"
    mapping["system.pcie.number_units"] = "default"
    mapping["system.pcie.num_channels"] = "default"
    mapping["system.flashc.number_flashcs"] = "default"
    mapping["system.flashc.type"] = "default"
    mapping["system.flashc.vdd"] = "default"
    mapping["system.flashc.power_gating_vcc"] = "default"
    mapping["system.flashc.withPHY"] = "default"
    mapping["system.flashc.peak_transfer_rate"] = "default"
    
    #Writing config value into xml-tree
    for child in root.iter('param'):                            #look only for 'param' from xml-tree
        name = child.attrib['name']                             #Got the name and value of each 'param'
        val = child.attrib['value']

        name = parent_map[child].attrib['id']+"."+name          #In the name we only have like clock_rate, we are using path like system.core0.clock_rate
        foundVal = getConfValue(mapping[name])                  #Get value from config file

        if mapping[name] == "default":                          #If are not changing this 'param'
            continue
        elif "," in mapping[name]:                              #If ',' in the mapping get separate value for each of them
            mappingArray = mapping[name].split(",")
            ans = ""

            for x in mappingArray:
                findMltVal = getConfValue(x)                    #Getting value from config file if it's -1 then not found in config file

                #associativity must be power of 2 if not then we are taking next power of 2
                if "assoc" in x:
                    if findMltVal and (findMltVal & (findMltVal-1)):
                        p = 1
                        while p<findMltVal:
                            p <<= 1
                        findMltVal = p

                #dcache size must be 8kb if not we are changing it
                if "dcache.size" in x:
                    if findMltVal<8192:
                        findMltVal = 8192

                #If can't find the value in config file look into default array if still not finding than make it to 1
                if findMltVal == -1:
                    if x in defaultChangedConfigValue:
                        ans += str(defaultChangedConfigValue[x])
                    else:
                        ans += str(1)
                        print (x + " not found in config file setting default value to 1...")
                else:
                    if findMltVal is None:
                        findMltVal = 0

                    ans += str(findMltVal)
                ans += ","


            #print "%s\t%s" % (name, ans[:-1])
            child.attrib['value'] = str(ans[:-1])

        #If can't find the value in config file look into default array
        elif foundVal == -1:

            if mapping[name] in defaultChangedConfigValue:
                val = defaultChangedConfigValue[mapping[name]]
                child.attrib['value'] = str(val)
            else:
                print (name + " Not found in config file setting 0")
                child.attrib['value'] = "0"
        else:
            if foundVal == "" or foundVal == "[]":
                print (name + " Value is null in config file")

            #if we found value in config file but have done operation on the value look into params array
            elif mapping[name] in params:
                val = params[mapping[name]]
                child.attrib['value'] = str(val)

            else:
                val = foundVal

                if val is None:
                    val = 0

                child.attrib['value'] = str(val)
                #print "%s\t%s" %(name,getConfValue(mapping[name]))

    print ("Done")




def getConfValue(confStr):

    #don't allow ',' in input string
    if "," in confStr:
        return -1

    confStrArray = confStr.split(".")
    currentConfig = config                                          #whole config file

    try:
        coreNo = confStrArray[1][3:]                                #check if like cpu0 than get the core no.
    except IndexError:
        return -1

    currentPath = ""

    for com in confStrArray:
        currentPath += com

        if not currentConfig:
            break

        #system.cpu0 will not be found we have make it like [system][cpu][0]
        if com not in currentConfig:
           if com[:3] == "cpu":
               currentConfig = currentConfig["cpu"]
           else:
               return -1                                            #value not found in config file
        elif com == "mem_ctrl":
            currentConfig = currentConfig[com]
        elif com == "isa":
            currentConfig = currentConfig[com]
        elif currentConfig:
            currentConfig = currentConfig[com]                      #every time assign sub component

        currentPath += "."
    #print currentConfig["system"]["cpu"][0]

    #clock will return array like [555] so we are taking value
    if confStrArray[2] == "clock":
        return currentConfig[0]
    else:
        return currentConfig



def writeStatValue(mcpatTemplateFile):
    print ("Reading mcpatTemplateFile:%s" + mcpatTemplateFile)

    #parent-child mapping
    parent_map = dict((c, p) for p in tree.iter() for c in p)

    root = tree.getroot()

    mapping["system.total_cycles"] = "system.total_cycles"
    mapping["system.idle_cycles"] = "system.cpu.idleCycles"

    #if we can't find some parameter in stats file, we are adding that into stat array
    stats["system.total_cycles"] = 0
    stats["system.idle_cycles"] = 0
    stats["system.clk_domain.clock"] = 1000000/stats["system.clk_domain.clock"]
    print (stats["system.clk_domain.clock"])

    for no in range(0,noCores):
        if noCores == 1:
            stats["system.total_cycles"] = stats["system.cpu.numCycles"]                               #If we have only 1 core so we have only 'cpu' or 'core' not 'cpu0' or 'core0'
            stats["system.idle_cycles"] = stats.get("system.cpu.idleCycles", stats.get("system.cpu.num_idle_cycles"))
        else:
            try:
                stats["system.total_cycles"] += stats["system.cpu.numCycles"]
            except KeyError:
                stats["system.cpu" + str(no) + ".numCycles"] = 0
            try:
                stats["system.idle_cycles"] += stats["system.cpu.idleCycles"]
            except KeyError:
                stats["system.cpu" + str(no) + ".idleCycles"] = 0

    mapping ["system.busy_cycles"] = "system.busy_cycles"
    # stats["system.busy_cycles"] = stats.get("system.cpu.num_busy_cycles", stats["system.total_cycles"] - stats["system.idle_cycles"])
    stats["system.busy_cycles"] = stats["system.total_cycles"] - stats["system.idle_cycles"]


    for no in range(0,noCores):

        mapping ["system.core"+str(no)+".total_instructions"] = "system.cpu.decode.decodedInsts"

        mapping ["system.core"+str(no)+".int_instructions"] = "system.cpu.intAluAccesses"
        mapping ["system.core"+str(no)+".fp_instructions"] = "system.cpu.fpAluAccesses"

        mapping ["system.core"+str(no)+".branch_instructions"] = "system.cpu.executeStats0.numBranches"
        mapping ["system.core"+str(no)+".branch_mispredictions"] = "system.cpu.iew.branchMispredicts"
        mapping ["system.core"+str(no)+".load_instructions"] = "system.cpu.executeStats0.numLoadInsts"
        mapping ["system.core"+str(no)+".store_instructions"] = "system.cpu.executeStats0.numStoreInsts"
        mapping ["system.core"+str(no)+".committed_int_instructions"] = "system.cpu.commitStats0.numIntInsts"
        mapping ["system.core"+str(no)+".committed_fp_instructions"] = "system.cpu.commitStats0.numFpInsts"
        mapping ["system.core"+str(no)+".committed_instructions"] = "system.cpu.commitStats0.numOps"
        #if noCores == 1:
        #    try:
        #        stats["system.core.committed_instructions"] = stats["system.cpu.commit.integer"] + stats["system.cpu.commit.floating"]
        #    except KeyError:
        #        stats["system.core.committed_instructions"] = stats.get("system.cpu.committedInsts", 0)
        #else:
        #    try:
        #        stats ["system.core"+str(no)+".committed_instructions"] = stats["system.cpu.commit.int_insts"] + stats["system.cpu.commit.floating"]
        #    except KeyError:
        #        stats["system.core" + str(no) + ".committed_instructions"] = stats.get("system.cpu"+ str(no) +".committedInsts", 0)

        mapping ["system.core"+str(no)+".pipeline_duty_cycle"] = "system.cpu.ipc"
        mapping ["system.core"+str(no)+".total_cycles"] = "system.cpu.numCycles"
        mapping ["system.core"+str(no)+".idle_cycles"] = "system.cpu.idleCycles"
        mapping ["system.core"+str(no)+".busy_cycles"] = "system.core"+str(no)+".busy_cycles"

        try:
            stats["system.core"+str(no)+".busy_cycles"] = stats["system.cpu.numCycles"] - stats["system.cpu.idleCycles"]
        except KeyError:
            stats["system.core"+str(no)+".busy_cycles"] = 0


        mapping ["system.core"+str(no)+".ROB_reads"] = "system.cpu.rob.reads"
        mapping ["system.core"+str(no)+".ROB_writes"] = "system.cpu.rob.writes"
        mapping ["system.core"+str(no)+".rename_reads"] = "system.cpu.rename.lookups"

        mapping ["system.core"+str(no)+".rename_writes"] = "system.core"+str(no)+".rename_writes"

        try:
            stats ["system.core"+str(no)+".rename_writes"] = stats["system.cpu.rename.renamedOperands"]*stats["system.cpu.rename.intLookups"]/stats["system.cpu.rename.lookups"]
        except KeyError:
            stats["system.core" + str(no) + ".rename_writes"] = 0


        mapping ["system.core"+str(no)+".fp_rename_reads"] = "system.core"+str(no)+".fp_rename_reads"
        
        try:
            stats["system.core" + str(no) + ".fp_rename_reads"] = stats["system.cpu.rename.lookups"] - stats["system.cpu.rename.intLookups"] - stats["system.cpu.rename.vecLookups"]
        except KeyError:
            stats["system.core" + str(no) + ".fp_rename_reads"] = 0
            
        mapping ["system.core"+str(no)+".fp_rename_writes"] = "system.core"+str(no)+".fp_rename_writes"
       
        try:
            stats["system.core" + str(no) + ".fp_rename_writes"] = stats["system.cpu.rename.renamedOperands"] * stats["system.core" + str(no) + ".fp_rename_reads"]/stats["system.cpu.rename.lookups"]
        except KeyError:
            stats["system.core" + str(no) + ".fp_rename_writes"] = 0
        
        mapping["system.core" + str(no) + ".rename_accesses"] = "system.core" + str(no) + ".rename_accesses"
        
        try:
            stats ["system.core"+str(no)+".rename_accesses"] = stats ["system.cpu.rename.intLookups"] + stats ["system.core.rename_writes"]
        except KeyError:
            stats["system.core"+str(no)+".rename_accesses"] = 0


        mapping["system.core" + str(no) + ".fp_rename_accesses"] = "system.core" + str(no) + ".fp_rename_accesses"

        try:
            stats ["system.core"+str(no)+".fp_rename_accesses"] = stats ["system.cpu.rename.fp_rename_lookups"] + stats ["system.core.fp_rename_writes"]
        except KeyError:
            stats["system.core"+str(no)+".fp_rename_accesses"] = 0
       

        mapping ["system.core"+str(no)+".inst_window_reads"] = "system.cpu.intInstQueueReads"
        mapping ["system.core"+str(no)+".inst_window_writes"] = "system.cpu.intInstQueueWrites"
        mapping ["system.core"+str(no)+".inst_window_wakeup_accesses"] = "system.cpu.intInstQueueWakeupAccesses"
        mapping ["system.core"+str(no)+".fp_inst_window_reads"] = "system.cpu.fpInstQueueReads"
        mapping ["system.core"+str(no)+".fp_inst_window_writes"] = "system.cpu.fpInstQueueWrites"
        mapping ["system.core"+str(no)+".fp_inst_window_wakeup_accesses"] = "system.cpu.fpInstQueueWakeupAccesses"
        mapping ["system.core"+str(no)+".int_regfile_reads"] = "system.cpu.executeStats0.numIntRegReads"
        mapping ["system.core"+str(no)+".float_regfile_reads"] = "system.cpu.executeStats0.numFpRegReads"
        mapping ["system.core"+str(no)+".int_regfile_writes"] = "system.cpu.executeStats0.numIntRegWrites"
        mapping ["system.core"+str(no)+".float_regfile_writes"] = "system.cpu.executeStats0.numFpRegWrites"

        mapping ["system.core"+str(no)+".function_calls"] = "system.cpu.commit.functionCalls"

        mapping ["system.core"+str(no)+".context_switches"] = "default"

        mapping ["system.core"+str(no)+".ialu_accesses"] = "system.cpu.intAluAccesses"
        mapping ["system.core"+str(no)+".fpu_accesses"] = "system.cpu.fpAluAccesses"
        mapping ["system.core"+str(no)+".mul_accesses"] = "system.core"+str(no)+".mul_accesses"

        try:
            #stats["system.core"+str(no)+".mul_accesses"] = stats["system.cpu.statIssuedInstType_0::IntDiv"] * stats["system.cpu.statIssuedInstType_0::No_OpClass"] + stats["system.cpu.statIssuedInstType_0::IntMult"] * stats["system.cpu.statIssuedInstType_0::No_OpClass"]
            stats["system.core"+str(no)+".mul_accesses"] = stats["system.cpu.statIssuedInstType_0::IntDiv"] + stats["system.cpu.statIssuedInstType_0::IntMult"] 
        except KeyError:
            stats["system.core.mul_accesses"] = 0
       

        mapping ["system.core"+str(no)+".cdb_alu_accesses"] = "system.cpu.intAluAccesses"
        mapping ["system.core"+str(no)+".cdb_mul_accesses"] = "system.core"+str(no)+".cdb_mul_accesses"

        try:
            stats ["system.core"+str(no)+".cdb_mul_accesses"] = stats ["system.core.mul_accesses"]
        except KeyError:
            stats["system.core"+str(no)+".cdb_mul_accesses"] = 0

        mapping ["system.core"+str(no)+".cdb_fpu_accesses"] = "system.cpu.fpAluAccesses"

        mapping ["system.core"+str(no)+".IFU_duty_cycle"] = "default"
        mapping ["system.core"+str(no)+".BR_duty_cycle"] = "default"
        mapping ["system.core"+str(no)+".LSU_duty_cycle"] = "default"
        mapping ["system.core"+str(no)+".MemManU_I_duty_cycle"] = "default"
        mapping ["system.core"+str(no)+".MemManU_D_duty_cycle"] = "default"
        mapping ["system.core"+str(no)+".ALU_duty_cycle"] = "default"
        mapping ["system.core"+str(no)+".MUL_duty_cycle"] = "default"
        mapping ["system.core"+str(no)+".FPU_duty_cycle"] =  "default"
        mapping ["system.core"+str(no)+".ALU_cdb_duty_cycle"] = "default"
        mapping ["system.core"+str(no)+".MUL_cdb_duty_cycle"] = "default"
        mapping ["system.core"+str(no)+".FPU_cdb_duty_cycle"] = "default"

        mapping["system.core"+str(no)+".itlb.total_accesses"] = "system.cpu.mmu.itb.accesses"
        mapping["system.core"+str(no)+".itlb.total_misses"] = "system.cpu.mmu.itb.misses"
        mapping["system.core"+str(no)+".itlb.conflicts"] = "default"

        mapping["system.core"+str(no)+".icache.read_accesses"] = "system.cpu.icache.ReadReq.accesses::total"
        mapping["system.core"+str(no)+".icache.read_misses"] = "system.cpu.icache.ReadReq.misses::total"
        mapping["system.core"+str(no)+".icache.conflicts"] = "default"
        mapping["system.core"+str(no)+".dtlb.total_accesses"] = "system.cpu.mmu.dtb.accesses"
        mapping["system.core"+str(no)+".dtlb.total_misses"] = "system.cpu.mmu.dtb.misses"
        mapping["system.core"+str(no)+".dtlb.conflicts"] = "default"
        mapping["system.core"+str(no)+".dcache.read_accesses"] = "system.cpu.dcache.ReadReq.accesses::total"
        mapping["system.core"+str(no)+".dcache.write_accesses"] = "system.cpu.dcache.WriteReq.accesses::total"
        mapping["system.core"+str(no)+".dcache.read_misses"] = "system.cpu.dcache.ReadReq.misses::total"
        mapping["system.core"+str(no)+".dcache.write_misses"] = "system.cpu.dcache.WriteReq.misses::total"
        mapping["system.core"+str(no)+".dcache.conflicts"] = "default"
        mapping["system.core"+str(no)+".BTB.read_accesses"] = "system.cpu.branchPred.BTBLookups"
        mapping["system.core"+str(no)+".BTB.write_accesses"] = "system.cpu.branchPred.BTBHits"

    mapping["system.L1Directory0.read_accesses"] = "default"
    mapping["system.L1Directory0.write_accesses"] = "default"
    mapping["system.L1Directory0.read_misses"] = "default"
    mapping["system.L1Directory0.write_misses"] = "default"
    mapping["system.L1Directory0.conflicts"] = "default"
    mapping["system.L1Directory0.duty_cycle"] = "default"
    mapping["system.L2Directory0.read_accesses"] = "default"
    mapping["system.L2Directory0.write_accesses"] = "default"
    mapping["system.L2Directory0.read_misses"] = "default"
    mapping["system.L2Directory0.write_misses"] = "default"
    mapping["system.L2Directory0.conflicts"] = "default"
    mapping["system.L2Directory0.duty_cycle"] = "default"


    mapping["system.L2.read_accesses"] = "system.cpu.l2cache.overallHits::total"
    mapping["system.L2.write_accesses"] = "system.L2.write_accesses"

    try:
        stats["system.L2.write_accesses"] = stats["system.cpu.l2cache.overallAccesses::total"] - stats["system.cpu.l2cache.overallHits::total"]
    except KeyError:
        stats["system.L2.write_accesses"] = 0
    try:
        stats["system.L2.write_misses"] = 0
    except KeyError:
        stats["system.L2.write_misses"] = 0

    mapping["system.L2.read_misses"] = "system.cpu.l2cache.overallMisses::total"
    mapping["system.L2.write_misses"] = "system.L2.write_misses"

    mapping["system.L2.conflicts"] = "default"
    mapping["system.L2.duty_cycle"] = "default"
    mapping["system.L2.coherent_read_accesses"] = "default"
    mapping["system.L2.coherent_write_accesses"] = "default"
    mapping["system.L2.coherent_read_misses"] = "default"
    mapping["system.L2.coherent_write_misses"] = "default"
    mapping["system.L2.dir_duty_cycle"] = "default"

    
    #If there are more than one L3s add loop here
    mapping["system.L30.read_accesses"] = "system.l3cache.overallHits::total"
    mapping["system.L30.write_accesses"] = "system.L30.write_accesses"
    mapping["system.L30.read_misses"] = "system.l3cache.overallMisses::total"
    mapping["system.L30.write_misses"] = "system.L30.write_misses"
    try:
        stats["system.L30.write_accesses"] = stats["system.l3cache.overallAccesses::total"] - stats["system.l3cache.overallHits::total"]
    except KeyError:
        stats["system.L30.write_accesses"] = 0

    try:
        stats["system.L30.write_misses"] = stats["system.l3cache.overallMisses::total"] - stats["system.l3cache.overallMisses::total"]
    except KeyError:
        stats["system.L30.write_misses"] = 0

    mapping["system.L30.conflicts"] = "default"
    mapping["system.L30.duty_cycle"] = "default"
    mapping["system.L30.coherent_read_accesses"] = "default"
    mapping["system.L30.coherent_write_accesses"] = "default"
    mapping["system.L30.coherent_read_misses"] = "default"
    mapping["system.L30.coherent_write_misses"] = "default"
    mapping["system.L30.dir_duty_cycle"] = "default"

    mapping["system.NoC0.total_accesses"] = "default"
    mapping["system.NoC0.duty_cycle"] = "default"

    mapping["system.mc.memory_reads"] = "system.mem_ctrl.readReqs"
    mapping["system.mc.memory_writes"] = "system.mem_ctrl.writeReqs"
    mapping["system.mc.memory_accesses"] = "system.mc.memory_accesses"
    stats["system.mc.memory_accesses"] = stats["system.mem_ctrl.readReqs"] + stats["system.mem_ctrl.writeReqs"]

    mapping["system.niu.duty_cycle"] = "default"
    mapping["system.niu.total_load_perc"] = "default"
    mapping["system.pcie.duty_cycle"] = "default"
    mapping["system.pcie.total_load_perc"] = "default"
    mapping["system.flashc.duty_cycle"] = "default"
    mapping["system.flashc.total_load_perc"] = "default"

    #Writing stat parameter into xml-tree
    for child in root.iter('stat'):                                     #look only 'stat' parameter from xml-tree
        name = child.attrib['name']
        val = child.attrib['value']
        #print name, parent_map[child].attrib['id'], "\n"
        name = parent_map[child].attrib['id']+"."+name                  #we are using path like system.core0.clock_rate

        if mapping[name]=="default":                                    #If found default in mapping than we are not changing it
            continue

        #if noCores == 1:                                                #If only 1 core but, we have mapping like 'system.core0' so we are removing 0 from core0
        #    print mapping[name]
        #    nmArray = mapping[name].split(".")
            #print nmArray[1][:3]
        #    if nmArray[1][:3] == "cpu":
        #        mapping[name] = mapping[name][:10]+mapping[name][11:]
        #        print mapping[name]
        #    elif nmArray[1][:4] == "core":
        #        mapping[name] = mapping[name][:11]+mapping[name][12:]
        #        print mapping[name]

        try:
            #print "n=",name,"m=",mapping[name],"v=",stats[mapping[name]], "\n"
            val = stats[mapping[name]]                                  #Get the value from stat file
        except KeyError:
            #still not found stat value make it 0
            #print mapping[name], "KeyError\n"
            val = 0
            #print "%s not found in Stat File setting default value to 0..." %mapping[name]

        if str(val)=="nan":
            val = 0
            #if value is nan make it 0

        #change the value in xml-tree
        child.attrib['value'] = str(val)

        #print "%s\t%s" %(name,val)


    print ("Done")
    #tree.write("out.xml")



def indent(elem, level=0):
    #we are spacing xml-tree with specific format
    #If we are in root and we got sub component of root that is system then we add space before system component
    #so for tail we are decreasing space before component

    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i



if __name__ == '__main__':
    main()
