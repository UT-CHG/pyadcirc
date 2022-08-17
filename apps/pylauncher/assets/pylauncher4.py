import copy
import socket
import glob
import json
import math
import os
import random
import re
import stat
import shutil
import stat
import subprocess
import sys
import time
import hostlist3 as hs

# Globals
pylauncherBarrierString = "__barrier__"


def RandomID():
    global randomid
    randomid += 7
    return randomid


def RandomDir():
    return "./pylauncher_tmpdir%d" % RandomID()


def MakeRandomDir():
    dirname = RandomDir()
    print("using random dir:",dirname)
    try:
        os.makedirs(dirname,exist_ok=True)
    except OSError:
        if not os.path.isdir(dirname):
            raise
    #if not os.path.isdir(dirname):
    #os.mkdir(dirname)
    return dirname


def CompactIntList(intlist):
    if len(intlist)==0:
        return ""
    elif len(intlist)==1:
        return str(intlist[0])
    else:
        compact = str(intlist[0]); base = intlist[0]
        if intlist[1]>intlist[0]+1:
            return str(intlist[0])+" "+CompactIntList(intlist[1:])
        else:
            for e in range(1,len(intlist)):
                if intlist[e]>intlist[0]+e:
                    return str(intlist[0])+"-"+str(intlist[e-1])+" "\
                        +CompactIntList(intlist[e:])
            return str(intlist[0])+"-"+str(intlist[-1])


class LauncherException(Exception):
    """A very basic exception mechanism"""

    def __init__(self, str):
        print(str)
        self.str = str

    def __str__(self):
        return self.str


debugtracefile = None
def DebugTraceMsg(msg, sw=False, prefix=""):
    global runtime, debugtracefile
    if not sw: return
    if msg[0]=="\n":
        print
        msg = msg[1:]
    longprefix = ""#"[t=%5.3f] " % time.time()-runtime
    if prefix!="":
        longprefix += prefix+": "
    for l in msg.split("\n"):
        print(longprefix+l)
        longprefix = len(longprefix)*" "
    if debugtracefile is not None:
        debugtracefile.write(msg+"\n")
        os.fsync()


def HostName():
    """This just returns the hostname. See also ``ClusterName``."""
    return socket.gethostname()


def ClusterName():
    """Assuming that a node name is along the lines of ``c123-456.cluster.tacc.utexas.edu``
    this returns the second member. Otherwise it returns None.
    """
    # name detection based on environment variables
    if "TACC_SYSTEM" in os.environ:
        system = os.environ["TACC_SYSTEM"]
        if "TACC_NODE_TYPE" in os.environ:
            system += "-" + os.environ["TACC_NODE_TYPE"]
        return system

    # name detection by splitting TACC hostname
    longname = HostName(); namesplit = longname.split(".")
    # case: mic on stampede
    nodesplit = namesplit[0].split("-")
    if len(nodesplit)==3 and nodesplit[2] in ["mic0","mic1"]:
        return "mic"
    # case: tacc cluster node
    if re.match("nid[0-9]",namesplit[0]):
        return "ls5"
    elif "tacc" in namesplit:
        if len(namesplit)>1 and re.match("c[0-9]",namesplit[0]):
            return namesplit[1]
        else: return None
    # case: unknown
    return None


def JobId():
    """This function is installation dependent: it inspects the environment variable
    that holds the job ID, based on the actual name of the host (see
     ``HostName``): this should only return a number if we are actually in a job.
    """
    hostname = ClusterName()
    if hostname=="ls4":
        return os.environ["JOB_ID"]
    elif hostname in ["ls5","maverick","stampede","stampede2","stampede2-knl","stampede2-skx"]:
        return os.environ["SLURM_JOB_ID"]
    else:
        return None


class HostList:
    """Object describing a list of hosts. Each host is a dictionary
    with a ``host`` and ``core`` field.

    Arguments:

    * list : list of hostname strings
    * tag : something like ``.tacc.utexas.edu`` may be necessary to ssh to hosts in the list

    This is an iteratable object; it yields the host/core dictionary objects.
    """

    def __init__(self, hostlist=[], tag=""):
        self.hostlist = [];
        self.tag = tag;
        self.uniquehosts = []
        for h in hostlist:
            self.append(h)

    def append(self, h, c=0):
        """
        Arguments:

        * h : hostname
        * c (optional, default zero) : core number
        """
        if not re.search(self.tag, h):
            h = h + self.tag
        if h not in self.uniquehosts:
            self.uniquehosts.append(h)
        self.hostlist.append({'host': h, 'core': c})

    def __len__(self):
        return len(self.hostlist)

    def __iter__(self):
        for h in self.hostlist:
            yield h

    def __str__(self):
        return str(self.hostlist)


class SGEHostList(HostList):
    def __init__(self,**kwargs):
        HostList.__init__(self,**kwargs)
        hostfile = os.environ["PE_HOSTFILE"]
        with open(hostfile,"r") as hostfile:
            for h in hostfile:
                line = h.strip(); line = line.split(); host = line[0]; n = line[1]
                for i in range(int(n)):
                    self.append(host,i)


class SLURMHostList(HostList):
    def __init__(self,**kwargs):
        HostList.__init__(self,**kwargs)
        hlist_str = os.environ["SLURM_NODELIST"]
        p = int(os.environ["SLURM_NNODES"])
        N = int(os.environ["SLURM_NPROCS"])
        n=N/p
        hlist = hs.expand_hostlist(hlist_str)
        for h in hlist:
            for i in range(int(n)):
                self.append(h,i)


def HostListByName(**kwargs):
    """Give a proper hostlist. Currently this work for the following TACC hosts:

    * ``ls4``: Lonestar4, using SGE
    * ``ls5``: Lonestar5, using SLURM
    * ``maverick``: Maverick, using SLURM
    * ``stampede``: Stampede, using SLURM
    * ``mic``: Intel Xeon PHI co-processor attached to a compute node

    We return a trivial hostlist otherwise.
    """
    debugs = kwargs.pop("debug","")
    debug = re.search("host",debugs)
    cluster = ClusterName()
    if cluster=="ls4":
        hostlist = SGEHostList(tag=".ls4.tacc.utexas.edu",**kwargs)
    elif cluster=="ls5": # ls5 nodes don't have fully qualified hostname
        hostlist = SLURMHostList(tag="",**kwargs)
    elif cluster=="maverick":
        hostlist = SLURMHostList(tag=".maverick.tacc.utexas.edu",**kwargs)
    elif cluster=="stampede":
        hostlist = SLURMHostList(tag=".stampede.tacc.utexas.edu",**kwargs)
    elif cluster in ["stampede2","stampede2-knl","stampede2-skx"]:
        hostlist = SLURMHostList(tag=".stampede2.tacc.utexas.edu",**kwargs)
    elif cluster=="mic":
        hostlist = HostList( ["localhost" for i in range(60)] )
    else:
        hostlist = HostList(hostlist=[HostName()])
    if debug:
        print("Hostlist on %s : %s" % (cluster,str(hostlist)))
    return hostlist


class HostPoolBase:
    """A base class that defines some methods and sets up
    the basic data structures.

    :param commandexecutor: (keyword, optional, default=``LocalExecutor``) the ``Executor`` object for this host pool
    :param workdir: (keyword, optional) the workdir for the command executor
    :param debug: (keyword, optional) a string of debug types; if this contains 'host', anything derived from ``HostPoolBase`` will do a debug trace
    """

    def __init__(self, **kwargs):
        self.nodes = []
        self.commandexecutor = kwargs.pop("commandexecutor", None)
        # print("set HostPoolBase commandexecutor to",str(self.commandexecutor))
        workdir = kwargs.pop("workdir", None)
        if self.commandexecutor is None:
            self.commandexecutor = LocalExecutor(workdir=workdir)
        elif workdir is not None:
            raise LauncherException("workdir arg is ignored with explicit executor")
        self.debugs = kwargs.pop("debug", "")
        self.debug = re.search("host", self.debugs)
        if len(kwargs) > 0:
            raise LauncherException("Unprocessed HostPool args: %s" % str(kwargs))

    def append_node(self, host="localhost", core=0):
        """Create a new item in this pool by specifying either a Node object
        or a hostname plus core number. This function is called in a loop when a
        ``HostPool`` is created from a ``HostList`` object."""
        if isinstance(host, (Node)):
            node = host
        else:
            node = Node(host, core, nodeid=len(self.nodes))
        self.nodes.append(node)
        self.commandexecutor.setup_on_node(node)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, i):
        return self.nodes[i]

    def hosts(self, pool):
        return [self[i] for i in pool]

    def unique_hostnames(self, pool=None):
        """Return a list of unique hostnames. In general each hostname appears
        16 times or so in a HostPool since each core is listed."""
        if pool is None:
            pool = range(len(self))
        u = []
        for h in self.hosts(pool):
            name = h.hostname
            if not name in u:
                u.append(name)
        return sorted(u)

    def request_nodes(self, request):
        """Request a number of nodes; this returns a HostLocator object"""
        DebugTraceMsg("request %d nodes" % request, self.debug, prefix="Host")
        start = 0;
        found = False
        while not found:
            if start + request > len(self.nodes):
                return None
            for i in range(start, start + request):
                found = self[i].isfree()
                if not found:
                    start = i + 1;
                    break
        if found:
            locator = HostLocator(pool=self, offset=start, extent=request)
            DebugTraceMsg("returning <<%s>>" % str(locator), self.debug, prefix="Host")
            return locator
        else:
            DebugTraceMsg("could not locate", self.debug, prefix="Host")
            return None

    def occupyNodes(self, locator, taskid):
        """Occupy nodes with a taskid

        Argument:
        * locator : HostLocator object
        * taskid : like the man says
        """
        nodenums = range(locator.offset, locator.offset + locator.extent)
        DebugTraceMsg("occupying nodes %s with %d" % (str(nodenums), taskid),
                      self.debug, prefix="Host")
        for n in nodenums:
            self[n].occupyWithTask(taskid)

    def releaseNodesByTask(self, taskid):
        """Given a task id, release the nodes that are associated with it"""
        done = False
        for n in self.nodes:
            if n.taskid == taskid:
                DebugTraceMsg("releasing %s, core %s"
                              % (str(n.hostname), str(n.core)),
                              self.debug, prefix="Host")
                n.release();
                done = True
        if not done:
            raise LauncherException("Could not find nodes associated with id %s"
                                    % str(taskid))

    def release(self):
        """If the executor opens ssh connections, we want to close them cleanly."""
        self.commandexecutor.terminate()

    def final_report(self):
        """Return a string that reports how many tasks were run on each node."""
        counts = [n.tasks_on_this_node for n in self]
        message = """
Host pool of size %d.

Number of tasks executed per node:
max: %d
avg: %d
""" % (len(self), max(counts), sum(counts) / len(counts))
        return message

    def printhosts(self):
        hostlist = ""
        for i, n in enumerate(self.nodes):
            hostlist += "%d : %s\n" % (i, str(n))
        return hostlist.strip()

    def __repr__(self):
        hostlist = str(["%d:%s" % (i, n.nodestring()) for i, n in enumerate(self.nodes)])
        return hostlist


class HostPool(HostPoolBase):
    """A structure to manage a bunch of Node objects.
    The main internal object is the ``nodes`` member, which
    is a list of Node objects.

    :param nhosts: the number of slots in the pool; this will use the localhost
    :param hostlist: HostList object; this takes preference over the previous option
    :param commandexecutor: (optional) a prefixer routine, by default LocalExecutor
    """
    def __init__(self,**kwargs):
        workdir = kwargs.pop("workdir",None)
        if workdir is None:
            executor = LocalExecutor()
        else:
            executor = LocalExecutor(workdir=workdir)
        HostPoolBase.__init__\
            (self,commandexecutor=kwargs.pop("commandexecutor",executor),
             debug=kwargs.pop("debug",""))
        hostlist = kwargs.pop("hostlist",None)
        if hostlist is not None and not isinstance(hostlist,(HostList)):
            raise LauncherException("hostlist argument needs to be derived from HostList")
        nhosts = kwargs.pop("nhosts",None)
        if hostlist is not None:
            if self.debug:
                print("Making hostpool on %s" % str(hostlist))
            nhosts = len(hostlist)
            for h in hostlist:
                self.append_node(host=h['host'],core = h['core'])
        elif nhosts is not None:
            if self.debug:
                print("Making hostpool size %d on localhost" % nhosts)
            localhost = HostName()
            hostlist = [ localhost for i in range(nhosts) ]
            for i in range(nhosts):
                self.append_node(host=localhost)
        else: raise LauncherException("HostPool creation needs n or list")
        #self.nhosts = nhosts
        if len(kwargs)>0:
            raise LauncherException("Unprocessed HostPool args: %s" % str(kwargs))
        DebugTraceMsg("Created host pool from <<%s>>" % str(hostlist),self.debug,prefix="Host")
    def __del__(self):
        """The ``SSHExecutor`` class creates a permanent ssh connection,
        which we try to release by this mechanism."""
        DebugTraceMsg("Releasing nodes",self.debug,prefix="Host")
        for node in self:
            self.commandexecutor.release_from_node(node)


class LocalHostPool(HostPool):
    """A host pool based on just the localhost, using the ``LocalExecutor``. This is for testing purposes.

    :param nhosts: (keyword, optional, default=1) number of times the localhost should be listed
    :param workdir: (keyword, optional) workdir for the commandexecutor
    """
    def __init__(self,**kwargs):
        nhosts = kwargs.pop("nhosts",1)
        self.debug = kwargs.pop("debug","")
        self.workdir=kwargs.pop("workdir",MakeRandomDir())
        HostPool.__init__(
            self, nhosts=nhosts,workdir=self.workdir,
            commandexecutor=LocalExecutor(
                debug=self.debug,workdir=self.workdir,
                #workdir=kwargs.pop("workdir",None),
                force_workdir=kwargs.pop("force_workdir",False)),
            debug=self.debug,**kwargs)


class OneNodePool(HostPoolBase):
    """This class is mostly for testing: it allows for a node to function
    as a host pool so that one can start a task on it."""
    def __init__(self,node,**kwargs):
        HostPoolBase.__init__(self,**kwargs)
        if not isinstance(node,(Node)):
            raise LauncherException("Invalid node type <<%s>>" % node)
        self.append_node(node)


class HostLocator:
    """A description of a subset from a HostPool. A locator
    object is typically created when a task asks for a set of nodes
    from a HostPool. Thus, a locator inherits the executor
    from the host pool from which it is taken.

    The only locator objects allowed at the moment are consecutive subsets.

    :param pool: HostPool (optional)
    :param extent: number of nodes requested
    :param offset: location of the first node in the pool

    """

    def __init__(self, pool=None, extent=None, offset=None):
        if extent is None or offset is None:
            raise LauncherException("Please specify extent and offset")
        self.pool = pool;
        self.offset = offset;
        self.extent = extent

    def __getitem__(self, key):
        index = self.offset + key
        if key >= self.extent:
            raise LauncherException("Index %d out of range for pool" % index)
        node = self.pool[index]
        if not isinstance(node, (Node)):
            raise LauncherException("Strange node type: <<%s>> @ %d" % (str(node), key))
        return node

    def firsthost(self):
        node = self[0]  # .pool[self.offset]
        return node.hostname

    def __len__(self):
        return self.extent

    def __str__(self):
        return "Locator: size=%d offset=%d <<%s>>" % \
               (self.extent, self.offset, str([str(self[i]) for i in range(self.extent)]))


class Executor:
    """Class for starting a commandline on some actual computing device.

    All derived classes need to define a ``execute`` method.

    :param catch_output: (keyword, optional, default=True) state whether command output gets caught, or just goes to stdout
    :param workdir: (optional, default="pylauncher_tmpdir_exec") directory for exec and out files
    :param debug: (optional) string of debug modes; include "exec" to trace this class

    Important note: the ``workdir`` should not already exist. You have to remove it yourself.
    """
    default_workdir = "pylauncher_tmpdir_exec"
    execstring = "exec"
    outstring = "out"

    def __init__(self,**kwargs):
        self.catch_output = kwargs.pop("catch_output",True)
        if self.catch_output:
            self.append_output = kwargs.pop("append_output",None)
        self.debugs = kwargs.pop("debug","")
        self.debug = re.search("exec",self.debugs)
        self.count = 0
        workdir = kwargs.pop("workdir",None)
        if workdir is None:
            self.workdir = self.default_workdir
        else: self.workdir = workdir
        force_workdir = kwargs.pop("force_workdir",False)
        if self.workdir[0]!="/":
            self.workdir = os.getcwd()+"/"+self.workdir
        DebugTraceMsg("Using executor workdir <<%s>>" % self.workdir,
                     self.debug,prefix="Exec")
        if os.path.isfile(self.workdir):
            raise LauncherException(
                "Serious problem creating executor workdir <<%s>>" % self.workdir)
        elif not os.path.isdir(self.workdir):
            os.mkdir(self.workdir)
            # if force_workdir:
            #     os.system("/bin/rm -rf %s" % self.workdir)
            # else:
            #     raise LauncherException(
            #         "I will not reuse an executor workdir <<%s>>" % self.workdir)
        if not self.workdir_is_safe():
            raise LauncherException("Unsafe working dir <<%s>>; pls remove" % self.workdir)
        if len(kwargs)>0:
            raise LauncherException("Unprocessed Executor args: %s" % str(kwargs))

    def workdir_is_safe(self):
        """Test that the working directory is (in) a subdirectory of the cwd"""
        here = os.getcwd(); os.chdir(self.workdir); there = os.getcwd(); os.chdir(here)
        return re.match(here,there) and not here==there

    def cleanup(self):
        if self.workdir_is_safe():
            shutil.rmtree(self.workdir)

    def setup_on_node(self,node):
        return

    def release_from_node(self,node):
        return

    def end_execution(self):
        return

    def smallfilenames(self):
        execfilename = "%s/%s%d" % (self.workdir,self.execstring,self.count)
        if self.catch_output:
            if self.append_output is not None:
                execoutname = self.append_output
            else:
                execoutname =  "%s/%s%d" % (self.workdir,self.outstring,self.count)
        else:
            execoutname = ""
        self.count += 1
        return execfilename,execoutname

    def wrap(self,command):
        """Take a commandline, write it to a small file, and return the
        commandline that sources that file
        """
        execfilename,execoutname = self.smallfilenames()
        if os.path.isfile(execfilename):
            raise LauncherException("exec file already exists <<%s>>" % execfilename)
        f = open(execfilename,"w")
        f.write("#!/bin/bash\n"+command+"\n")
        f.close()
        os.chmod(execfilename,stat.S_IXUSR++stat.S_IXGRP+stat.S_IXOTH+\
                     stat.S_IWUSR++stat.S_IWGRP+stat.S_IWOTH+\
                     stat.S_IRUSR++stat.S_IRGRP+stat.S_IROTH)
        if self.catch_output:
            if self.append_output is not None:
                pipe = ">>"
                #execoutname = self.append_output
            else:
                pipe = ">"
                #execoutname =  "%s/%s%d" % (self.workdir,self.outstring,self.count)
            wrappedcommand = "%s %s %s 2>&1" % (execfilename,pipe,execoutname)
        else:
            wrappedcommand = execfilename
        DebugTraceMsg("file <<%s>>\ncontains <<%s>>\nnew commandline <<%s>>" % \
                          (execfilename,command,wrappedcommand),
                      self.debug,prefix="Exec")
        return wrappedcommand

    def execute(self, command, **kwargs):
        raise LauncherException("Should not call default execute")

    def terminate(self):
        return


class LocalExecutor(Executor):
    """Execute a commandline locally, in the background.

    :param prefix: (keyword, optional, default null string) for recalcitrant shells, the possibility to specify '/bin/sh' or so
    """
    def __init__(self, **kwargs):
        self.prefix = kwargs.pop("prefix", "")
        Executor.__init__(self, **kwargs)
        DebugTraceMsg("Created local Executor", self.debug, prefix="Exec")

    def execute(self, command, **kwargs):
        wrapped = self.wrap(command)
        pre_process = kwargs.pop("pre_process", None)
        post_process = kwargs.pop("post_process", None)
        full_commandline = "%s%s & " % (self.prefix,wrapped)
        if pre_process is not None:
            full_commandline = pre_process + ";" + full_commandline
        if post_process is not None:
            full_commandline = full_commandline + ";" + post_process
        DebugTraceMsg("subprocess execution of:\n<<%s>>" % full_commandline, self.debug,prefix="Exec")
        p = subprocess.Popen(full_commandline, shell=True, env=os.environ, stderr=subprocess.STDOUT)
        # !!! why that os.environ and the env prefix?


class IbrunExecutor(Executor):
    """An Executor derived class for the shift/offset version of ibrun
    that is in use at TACC

    :param pool: (required) ``HostLocator`` object
    :param stdout: (optional) a file that is open for writing; by default ``subprocess.PIPE`` is used
    """
    def __init__(self,**kwargs):
        catch_output = kwargs.pop("catch_output","foo")
        if catch_output!="foo":
            raise LauncherException("IbrunExecutor does not take catch_output parameter")
        Executor.__init__(self,catch_output=False,**kwargs)
        self.popen_object = None

    def execute(self, command, **kwargs):
        """Much like ``SSHExecutor.execute()``, except that it prefixes
        with ``ibrun -n -o``
        """
        pool = kwargs.pop("pool", None)
        pre_process = kwargs.pop("pre_process", None)
        post_process = kwargs.pop("post_process", None)
        remora = kwargs.pop("remora", False)
        if pool is None:
            raise LauncherException("SSHExecutor needs explicit HostPool")
        wrapped_command = self.wrap(command)
        stdout = kwargs.pop("stdout", subprocess.PIPE)
        full_commandline = "ibrun -o %d -n %d %s" % (pool.offset, pool.extent, wrapped_command)
        # Remora for gathering statistics on run
        if remora:
            full_commandline = "remora " full_commandline
        # Pre and post process commands not run in parallel
        if pre_process is not None:
            full_commandline = pre_process + ";" + full_commandline
        if post_process is not None:
            full_commandline = full_commandline + ";" + post_process
        DebugTraceMsg("executed commandline: <<%s>>" % str(full_commandline), self.debug, prefix="Exec")
        p = subprocess.Popen(full_commandline, shell=True, stdout=stdout)
        self.popen_object = p

    def terminate(self):
        if self.popen_object is not None:
            self.popen_object.terminate()


class Node:
    """A abstract object for a slot to execute a job. Most of the time
    this will correspond to a core.

    A node can have a task associated with it or be free."""

    def __init__(self, host=None, core=None, nodeid=-1):
        self.core = core
        self.hostname = host
        self.nodeid = nodeid
        # two initializations before the first ``release`` call:
        self.free = None
        self.tasks_on_this_node = -1
        self.release()

    def occupyWithTask(self, taskid):
        """Occupy a node with a taskid"""
        self.free = False
        self.taskid = taskid

    def release(self):
        """Make a node unoccupied"""
        if self.free is not None and self.free:
            raise LauncherException("Attempting to release a free node")
        self.free = True
        self.taskid = -1
        self.tasks_on_this_node += 1

    def isfree(self):
        """Test whether a node is occupied"""
        return self.free

    def nodestring(self):
        if self.free:
            return "X"
        else:
            return str(self.taskid)

    def __str__(self):
        return "h:%s, c:%s, id:%s" % (self.hostname, str(self.core), str(self.nodeid))


class Completion:
    """Define a completion object for a task.

    The base class doesn't do a lot: it immediately returns true on the
    completion test_sweep.
    """
    workdir = "."

    def __init__(self,taskid=0):
        self.taskid = taskid
        self.stampdir = "."

    def set_workdir(self,workdir):
        self.workdir = workdir
        if self.workdir[0]!="/":
            self.workdir = os.getcwd()+"/"+self.workdir
        # create stampdir. maybe this should be in the attach method?
        if not os.path.isdir(self.workdir):
            os.makedirs(self.workdir)

    def attach(self,txt):
        """Attach a completion to a command, giving a new command"""
        return txt

    def test(self):
        """Test whether the task has completed"""
        return True


class FileCompletion(Completion):
    """FileCompletion is the most common type of completion. It appends
    to a command the creation of a zero size file with a unique name.
    The completion test_sweep then tests for the existence of that file.

    :param taskid: (keyword, required) this has to be unique. Unfortunately we can not test_sweep for that.
    :param stampdir: (keyword, optional, default is self.stampdir, which is ".") directory where the stampfile is left
    :param stamproot: (keyword, optional, default is "expire") root of the stampfile name
    """
    stamproot = "expire"
    stampdir = "."
    def __init__(self,**kwargs):
        taskid = kwargs.pop("taskid",-1)
        if taskid==-1:
            raise LauncherException("Need an explicit task ID")
        Completion.__init__(self,taskid)
        self.set_workdir( kwargs.pop("stampdir",self.stampdir) )
        self.stamproot = kwargs.pop("stamproot",self.stamproot)
        if len(kwargs)>0:
            raise LauncherException("Unprocessed FileCompletion args: %s" % str(kwargs))
    def stampname(self):
        """Internal function that gives the name of the stamp file,
        including directory path"""
        return "%s/%s%s" % (self.workdir,self.stamproot,str(self.taskid))
    def attach(self,txt):
        """Append a 'touch' command to the txt argument"""
        os.system("mkdir -p %s" % self.workdir)
        if re.match('^[ \t]*$',txt):
            return "touch %s" % self.stampname()
        else:
            return "%s ; touch %s" % (txt,self.stampname())
    def test(self):
        """Test for the existence of the stamp file"""
        return os.path.isfile(self.stampname())
    def cleanup(self):
        os.system("rm -f %s" % self.stampname())


class Task:
    """A Task is an abstract object associated with a commandline

    :param command: (required) Commandline object; note that this contains the core count
    :param completion: (keyword, optional) Completion object; if unspecified the trivial completion is used.
    :param taskid: (keyword) identifying number of this task; has to be unique in a job, also has to be equal to the taskid of the completion
    :param debug: (keyword, optional) string of debug keywords
    """
    def __init__(self,command,**kwargs):
        self.command = command["command"]
        self.pre_process = command["pre_process"]
        self.post_process = command["post_process"]
        self.remora = command["remora"]
        # make a default completion if needed
        self.completion = kwargs.pop("completion",None)
        self.taskid = kwargs.pop("taskid",0)
        if self.completion is None:
            self.completion = Completion(taskid=self.taskid)
        if self.taskid!=self.completion.taskid:
            raise LauncherException("Incompatible taskids")
        self.size = command["cores"]
        self.debugs = kwargs.pop("debug","")
        self.debug = re.search("task",self.debugs)
        if len(kwargs)>0:
            raise LauncherException("Unprocessed args: %s" % str(kwargs))
        self.has_started = False
        DebugTraceMsg("created task <<%s>>" % str(self),self.debug,prefix="Task")
        self.nodes = None

    def start_on_nodes(self,**kwargs):
        """Start the task.

        :param pool: HostLocator object (keyword, required) : this describes the nodes on which to start the task
        :param commandexecutor: (keyword, optional) prefixer routine, by default the commandexecutor of the pool is used

        This sets ``self.startime`` to right before the execution begins. We do not keep track
        of the endtime, but instead set ``self.runningtime`` in the ``hasCompleted`` routine.
        """
        self.pool = kwargs.pop("pool",None)
        self.starttick = kwargs.pop("starttick",0)
        if self.pool is None:
            self.pool = LocalHostPool(nhosts=self.size,debug=self.debugs
                                      ).request_nodes(self.size)
        elif isinstance(self.pool,(Node)):
            if self.size>1:
                raise LauncherException("Can not start size=%d on sing Node" % self.size)
            self.pool = OneNodePool( self.pool,debug=self.debugs ).request_nodes(self.size)
        if not isinstance(self.pool,(HostLocator)):
            raise LauncherException("Invalid locator object")
        if len(kwargs)>0:
            raise LauncherException("Unprocessed Task.start_on_nodes args: %s" % str(kwargs))
        # wrap with stamp detector
        wrapped = self.line_with_completion()
        DebugTraceMsg(
            "starting task %d of size %d on <<%s>>\nin cwd=<<%s>>\ncmd=<<%s>>" % \
                (self.taskid,self.size,str(self.pool),os.getcwd(),wrapped),
            self.debug,prefix="Task")
        self.starttime = time.time()
        commandexecutor = self.pool.pool.commandexecutor
        commandexecutor.execute(wrapped, pool=self.pool,
                pre_process=self.pre_process,
                post_process=self.post_process,
                remora=self.remora)
        self.has_started = True
        DebugTraceMsg("started %d" % self.taskid,self.debug,prefix="Task")

    def line_with_completion(self):
        """Return the task's commandline with completion attached"""
        line = re.sub("PYL_ID", str(self.taskid), self.command)
        return self.completion.attach(line)

    def isRunning(self):
        return self.has_started

    def hasCompleted(self):
        """Execute the completion test_sweep of this Task"""
        completed = self.has_started and self.completion.test()
        if completed:
            self.runningtime = time.time()-self.starttime
            DebugTraceMsg("completed %d in %5.3f" % (self.taskid, self.runningtime),
                          self.debug, prefix="Task")
        return completed

    def __repr__(self):
        s = "Task %d, commandline: [%s], pool size %d" \
            % (self.taskid, self.command, self.size)
        return s


class TaskQueue:

    """Object that does the maintains a list of Task objects.
    This is internally created inside a ``LauncherJob`` object."""
    def __init__(self,**kwargs):
        self.queue = []; self.running = []; self.completed = []; self.aborted = []
        self.maxsimul = 0; self.submitdelay = 0
        self.debug = kwargs.pop("debug",False)
        if len(kwargs) > 0:
            raise LauncherException("Unprocessed TaskQueue args: %s" % str(kwargs))

    def isEmpty(self):
        """Test whether the queue is empty and no tasks running"""
        return self.queue==[] and self.running==[]

    def enqueue(self,task):

        """Add a task to the queue"""
        DebugTraceMsg("enqueueing <%s>" % str(task),self.debug,prefix="Queue")
        self.queue.append(task)

    def startQueued(self,hostpool,**kwargs):
        """for all queued, try to find nodes to run it on;
        the hostpool argument is a HostPool object"""
        tqueue = copy.copy(self.queue)
        tqueue.sort( key=lambda x:-x.size )
        max_gap = len(hostpool)
        starttick = kwargs.pop("starttick",0)
        for t in tqueue:
            # go through tasks in descending size
            # if one doesn't fit, skip all of same size
            requested_gap = t.size
            if requested_gap>max_gap:
                continue
            locator = hostpool.request_nodes(requested_gap)
            if locator is None:
                DebugTraceMsg("could not find nodes for <%s>" % str(t),
                              self.debug,prefix="Queue")
                max_gap = requested_gap-1
                continue
            if self.submitdelay>0:
                time.sleep(self.submitdelay)
            DebugTraceMsg("starting task <%s> on locator <%s>" % (str(t),str(locator)),
                          self.debug,prefix="Queue")
            t.start_on_nodes(pool=locator,starttick=starttick)
            hostpool.occupyNodes(locator,t.taskid)
            self.queue.remove(t)
            self.running.append(t)
            self.maxsimul = max(self.maxsimul,len(self.running))

    def find_recently_completed(self):
        """Find the first recently completed task.
        Note the return, not yield.
        """
        for t in self.running:
            if t.hasCompleted():
                DebugTraceMsg(".. job completed: %d" % t.taskid,
                              self.debug,prefix="Queue")
                return t
        return None

    def find_recently_aborted(self,abort_test):
        """Find the first recently aborted task.
        Note the return, not yield.
        """
        for t in self.running:
            if abort_test(t):
                DebugTraceMsg(".. job aborted: %d ran from %d" \
                                  % (t.taskid,t.starttick),
                              self.debug,prefix="Queue")
                return t
        return None

    def __repr__(self):
        completed = sorted( [ t.taskid for t in self.completed ] )
        aborted = sorted( [ t.taskid for t in self.aborted] )
        queued = sorted( [ t.taskid for t in self.queue] )
        running = sorted( [ t.taskid for t in self.running ] )
        return "completed: "+str( CompactIntList(completed) )+\
               "\naborted: " +str( CompactIntList(aborted) )+\
               "\nqueued: " +str( CompactIntList(queued) )+\
               "\nrunning: "+str( CompactIntList(running) )+"."

    def savestate(self):
        state = ""
        state += "queued\n"
        for t in self.queue:
            state += "%s: %s\n" % (t.taskid,t.command)
        state += "running\n"
        for t in self.running:
            state += "%s: %s\n" % (t.taskid,t.command)
        state += "completed\n"
        for t in self.completed:
            state += "%s: %s\n" % (t.taskid,t.command)
        return state
        f = open("queuestate","w")
        f.write("queued\n")
        for t in self.queue:     f.write("%s: %s\n" % (t.taskid,t.command))
        f.write("running\n")
        for t in self.running:   f.write("%s: %s\n" % (t.taskid,t.command))
        f.write("completed\n")
        for t in self.completed: f.write("%s: %s\n" % (t.taskid,t.command))
        f.close()

    def final_report(self):
        """Return a string describing the max and average runtime for each task."""
        times = [ t.runningtime for t in self.completed]
        message = """# tasks completed: %d
tasks aborted: %d
max runningtime: %6.2f
avg runningtime: %6.2f
""" % ( len(self.completed), len(self.aborted),
        max( times ), sum( times )/len(self.completed) )
        return message


class Commandline:
    """A Commandline is basically a dict containing at least the following members:

    * command : a unix commandline
    * cores : an integer core count

    It optionally contains the following parameters:
    * pre_process: a unix pre-process command, to be run before command
    * post_process: a unix post-process command, to be run after command
    * remora: Flag indicating whether to wrap ibrun call with remora monitoring
    * id: a user-supplied task identifier
    """

    def __init__(self, command, cores=1, pre_process=None, post_process=None, remora=False, **kwargs):
        self.data = {'command' : command, "cores": int(cores),
                     'pre_process': pre_process, 'post_process':post_process,
                     'remora': remora, **kwargs}

    def __getitem__(self, ind):
        return self.data.get(ind)

    def __str__(self):
        r = "command=<<%s>>, cores=%d, pre_process=<<%s>>, post_process=<<%s>>, remora=<<%s>>" \
            % (self.data["command"], self.data["cores"], self.data["pre_process"], self.data["post_process"], self.data['remora'])
        return r


class CommandlineGenerator:
    """An iterable class that generates a stream of ``Commandline`` objects.

    The behaviour of the generator depends on the ``nmax`` parameter:

    * nmax is None: exhaust the original list
    * nmax > 0: keep popping until the count is reached; if the initial list is shorter, someone will have to fill it,
    which this class is not capable of
    * nmax == 0 : iterate indefinitely, wait for someone to call the ``finish`` method

    In the second and third scenario it can be the case that the list is empty.
    In that case, the generator will yield a COMMAND that is ``stall``.

    :param list: (keyword, default [] ) initial list of Commandline objects
    :param nax: (keyword, default None) see above for explanation
    """
    def __init__(self, **kwargs):
        self.list = [e for e in kwargs.pop("list",[])];
        self.ncommands = len(self.list)
        self.njobs = 0
        nmax = kwargs.pop("nmax", None)
        if nmax is None:
            if len(self.list)==0:
                raise LauncherException("Empty list requires nmax==0")
            self.nmax = len(self.list)
        else:
            self.nmax = nmax
        debugs = kwargs.pop("debug", "")
        self.debug = re.search("command", debugs)
        if len(kwargs) > 0:
            raise LauncherException("Unprocessed CommandlineGenerator args: %s" % str(kwargs))
        self.stopped = False

    def finish(self):
        """Tell the generator to stop after the commands list is depleted"""
        DebugTraceMsg("declaring the commandline generator to be finished",
                      self.debug,prefix="Cmd")
        self.nmax = self.njobs+len(self.list)

    def abort(self):
        """Stop the generator, even if there are still elements in the commands list"""
        DebugTraceMsg("gettingthe commandline generator to abort",
                      self.debug,prefix="Cmd")
        self.stopped = True

    def next(self):
        """Produce the next Commandline object, or return an object telling that the
        generator is stalling or has stopped"""
        if self.stopped:
            DebugTraceMsg("stopping the commandline generator",
                          self.debug,prefix="Cmd")
            return Commandline("stop")
            # raise StopIteration
        elif (len(self.list) == 0 and self.nmax != 0) or \
                (self.nmax > 0 and self.njobs==self.nmax):
            DebugTraceMsg("time to stop commandline generator",
                          self.debug,prefix="Cmd")
            return Commandline("stop")
            # raise StopIteration
        elif len(self.list)>0:
            j = self.list[0]; self.list = self.list[1:]
            DebugTraceMsg("Popping command off list <<%s>>" % str(j),
                          self.debug,prefix="Cmd")
            self.njobs += 1; return j
        else:
            return Commandline("stall")

    def __iter__(self): return self

    def __len__(self): return len(self.list)


class ListCommandlineGenerator(CommandlineGenerator):
    """A generator from an explicit list of commandlines.

    * cores is 1 by default, other constants allowed.
    """
    def __init__(self,**kwargs):
        cores = kwargs.pop("cores", 1)
        commandlist = [Commandline(l, cores=cores) for l in kwargs.pop("list", [])]
        CommandlineGenerator.__init__(self,list=commandlist,**kwargs)


class FileCommandlineGenerator(CommandlineGenerator):
    """A generator for commandline files:
    blank lines and lines starting with the comment character '#' are ignored

    * cores is 1 by default, other constants allowed.
    * cores=='file' means the file has << count,command >> lines
    * if the file has core counts, but you don't specify the 'file' value, they are ignored.

    :param filename: (required) name of the file with commandlines
    :param cores: (keyword, default 1) core count to be used for all commands
    :param pre_post_process: (keyword, default False) are pre-process and post-process commands per line?
    :param remora: (keyword, default False) run resource monitoring (remora) on job?
    """
    def __init__(self, filename, cores=1, pre_post_process=False, remora=False,
            **kwargs):
        if filename.endswith(".json"):
            commandlist = self._init_from_json(filename, cores=cores)
        else:
            commandlist = self._init_from_csv(filename, cores=cores, pre_post_process=pre_post_process)

        CommandlineGenerator.__init__(self, list=commandlist, **kwargs)

    def _init_from_csv(self, filename, cores=1, pre_post_process=False):
        """Parse a list of commands from a .txt/.csv file, line by line.
        """

        file = open(filename)
        commandlist = []
        count = 0
        for line in file.readlines():
            line = line.strip()
            if re.match('^ *#',line) or re.match('^ *$',line):
                continue # skip blank and comment
            split = line.split(",", 1)
            if len(split) == 1:
                c = cores;
                l = split[0]
            else:
                c, l = split
            td = str(count)
            if pre_post_process:
                split = l.split(";", 2)
                if len(split) < 3:
                    raise LauncherException("No pre/post tasks found <<%s>>" % split)
                pre, l, post = split
            else:
                pre = None
                post = None
            if cores == "file":
                if not re.match("[0-9]+", c):
                    raise LauncherException("First field <<%s>> is not a core count; line:\n<<%s>>" % (c, line))
            else:
                c = cores
            commandlist.append(Commandline(l, cores=c, pre_process=pre, post_process=post))
            count += 1

        return commandlist

    def _init_from_json(self, filename, cores=1):
        """Parse a list of commands from a JSON file

        This allows for much greater flexibility in passing arguments.
        """

        with open(filename, "r") as fp:
            task_list = json.load(fp)

        commandlist = []
        for i, t in enumerate(task_list):
            if "main" not in t:
                raise LauncherException(f"Task {t} has no 'main' command specified!")
            task_cores = t.pop("cores", cores)
            task_id = t.pop("id", i)
            # Pass any extra task parameters directly to the Commandline object
            commandlist.append(
                Commandline(t["main"], cores=task_cores, id=task_id, **t)
            )
        return commandlist

class TaskGenerator:
    """iterator class that can yield the following:

    * a Task instance, or
    * the keyword ``stall``; this indicates that the commandline generator is stalling and this will be resolved when the outer environment does an ``append`` on the commandline generator.
    * the ``pylauncherBarrierString``; in this case the outer environment should not call the generator until all currently running tasks have concluded.
    * the keyword ``stop``; this means that the commandline generator is exhausted. The ``next`` function can be called repeatedly on a stopped generator.

    You can iterate over an instance, or call the ``next`` method. The ``next`` method
    can accept an imposed taskcount number.

    :param commandlinegenerator: either a list of unix commands, or a CommandlineGenerator object
    :param completion: (optional) a function of one variable (the task id) that returns Completion objects
    :param debug: (optional) string of requested debug modes
    :param skip: (optional) list of tasks to skip, this is for restarted jobs

    """
    def __init__(self,commandlines,**kwargs):
        if isinstance(commandlines,(list)):
            self.commandlinegenerator = ListCommandlineGenerator(list=commandlines)
        elif isinstance(commandlines,(CommandlineGenerator)):
            self.commandlinegenerator = commandlines
        else:
            raise LauncherException("Invalid commandline generator object")
        self.taskcount = 0; self.paused = False
        self.debugs = kwargs.pop("debug","")
        self.debug = re.search("task",self.debugs)
        self.completion = kwargs.pop("completion",lambda x:Completion(taskid=x))
        self.skip = kwargs.pop("skip",[])
        if len(kwargs)>0:
            raise LauncherException("Unprocessed TaskGenerator args: %s" % str(kwargs))

    def next(self,imposedcount=None):
        """Deliver a Task object, or a special string:

        * "stall" : the commandline generator will give more, all in good time
        * "stop" : we are totally done
        """
        comm = self.commandlinegenerator.next()
        command = comm["command"]
            # DebugTraceMsg("commandline generator ran out",
            #               self.debug,prefix="Task")
            # command = "stop"
        if command in ["stall","stop"]:
            # the dynamic commandline generator is running dry
            return command
        elif command==pylauncherBarrierString:
            # this is not working yet
            return command
        else:
            if imposedcount is not None:
                taskid = imposedcount
            else:
                taskid = self.taskcount
            self.taskcount += 1
            if taskid in self.skip:
                return self.next(imposedcount=imposedcount)
            else:
                return Task(comm,taskid=taskid,debug=self.debugs,
                            completion=self.completion(taskid),
                            )

    def __iter__(self): return self


class LauncherJob:
    """LauncherJob class. Keyword arguments:

    :param hostpool: a HostPool instance (required)
    :param taskgenerator: a TaskGenerator instance (required)
    :param delay: between task checks  (optional)
    :param debug: list of keywords (optional)
    :param gather_output: (keyword, optional, default None) filename to gather all command output
    :param maxruntime: (keyword, optional, default zero) if nonzero, maximum running time in seconds
    """
    def __init__(self,**kwargs):
        self.debugs = kwargs.pop("debug","")
        self.hostpool = kwargs.pop("hostpool",None)
        if self.hostpool is None:
            raise LauncherException("Need a host pool")
        self.workdir = kwargs.pop("workdir",".")
        DebugTraceMsg("Host pool: <<%s>>" % str(self.hostpool),
                      re.search("host",self.debugs),"Job")
        self.taskgenerator = kwargs.pop("taskgenerator",None)
        if self.taskgenerator is None:
            raise LauncherException("Need a task generator")
        self.delay = kwargs.pop("delay", .5)
        self.queue = TaskQueue(debug=self.debugs)
        self.maxruntime = kwargs.pop("maxruntime", 0)
        self.taskmaxruntime = kwargs.pop("taskmaxruntime", 0)
        self.debug = re.search("job", self.debugs)
        self.completed = 0; self.aborted = 0; self.tock = 0; self.barriertest = None
        self.gather_output = kwargs.pop("gather_output",None)
        self.running_time = 0.0
        if len(kwargs) > 0:
            raise LauncherException("Unprocessed LauncherJob args: %s" % str(kwargs))

    def finish_or_continue(self):
        # auxiliary routine, purely to make ``tick`` look shorter
        if self.queue.isEmpty():
            if self.completed==0:
                raise LauncherException("Done before we started....")
            DebugTraceMsg("Generator and tasks finished",self.debug,prefix="Job")
            message = "finished"
        else:
            DebugTraceMsg("Generator finished; tasks still running",self.debug,prefix="Job")
            message = "continuing"
        return message

    def enqueue_task(self,task):
        # auxiliary routine, purely to make ``tick`` look shorter
        if not isinstance(task,(Task)):
            raise LauncherException("Not a task: %s" % str(task))
        DebugTraceMsg("enqueueing new task <%s>" % str(task),
                      self.debug,prefix="Job")
        self.queue.enqueue(task)

    def tick(self):
        """This routine does a single time step in a launcher's life, and reports back
        to the user. Specifically:

        * It tries to start any currently queued jobs. Also:
        * If any jobs are finished, it detects exactly one, and reports its ID to the user in a message ``expired 123``
        * If there are no finished jobs, it invokes the task generator; this can result in a new task and the return message is ``continuing``
        * if the generator stalls, that is, more tasks will come in the future but none are available now, the message is ``stalling``
        * if the generator is finished and all jobs have finished, the message is ``finished``

        After invoking the task generator, a short sleep is inserted (see the ``delay`` parameter)
        """
        DebugTraceMsg("\ntick %d\nQueue:\n%s" % (self.tock,str(self.queue)),self.debug)
        self.tock += 1
        # see if the barrier test_sweep is completely satisfied
        if self.barriertest is not None:
            if reduce( lambda x,y: x and y,
                       [ t.completed() for t in self.barriertest ] ):
                self.barriertest = None
                message = "continuing"
            else:
                # if the barrier still stands, stall
                message = "stalling"
        else:
            # if the barrier is resolved, queue and test_sweep and whatnot
            self.queue.startQueued(self.hostpool,starttick=self.tock)
            message = None

            self.handle_completed()
            self.handle_aborted()
            message = self.handle_enqueueing()
            #if message in ["stalling","continuing"]:
            time.sleep(self.delay)

        if re.search("host",self.debugs):
            DebugTraceMsg(str(self.hostpool))
        DebugTraceMsg("status: %s" % message,self.debug,prefix="Job")
        return message

    def handle_completed(self):
        message = None
        completed_task = self.queue.find_recently_completed()
        if not completed_task is None:
            self.queue.running.remove(completed_task)
            self.queue.completed.append(completed_task)
            completeID = completed_task.taskid
            DebugTraceMsg("completed: %d" % completeID,self.debug,prefix="Job")
            self.completed += 1
            self.hostpool.releaseNodesByTask(completeID)
            message = "expired %s" % str(completeID)
        return message

    def handle_aborted(self):
        message = None
        aborted_task = self.queue.find_recently_aborted(
            lambda t:self.taskmaxruntime>0 and self.tock-t.starttick>self.taskmaxruntime )
        if not aborted_task is None:
            self.queue.running.remove(aborted_task)
            self.queue.aborted.append(aborted_task)
            completeID = aborted_task.taskid
            DebugTraceMsg("aborted: %d" % completeID,self.debug,prefix="Job")
            self.aborted += 1
            self.hostpool.releaseNodesByTask(completeID)
            message = "truncated %s" % str(completeID)
        return message

    def handle_enqueueing(self):
        message = None
        #try:
        if True:
            task = self.taskgenerator.next()
            if task==pylauncherBarrierString:
                message = "stalling"
                self.barriertest = [ t.completion for t in self.queue.running ]
                DebugTraceMsg("barrier encountered",self.debug,prefix="Job")
            elif task=="stall":
                message = "stalling"
                DebugTraceMsg("stalling",self.debug,prefix="Job")
            elif task=="stop":
                message = self.finish_or_continue()
                DebugTraceMsg("rolling till completion",self.debug,prefix="Job")
            else:
                self.enqueue_task(task)
                message = "enqueueing"
            # except: message = self.finish_or_continue()
        return message

    def post_process(self,taskid):
        DebugTraceMsg("Task %s expired" % str(taskid),self.debug,prefix="Job")

    def run(self):
        """Invoke the launcher job, and call ``tick`` until all jobs are finished."""
        if re.search("host", self.debugs):
            self.hostpool.printhosts()
        start_time = time.time()
        while True:
            elapsed = time.time()-start_time
            runtime = "Time: %d" % int(elapsed)
            if self.maxruntime>0:
                runtime += " (out of %d)" % int(self.maxruntime)
            DebugTraceMsg(runtime,self.debug,prefix="Job")
            if self.maxruntime>0:
                if elapsed>self.maxruntime:
                    break
            res = self.tick()
            # update the restart file
            state_f = open(self.workdir+"/queuestate","w")
            state_f.write( self.queue.savestate() )
            state_f.close()
            # process the result
            # if re.match("expired",res):
            #     self.post_process( res.split(" ",1)[1] )
            if res=="finished":
                break
        self.running_time = time.time()-start_time
        self.finish()

    def finish(self):
        self.hostpool.release()

    def final_report(self):
        """Return a string describing the total running time, as well as
        including the final report from the embedded ``HostPool`` and ``TaskQueue``
        objects."""
        message = """
==========================
Launcherjob run completed.

total running time: %6.2f

%s

%s
==========================
""" % (self.running_time, self.queue.final_report(), self.hostpool.final_report())
        return message


def IbrunLauncher(commandfile,**kwargs):
    """A LauncherJob for a file of small MPI jobs.

    The following values are specified for your convenience:

    * hostpool : based on HostListByName
    * commandexecutor : IbrunExecutor
    * taskgenerator : based on the ``commandfile`` argument
    * completion : based on a directory ``pylauncher_tmp`` with jobid environment variables attached

    :param commandfile: name of file with commandlines (required)
    :param cores: number of cores (keyword, optional, default=4, see ``FileCommandlineGenerator`` for more explanation)
    :param workdir: directory for output and temporary files (optional, keyword, default uses the job number); the launcher refuses to reuse an already existing directory
    :param debug: debug types string (optional, keyword)
    """
    jobid = JobId()
    debug = kwargs.pop("debug","")
    workdir = kwargs.pop("workdir","pylauncher_tmp"+str(jobid) )
    cores = kwargs.pop("cores",4)
    pre_post_process = kwargs.pop("pre_post_process", False)
    job = LauncherJob(
        hostpool=HostPool(hostlist=HostListByName(debug=debug),
                          commandexecutor=IbrunExecutor(workdir=workdir, debug=debug), debug=debug),
        taskgenerator=TaskGenerator(
            FileCommandlineGenerator(commandfile, cores=cores, debug=debug, pre_post_process=pre_post_process),
            completion=lambda x: FileCompletion(taskid=x, stamproot="expire", stampdir=workdir), debug=debug),
        debug=debug, **kwargs)
    job.run()
    print(job.final_report())
