import copy
import pdb
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
import logging
from pathlib import Path
import tempfile

# Flags, all possible are: job, host, task, exec
DEBUG_FLAGS = "job+host+task+exec+cmd"

# TACC Systems
TACC_SYSTEMS = [
        "frontera",
        "ls6",
        "maverick",
        "stampede",
        "stampede2",
        "stampede2-knl",
        "stampede2-skx"]

# create logger
logger = logging.getLogger(__name__)
logformat ='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=logformat,
    datefmt="%Y-%m-%d %H:%M:%S"
)


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

def set_debug_flags(flags):
    """Set globla debug flags"""
    global DEBUG_FLAGS
    if flags is not None:
        DEBUG_FLAGS = flags
        logger.info(f"Set debug flags to {flags}")


def debug_msg(msg, prefix=""):
    """Print debug messages if necessary"""
    if prefix in DEBUG_FLAGS:
        logger.debug(f"{prefix} : {msg}")


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
    longname = HostName()
    namesplit = longname.split(".")
    nodesplit = namesplit[0].split("-")
    if len(namesplit)>1 and re.match("c[0-9]",namesplit[0]):
        return namesplit[1]
    else:
        return None

    return None


class HostList:
    """
    HostList

    A list of hosts, with a ``host`` and ``core`` field. This is an iteratable
    object; it yields the host/core dictionary objects.

    Arguments:

    Attributes
    ----------
    hostlist :
        list of hostname strings
    tag :
        something like ``.tacc.utexas.edu``, indicating full path to host.
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
        unique_hosts = set([x['host'] for x in self.hostlist])
        hostlist = [(host,CompactIntList([y['core'] for y in self.hostlist if y['host'] in host])) for host in unique_hosts]
        return str(hostlist)

class SLURMHostList(HostList):
    def __init__(self, hostlist=[], tag=""):
        super().__init__(hostlist=hostlist, tag=tag)
        hlist_str = os.environ["SLURM_NODELIST"]
        p = int(os.environ["SLURM_NNODES"])
        N = int(os.environ["SLURM_NPROCS"])
        n=N/p
        hlist = hs.expand_hostlist(hlist_str)
        for h in hlist:
            for i in range(int(n)):
                self.append(h,i)

def HostListByName():
    """Give a proper hostlist. Currently this work for the following TACC hosts:

    * ``ls6``: Lonestar6, using SLURM
    * ``maverick``: Maverick, using SLURM
    * ``stampede``: Stampede, using SLURM
    * ``mic``: Intel Xeon PHI co-processor attached to a compute node

    We return a trivial hostlist otherwise.
    """
    cluster = ClusterName()
    if cluster in TACC_SYSTEMS:
        hostlist = SLURMHostList(tag=f".{cluster}.tacc.utexas.edu")
    elif cluster=="mic":
        hostlist = HostList( ["localhost" for i in range(60)] )
    else:
        hostlist = HostList(hostlist=[HostName()])

    logger.debug(f"Hostlist on {cluster} : {hostlist}")

    return hostlist


class HostPool():
    """
    Host Pool Class

    Defines a set of nodes to execute jobs on.

    Parameters
    ----------
    """

    """A structure to manage a bunch of Node objects.
    The main internal object is the ``nodes`` member, which
    is a list of Node objects.

    :param nhosts: the number of slots in the pool; this will use the localhost
    :param hostlist: HostList object; this takes preference over the previous option
    """
    def __init__(self, hostlist=None, nhosts=None):
        self.nodes = []
        if hostlist is not None and not isinstance(hostlist,(HostList)):
            raise LauncherException("hostlist argument needs to be derived from HostList")
        if hostlist is not None:
            debug_msg(f"Making hostpool on {hostlist}",prefix="host")
            nhosts = len(hostlist)
            for h in hostlist:
                self.append_node(host=h['host'],core = h['core'])
        elif nhosts is not None:
            debug_msg(f"Making hostpool size {nhosts} on localhost",
                    prefix="host")
            localhost = HostName()
            hostlist = [ localhost for i in range(nhosts) ]
            for i in range(nhosts):
                self.append_node(host=localhost)
        else:
            raise LauncherException("HostPool creation needs n or list")

        debug_msg(f"Created host pool from <<{hostlist}>>",prefix="host")

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, i):
        return self.nodes[i]

    def hosts(self, pool):
        return [self[i] for i in pool]

    def append_node(self, host="localhost", core=0):
        """Create a new item in this pool by specifying either a Node object
        or a hostname plus core number. This function is called in a loop when a
        ``HostPool`` is created from a ``HostList`` object."""
        if isinstance(host, (Node)):
            node = host
        else:
            node = Node(host, core, nodeid=len(self.nodes))
        self.nodes.append(node)

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
        debug_msg(f"request {request} nodes", prefix="host")
        logging.debug(f'request {request}')

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
            debug_msg(f"returning <<{locator}>>", prefix="host")
            return locator
        else:
            debug_msg("could not locate", prefix="host")
            return None

    def occupyNodes(self, locator, taskid):
        """Occupy nodes with a taskid

        Argument:
        * locator : HostLocator object
        * taskid : like the man says
        """
        nodenums = range(locator.offset, locator.offset + locator.extent)
        debug_msg(f"occupying nodes {nodenums} with {taskid}", prefix="host")
        for n in nodenums:
            self[n].occupyWithTask(taskid)

    def releaseNodesByTask(self, taskid):
        """Given a task id, release the nodes that are associated with it"""
        done = False
        for n in self.nodes:
            if n.taskid == taskid:
                debug_msg(f"releasing {n.hostname}, core {n.core}", prefix="host")
                n.release();
                done = True
        if not done:
            raise LauncherException("Could not find nodes associated with id %s"
                                    % str(taskid))

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


class HostLocator:
    """A description of a subset from a HostPool. A locator
    object is typically created when a task asks for a set of nodes
    from a HostPool.

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


class IbrunExecutor():
    """
    IbrunExector

    An class executing ibrun commands on TACC resources. Uses shift/offset
    version of ibrun that is in use at TACC.

    :param pool: (required) ``HostLocator`` object
    :param stdout: (optional) a file that is open for writing; by default ``subprocess.PIPE`` is used
    :param catch_output: (keyword, optional, default=True) state whether command output gets caught, or just goes to stdout
    :param workdir: (optional, default="pylauncher_tmpdir_exec") directory for exec and out files
    :param debug: (optional) string of debug modes; include "exec" to trace this class

    Important note: the ``workdir`` should not already exist. You have to remove it yourself.
    """
    execstring = "exec"
    outstring = "out"

    def __init__(self,
            workdir,
            log=True,
            **kwargs):

        self.workdir = workdir
        self.logfile = self.workdir / 'log'
        self.errfile = self.workdir / 'err'
        self.execfile = self.workdir / 'exec'

        self.log = log
        debug_msg(f"Using executor workdir <<{self.workdir}>>", prefix="exec")
        self.popen_object = None

    def wrap(self, command, pool, cdir=None, pre_process=None, post_process=None):
        """Take a commandline, write it to a small file, and return the
        commandline that sources that file
        """
        f = open(self.execfile,"w")
        f.write("#!/bin/bash\n\n")
        if pre_process is not None:
            f.write(f"{pre_process}\n")
            f.write("if [ $? -ne 0 ]\n")
            f.write("then\n")
            f.write("  exit 1\n")
            f.write("fi\n")
        if cdir is not None:
            f.write(f"cd {cdir}\n")
            f.write(f"cwd=$(pwd)\n")
        f.write(f"ibrun -o {pool.offset} -n {pool.extent} {command}\n")
        if cdir is not None:
            f.write(f"cd $cwd\n")
        f.write("if [ $? -ne 0 ]\n")
        f.write("then\n")
        f.write("  exit 1\n")
        f.write("fi\n")
        if post_process is not None:
            f.write(f"{post_process}\n")
            f.write("if [ $? -ne 0 ]\n")
            f.write("then\n")
            f.write("  exit 1\n")
            f.write("fi\n")
            f.close()
        os.chmod(self.execfile,stat.S_IXUSR++stat.S_IXGRP+stat.S_IXOTH+\
                     stat.S_IWUSR++stat.S_IWGRP+stat.S_IWOTH+\
                     stat.S_IRUSR++stat.S_IRGRP+stat.S_IROTH)

        new_command = f"{self.execfile} > {self.logfile} 2> {self.errfile}"
        debug_msg(f"new command stored in : {self.execfile}", prefix="exec")
        debug_msg(f"new command: {new_command}", prefix="exec")

        return new_command

    def execute(self, command, pool):
        """
        Execute a command on subprocesses.
        Much like ``SSHExecutor.execute()``, except that it prefixes
        with ``ibrun -n -o``
        """
        command = self.wrap(
                command["command"],
                pool,
                cdir=command["dir"],
                pre_process=command["pre_process"],
                post_process=command["post_process"])
        # Pre and post process commands not run in parallel
        p = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE)
        self.popen_object = p
        debug_msg(f"Process {p.pid} : <<{command}>>", prefix="exec")

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


class Task:
    """A Task is an abstract object associated with a commandline

    :param command: (required) Commandline object; note that this contains the core count
    :param taskid: (keyword) identifying number of this task; has to be unique in a job
    :param debug: (keyword, optional) string of debug keywords
    """
    def __init__(self, command, workdir, taskid=0):

        self.command = command
        self.cores = command["cores"]
        self.taskid = taskid
        self.workdir = workdir / f'task_{taskid}'
        self.workdir.mkdir()

        self.nodes = None
        self.executor = None
        self.start_ts = None
        self.end_ts = None
        self.runningtime = None
        self.rc = None

        debug_msg(f"created task <<{self}>>", prefix="task")

    def start_on_nodes(self, nodes):
        """Start the task.

        :param pool: HostLocator object (keyword, required) : this describes the nodes on which to start the task
        :param commandexecutor: (keyword, optional) prefixer routine, by default the commandexecutor of the pool is used

        This sets ``self.startime`` to right before the execution begins. We do not keep track
        of the endtime, but instead set ``self.runningtime`` in the ``hasCompleted`` routine.
        """

        debug_msg(f"starting {self.taskid}", prefix="task")
        self.nodes = nodes
        self.executor = IbrunExecutor(self.workdir)
        self.executor.execute(self.command,pool=self.nodes)
        self.start_ts = time.time()
        debug_msg(f"started {self.taskid}",prefix="task")

    def get_rc(self):
        """Check process to see if completed"""
        self.rc = self.executor.popen_object.poll()
        if self.rc is not None:
            self.end_ts = time.time()
            self.runningtime = self.end_ts - self.start_ts
            debug_msg(f"completed {self.taskid} in {self.runningtime:5.3f}",
                          prefix="task")
            return self.rc

        return -1

    def __repr__(self):
        s = "Task %d, commandline: [%s], pool size %d" \
            % (self.taskid, self.command, self.cores)
        return s


class TaskQueue:

    """Object that does the maintains a list of Task objects.
    This is internally created inside a ``LauncherJob`` object."""
    def __init__(self, workdir, maxsimul=0, submitdelay=0):

        self.workdir = workdir
        self.queue = []
        self.running = []
        self.completed = []
        self.errored = []
        self.aborted = []
        self.maxsimul = maxsimul
        self.submitdelay = submitdelay


    def isEmpty(self):
        """Test whether the queue is empty and no tasks running"""
        return self.queue==[] and self.running==[]

    def enqueue(self, task):
        """Add a task to the queue"""
        debug_msg(f"enqueueing <{task}>", prefix="queue")
        self.queue.append(task)

    def start_queued(self, hostpool):
        """
        Start queued jobs

        for all queued, try to find nodes to run it on;
        the hostpool argument is a HostPool object
        """
        tqueue = copy.copy(self.queue)
        tqueue.sort( key=lambda x:-x.cores )
        max_gap = len(hostpool)
        for t in tqueue:
            # go through tasks in descending size
            # if one doesn't fit, skip all of same size
            requested_gap = t.cores
            if requested_gap>max_gap:
                continue
            locator = hostpool.request_nodes(requested_gap)
            if locator is None:
                debug_msg(f"could not find nodes for <{t}>",
                              prefix="queue")
                max_gap = requested_gap-1
                continue
            if self.submitdelay>0:
                time.sleep(self.submitdelay)
            debug_msg(f"starting task <{t}> on locator <{locator}>",
                          prefix="queue")
            t.start_on_nodes(locator)
            hostpool.occupyNodes(locator, t.taskid)
            self.queue.remove(t)
            self.running.append(t)
            self.maxsimul = max(self.maxsimul,len(self.running))

    def update(self):
        """Update queue"""
        task_ids = []
        running = []
        for t in self.running:
            rc = t.get_rc()
            if rc == 0:
                self.completed.append(t)
                task_ids.append(t.taskid)
                debug_msg(f"Added {t.taskid} to completed: {self.completed}",
                        prefix="queue")
            elif rc > 0:
                self.errored.append(t)
                task_ids.append(t.taskid)
                debug_msg(f"Added {t.taskid} to errored: {self.errored}",
                        prefix="queue")
            else:
                rt = time.time() - t.start_ts
                # TODO: implement timeout for aborted
                running.append(t)
        self.running = running
        return task_ids

    def __repr__(self):
        completed = sorted( [ t.taskid for t in self.completed ] )
        aborted = sorted( [ t.taskid for t in self.aborted] )
        queued = sorted( [ t.taskid for t in self.queue] )
        running = sorted( [ t.taskid for t in self.running ] )
        return "completed: "+str( CompactIntList(completed) )+\
               "\naborted: " +str( CompactIntList(aborted) )+\
               "\nqueued: " +str( CompactIntList(queued) )+\
               "\nrunning: "+str( CompactIntList(running) )+"."

    def final_report(self):
        """Return a string describing the max and average runtime for each task."""
        times = [ t.runningtime for t in self.completed]
        message = f"# tasks completed: {len(self.completed)}"
        return message


class Commandline:
    """A Commandline is basically a dict containing at least the following members:

    * command : a unix commandline
    * cores : an integer core count

    It optionally contains the following parameters:
    * pre_process: a unix pre-process command, to be run before command
    * post_process: a unix post-process command, to be run after command
    * id: a user-supplied task identifier
    """

    def __init__(self, command, cores=1, **kwargs):
        self.data = {'command' : command,
                "cores": int(cores), **kwargs}

    def __getitem__(self, ind):
        return self.data.get(ind, None)

    def __str__(self):
        r = f"command=<<{self.__getitem__('command')}>>"
        r += f", cores={self.__getitem__('cores')}"
        if self.__getitem__('pre_process') is not None:
            r += f", pre=<<{self.__getitem__('pre_process')}>>"
        if self.__getitem__('post_process') is not None:
            r += f", post=<<{self.__getitem__('post_process')}>>"
        return r


class LauncherJob:
    """LauncherJob class. Keyword arguments:

    :param hostpool: a HostPool instance (required)
    :param taskgenerator: a TaskGenerator instance (required)
    :param delay: between task checks  (optional)
    :param debug: list of keywords (optional)
    :param gather_output: (keyword, optional, default None) filename to gather all command output
    :param maxruntime: (keyword, optional, default zero) if nonzero, maximum running time in seconds
    """
    def __init__(self,
            commandfile: str,
            hostpool = None,
            delay: float=0.5,
            workdir: str=None,
            maxruntime: float=1e10):

        self.commands = self.commands_from_json(commandfile)
        self.taskcount = 0

        self.hostpool = hostpool
        self.delay = delay
        self.workdir = workdir
        self.maxruntime = maxruntime

        if self.workdir is None:
            self.workdir = Path(
                    tempfile.mkdtemp(
                        prefix=f'job{os.environ["SLURM_JOB_ID"]}-',
                        dir=Path.cwd()))
        else:
            self.workdir = Path(workdir) if type(workdir)!=Path else workdir
            self.workdir.mkdir(exist_ok=True)

        # Initialize Launcher Job Task Queue
        self.queue = TaskQueue(workdir)
        self.running_time = 0.0

        if self.hostpool is None:
            self.hostpool = HostPool(hostlist=HostListByName())

        # Print hostpool being used if host debug set
        debug_msg(f"Host pool: <<{self.hostpool}>>",prefix="job")

    def commands_from_json(self, filename, cores=1):
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


    def next_command(self):
        """Deliver a Task object, or a special string:

        * "stall" : the commandline generator will give more, all in good time
        * "stop" : we are totally done
        """
        try:
            command = self.commands.pop()
        except IndexError:
            return None

        taskid = self.taskcount
        self.taskcount += 1
        return Task(command, self.workdir, taskid=taskid)

    def run(self):
        """
        Runs tasks in taskgenerator using configured hostpool.
        """
        start_time = time.time()
        debug_msg("Starting launcher job",prefix="job")
        while True:
            elapsed = time.time()-start_time
            if elapsed>self.maxruntime:
                debug_msg("Exceeded max runtime",prefix="job")
                break

            # Start queued jobs
            self.queue.start_queued(self.hostpool)

            completed_tasks = self.queue.update()
            for taskid in completed_tasks:
                self.hostpool.releaseNodesByTask(taskid)

            next_task = self.next_command()

            if next_task is not None:
                debug_msg(f"enqueueing new task <{next_task}>", prefix="job")
                self.queue.enqueue(next_task)
                time.sleep(self.delay)
            else:
                if len(self.queue.running)==0:
                    break

        self.running_time = time.time()-start_time

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


def IbrunLauncher(commandfile,
        debug: str=None,
        workdir: str=None,
        cores: int=4,
        **kwargs):
    """A LauncherJob for a file of small MPI jobs.

    The following values are specified for your convenience:

    * hostpool : based on HostListByName
    * taskgenerator : based on the ``commandfile`` argument

    :param commandfile: name of file with commandlines (required)
    :param cores: number of cores (keyword, optional, default=4, see ``FileCommandlineGenerator`` for more explanation)
    :param workdir: directory for output and temporary files (optional, keyword, default uses the job number); the launcher refuses to reuse an already existing directory
    :param debug: debug types string (optional, keyword)
    """
    set_debug_flags(debug)

    job = LauncherJob(commandfile)

    job.run()

    print(job.final_report())
