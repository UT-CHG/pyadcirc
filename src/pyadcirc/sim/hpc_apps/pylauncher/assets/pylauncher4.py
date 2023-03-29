import argparse as ap
import copy
import json
import logging
import os
import pdb
import re
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _expand_int_list(s):
    """Expands int lists with ranges."""
    r = []
    for i in s.split(","):
        if "-" not in i:
            r.append(int(i))
        else:
            l, h = map(int, i.split("-"))
            r += range(l, h + 1)
    return r


def _compact_int_list(i, delim=","):
    """Compacts int lists with ranges."""
    if len(i) == 0:
        return ""
    elif len(i) == 1:
        return f"{i[0]}"
    if i[1] != (i[0] + 1):
        return f"{i[0]}{delim}{_compact_int_list(i[1:])}"
    else:
        for e in range(1, len(i)):
            if i[e] != i[0] + e:
                return f"{i[0]}-{i[e-1]}{delim}{_compact_int_list(i[e:])}"
        return f"{i[0]}-{i[-1]}"


class SLURMTaskException(Exception):
    """A very basic exception mechanism"""

    def __init__(self, str):
        print(str)
        self.str = str

    def __str__(self):
        return self.str


class SLURMTaskSlot:
    """
    Combination of (host, idx) that can run a SLURM task. A slot can have
    a task associated with it or be free. `host` corresponds to the name of the
    host that can execute the task, as accessible in the SLURM environment
    variable SLURM_JOB_NODELIST or, locally from the host, in SLURM_NODENAME.
    `idx` corresponds to the index in the total task list available to the
    SLURM job that the slot corresponds to. For example if a job is being run
    with 3 nodes and 5 total tasks (`-N 3 -n 5`), then the SLURM execution
    environment will look something like:

    .. code-block:: bash
        SLURM_JOB_NODELIST=c303-[005-006],c304-005
        SLURM_TASKS_PER_NODE=2(x2),1

    In this scenario, host `cs303-005` would have slot idxs 1 and
    2, `cs303-006` would have slots 3 and 4 associated with it, and host
    `cs304-005` would have only slot 5 associated with it. Note that these task
    slots do not corespond to the available CPUs per host available, which can
    vary depending on the cluster being used.

    Attributes
    ----------
    host : str
        Name of compute node on a SLURM execution system. Corresponds to a host
        listed in the environment variable SLURM_JOB_NODELIST.
    idx : int
        Index of task in total available SLURM task slots. See above for more
        details.
    free : bool
        False if slot is being occupied currently by a task, True otherwise.
    tasks : List[SLURMTask]
        List of SLURMTask objects that have been executed on this slot. If the
        slot is currently occupied, the last element in the list corresponds to
        the currently running task.
    """

    def __init__(self, host, idx):
        self.host = host
        self.idx = idx
        self.free = True
        self.tasks = []

    def occupy(self, task):
        """
        Occupy a slot with a task.

        Parameters
        ----------
        task: :class:SLURMTask
            Task object that will occupy the slot.

        Raises
        -------
        ValueError
            If trying to occupy a slot that is not currently free.

        """
        if not self.free:
            raise ValueError(f"Trying to occupy a busy node {self}")
        self.tasks.append(task)
        self.free = False

    def release(self):
        """Make slot unoccupied."""
        self.free = True

    def isfree(self):
        """Test whether slot is occupied"""
        return self.free

    def __str__(self):
        s = "FREE - " if self.free else "BUSY - "
        s += f"h:{self.host}, c:{self.idx}, tasks:{self.tasks}"
        return s


class SLURMTask:
    """
    Command to be executed in parallel using ibrun on a slot of SLURM tasks as
    designated by :class:SLURMTaskQueue. This class contains the particulars of
    a task to be executing, including the main parallel command to be executed
    in parallel using ibrun, optional pre/post process commands to be executed
    serially, and an optional directory to change to before executing the main
    parallel command. Once appropriate resources for the task have been found,
    and the execute() method is called, the class `slots` attribute will be
    filled with an `(offset, extent)` pair indicating what continuous region of
    the available task slots is being occupied by the currently running task.
    The command to be executed is then wrapped into a script file that is stored
    in `workdir` and a :class:subprocess.Popen object is opened to execute the
    script.

    Note that the :class:SLURMTaskQueue handles the initialization and
    management of task objects, and in general a user has no need to initialize
    task objects individually.

    Attributes
    ----------
    task_id : int
        Unique task ID to assign to this SLURM Task.
    cmnd : str
        Main command to be wrapped in `ibrun` with the appropriate offset/extent
        parameters for parallel execution.
    cores : int
        Number of cores, which correspond to SLURM job task numbers, to use for
        the job.
    pre_proccess : str
        Command to be executed in serial before the main parallel command.
    post_proccess : str
        Command to be executed in serial after the main parallel command.
    cdir: str
        directory to change to before executing the main parallel command.
    workdir : str
        directory to store execution script, along with output and error files.
    execfile : str
        Path to shell script containing wrapped command to be run by the
        subprocess that is spawned to executed the SLURM task. Note this file
        won't exist until the task is executed.
    logfile : str
        Path to file where stdout of the SLURM task will be redirected to. Note
        this file won't exist until the task is executed.
    errfile : str
        Path to file where stderr of the SLURM task will be redirected to. Note
        this file won't exist until the task is executed.
    slots : Tuple(int)
        `(offset, extent)` tuple in SLURM Task slots where task is currently
        being/was executed, or None if task has not been executed yet.
    start_ts : float
        Timestamp, in seconds since epoch, when task execution started, or None
        if task has not been executed  yet.
    end_ts : float
        Timestamp, in seconds since epoch, when task execution finished, as
        measured by first instance the process is polled using `get_rc()` with a
        non-negative response, or None if task has not finished yet.
    running_time : float
        Timestamp, in seconds since epoch, when task execution finished, as
        measured by first instance the process is polled using `get_rc()` with a
        non-negative response, or None if task has not finished yet.
    """

    def __init__(
        self,
        task_id: int,
        cmnd: str,
        cores: int = 1,
        pre_process: str = None,
        post_process: str = None,
        cdir: str = None,
        workdir: str = None,
    ):

        self.task_id = task_id
        self.command = cmnd
        self.cores = int(cores)
        self.pre_process = pre_process
        self.post_process = post_process
        self.cdir = cdir

        if workdir is None:
            workdir = Path.cwd() if workdir is None else Path(workdir)
            self.workdir = workdir / f"task_{task_id}"
        else:
            self.workdir = Path(workdir)
        self.workdir.mkdir(exist_ok=exist_ok)

        self.logfile = self.workdir / f"{task_id}-log"
        self.errfile = self.workdir / f"{task_id}-err"
        self.execfile = self.workdir / f"{task_id}-exec"

        self.slots = None
        self.start_ts = None
        self.end_ts = None
        self.running_time = None
        self.rc = None
        self.sub_proc = None

    def __getitem__(self, ind):
        return self.data.get(ind, None)

    def __str__(self):
        r = f"command=<<{self.command}>>"
        r += f", cores={self.cores}"
        if self.pre_process is not None:
            r += f", pre=<<{self.pre_process}>>"
        if self.post_process is not None:
            r += f", pre=<<{self.post_process}>>"
        if self.cdir is not None:
            r += f", cdir=<<{self.cdir}>>"
        return r

    def _wrap(self, offset, extent):
        """Take a commandline, write it to a small file, and return the
        commandline that sources that file
        """
        f = open(self.execfile, "w")
        f.write("#!/bin/bash\n\n")
        if self.pre_process is not None:
            f.write(f"{self.pre_process}\n")
            f.write("if [ $? -ne 0 ]\n")
            f.write("then\n")
            f.write("  exit 1\n")
            f.write("fi\n")
        if self.cdir is not None:
            f.write(f"cd {self.cdir}\n")
            f.write(f"cwd=$(pwd)\n")
        f.write(f"ibrun -o {offset} -n {extent} {self.command}\n")
        if self.cdir is not None:
            f.write(f"cd $cwd\n")
        f.write("if [ $? -ne 0 ]\n")
        f.write("then\n")
        f.write("  exit 1\n")
        f.write("fi\n")
        if self.post_process is not None:
            f.write(f"{self.post_process}\n")
            f.write("if [ $? -ne 0 ]\n")
            f.write("then\n")
            f.write("  exit 1\n")
            f.write("fi\n")
        f.close()
        os.chmod(
            self.execfile,
            stat.S_IXUSR
            + +stat.S_IXGRP
            + stat.S_IXOTH
            + stat.S_IWUSR
            + +stat.S_IWGRP
            + stat.S_IWOTH
            + stat.S_IRUSR
            + +stat.S_IRGRP
            + stat.S_IROTH,
        )

        new_command = f"{self.execfile} > {self.logfile} 2> {self.errfile}"

        return new_command

    def execute(self, offset, extent):
        """
        Execute a wrapped command on subprocesses given a task slot range.

        Parameters
        ----------
        offset : int
            Offset in list of total available SLURM tasks available. This will
            determine the `-o` paraemter to run ibrun with.
        extent : int
            Extent, or number of slots, to occupy in list of total available
            SLURM tasks available. This will determine the `-n` parameter to
            run ibrun with.

        """
        self.start_ts = time.time()
        self.slots = (offset, extent)
        self.sub_proc = subprocess.Popen(
            self._wrap(offset, extent), shell=True, stdout=subprocess.PIPE
        )
        logger.info(f"{self.task_id} running on process {self.sub_proc.pid}")

    def terminate(self):
        """Terminate subprocess executing task if it exists."""
        if self.sub_proc is not None:
            self.sub_proc.terminate()

    def get_rc(self):
        """Poll process to see if completed"""
        self.rc = self.sub_proc.poll()
        if self.rc is not None:
            self.end_ts = time.time()
            self.running_time = self.end_ts - self.start_ts
            logging.info(f"completed {self.task_id} in {self.running_time:5.3f}")
            return self.rc

        return -1

    def __repr__(self):
        s = f"task_id: {self.task_id}, cmnd: <<{self.command}>>, cores: {self.cores}"
        return s


class SLURMTaskQueue:
    """
    Object that does the maintains a list of Task objects.
    This is internally created inside a ``LauncherJob`` object.


    Attributes
    ----------
    task_slots : List(:class:SLURMTaskSlot)
        List of task slots available. This is parsed upon initialization from
        SLURM environment variables SLURM_JOB_NODELIST and SLURM_TASKS_PER_NODE.
    workdir : str
        Path to directory to store files for tasks executed, if the tasks
        themselves dont specify their own work directories. Defaults to a
        directory with the prefix `.pylauncher-job{SLURM_JOB_ID}-` in the
        current working directory.
    delay : float
        Number of seconds to pause between iterations of updating the queue.
        Default is 1 second. Note this affects the poll rate of tasks runing
        in the queue.
    task_max_runtime : float
        Max run time, in seconds, any individual task in the queue can run for.
    max_runtime : float
        Max run time, in seconds, for execution of `run()` to empty the queue.
    task_count : int
        Running counter, starting from 0, of total tasks that pass through the
        queue. The current count is used for the task_id of the next task added
        to the queue, so that a tasks task_id corresponds to the order in which
        it was added to the queue.
    running_time : float
        Total running time of the queue when `run()` is executed.
    queue : List(:clas:SLURMTask)
        List of :class:SLURMTasks in queue. Populated via the
        `enqueue_from_json()` method.
    running : List(:clas:SLURMTask)
        List of :class:SLURMTasks that are currently running.
    completed : List(:clas:SLURMTask)
        List of :class:SLURMTasks that are completed running successfully, in
        that the process executing them returned a 0 exit code.
    errored : List(:clas:SLURMTask)
        List of :class:SLURMTasks that failed to run successfully in that the
        processes executing them returned a non-zero exit code..
    timed_out : List(:clas:SLURMTask)
        List of :class:SLURMTasks that failed to run successfully in that the
        their runtime exceeded `task_max_runtime`.
    invalid : List(:clas:SLURMTask)
        List of :class:SLURMTasks that were not run because their configurations
        were invalid, or the amount of resources required to run them was too
        large.
    """

    def __init__(
        self,
        commandfile: str,
        workdir: str = None,
        task_max_runtime: float = 1e10,
        max_runtime: float = 1e10,
        delay: float = 1,
    ):

        # Node list - Initialize from SLURM environment
        self.task_slots = []
        self._init_task_slots()

        # Default workdir for executing tasks if task doesn't specify workdir
        self.workdir = workdir
        if self.workdir is None:
            self.workdir = Path(
                tempfile.mkdtemp(
                    prefix=f'.pylauncher-job{os.environ["SLURM_JOB_ID"]}-',
                    dir=Path.cwd(),
                )
            )
        else:
            self.workdir = Path(workdir) if type(workdir) != Path else workdir
            self.workdir.mkdir(exist_ok=True)

        # Set queue runtime constants
        self.delay = delay
        self.task_max_runtime = task_max_runtime
        self.max_runtime = maxruntime

        # Initialize Task Queue Arrays
        self.task_count = 0
        self.running_time = 0.0
        self.queue = []
        self.running = []
        self.completed = []
        self.errored = []
        self.timed_out = []
        self.invalid = []

        # Enqueue tasks from json file
        self.enqueue_from_json(commandfile)

    def __repr__(self):
        completed = sorted([t.task_id for t in self.completed])
        timed_out = sorted([t.task_id for t in self.timed_out])
        errored = sorted([t.task_id for t in self.errored])
        queued = sorted([t.task_id for t in self.queue])
        running = sorted([t.task_id for t in self.running])
        return (
            "completed: "
            + str(_compact_int_list(completed))
            + "\ntimed_out: "
            + str(_compact_int_list(timed_out))
            + "\nerrored: "
            + str(_compact_int_list(errored))
            + "\nqueued: "
            + str(_compact_int_list(queued))
            + "\nrunning: "
            + str(_compact_int_list(running))
            + "."
        )

    def _init_task_slots(self):
        """Initialize available task slots from SLURM environment variables"""
        hl = []
        host_groups = re.split(r",\s*(?![^\[\]]*\])", os.environ["SLURM_JOB_NODELIST"])
        for hg in host_groups:
            splt = hg.split("-")
            h = splt[0] if type(splt) == list else splt
            ns = "-".join(splt[1:])
            ns = ns[1:-1] if ns[0] == "[" else ns
            padding = min([len(x) for x in re.split(r"[,-]", ns)])
            hl += [f"{h}-{str(x).zfill(padding)}" for x in _expand_int_list(ns)]

        tasks_per_host = []
        for idx, tph in enumerate(os.environ["SLURM_TASKS_PER_NODE"].split(",")):
            mult_split = tph.split("(x")
            ntasks = int(mult_split[0])
            if len(mult_split) > 1:
                for i in range(int(mult_split[1][:-1])):
                    tasks_per_host.append(ntasks)
                    for j in range(ntasks):
                        self.task_slots.append(SLURMTaskSlot(hl[idx], j))
            else:
                for j in range(ntasks):
                    self.task_slots.append(SLURMTaskSlot(hl[idx], j))

    def _request_slots(self, task):
        """Request a number of slots for a task"""
        start = 0
        found = False
        cores = task.cores
        while not found:
            if start + cores > len(self.task_slots):
                return False
            for i in range(start, start + cores):
                found = self.task_slots[i].isfree()
                if not found:
                    start = i + 1
                    break

        # Execute task
        task.execute(locator[0], locator[1])

        # Mark slots as occupied with with task_id
        for n in range(start, start + cores):
            self.task_slots[n].occupy(task)

        return True

    def _release_slots(self, task_id):
        """Given a task id, release the slots that are associated with it"""
        for s in self.task_slots:
            if s.task_id == task_id:
                s.release()

    def _start_queued(self):
        """
        Start queued tasks. For all queued, try to find a continuous set of
        slots equal to the number of cores required for the task. The tasks are
        looped through in decreasing order of number of cores required. If the
        task is to big for the whole set of available slots, it is automatically
        added to the invalid list. Otherwise `_request_slots` is called to see
        if there space for the task to be run in the available slots.
        """
        # Sort queue in decreasing order of # of cores
        tqueue = copy.copy(self.queue)
        tqueue.sort(key=lambda x: -x.cores)
        for task in tqueue:
            if task.cores > len(self.task_slots):
                logger.info(f"Task {task} to large. Adding to invalid list.")
                self.queue.remove(task)
                self.invalid.append(task)
                continue
            if self._request_slots(task):
                logger.info(f"Successfully found resources for task {task}")
                self.queue.remove(task)
                self.running.append(task)
            else:
                logger.info(f"Unable to find resources for {task}.")

    def _update(self):
        """
        Update status of tasks in queue by calling polling subprocesses
        executing them with `get_rc()`. Tasks are added to the erorred or
        completed lists, or terminated and added to timed_out list if
        `task_max_runtime` is exceeded.
        """
        to_release_task_ids = []
        running = []
        for t in self.running:
            rc = t.get_rc()
            if rc == 0:
                self.completed.append(t)
                to_release_task_ids.append(t.task_id)
            elif rc > 0:
                self.errored.append(t)
                to_release_task_ids.append(t.task_id)
            else:
                rt = time.time() - t.start_ts
                if rt > self.task_max_runtime:
                    logging.info(f"Task {t} has exceeded max runtime: {rt}")
                    logging.info(f"Aborting task {t.task_id}")
                    t.terminate()
                    self.timed_out.append(t)
                else:
                    running.append(t)
        self.running = running

        # Release slots for completed tasks
        for task_id in to_release_task_ids:
            self._release_slots(task_id)

    def enqueue_from_json(self, filename, cores=1):
        """
        Add a list of tasks to the queue from a JSON file. The json file must
        contain a list of configurations, with at mininum each containing a
        `cmnd` field indicating the command to be executed in parallel using
        a corresponding number of `cores`, which defaults to the passed in value
        if not specified per task configuration.

        Parameters
        ----------
        filename : str
            Path to json files containing list of json configurations, one per
            task to add to the queue.
        cores : int
            Default number of cores to use for each task if not specified within
            task configuration.

        """
        with open(filename, "r") as fp:
            task_list = json.load(fp)

        for i, t in enumerate(task_list):
            task = SLURMTask(
                self.task_count,
                t.pop("cmnd", None),
                t.pop("cores", cores),
                t.pop("pre_process", None),
                t.pop("post_process", None),
                t.pop("cdir", None),
                t.pop("workdir", self.wordir),
            )
            self.queue.append(task)
            self.task_count += 1

    def run(self):
        """
        Runs tasks and wait for all tasks in queue to complete, or until
        `max_runtime` is exceeded.
        """
        self.start_ts = time.time()
        logging.info("Starting launcher job")
        while True:
            # Check to see if max runtime is exceeded
            elapsed = time.time() - self.start_ts
            if elapsed > self.maxruntime:
                logger.info("Exceeded max runtime")
                break

            # Start queued jobs
            self._start_queued()

            # Update queue for completed/errored jobs
            self._update()

            # Wait for a bit
            time.sleep(self.delay)

            # Check if done
            if len(self.running) == 0:
                if len(self.queue) == 0:
                    logging.info(f"Running and queue are empty.")
                    break
                else:
                    logging.info(f"Running list empty but queue is not {self.queue}")

        self.running_time = time.time() - self.start_ts


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("infile", nargs="?", default="jobs_list.json")
    args = parser.parse_args()

    # Initialize Logging
    logger = logging.getLogger("pylauncher")
    logformat = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stdout,
        format=logformat,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info(f"Initializing Task Queue from file {args.infile}")
    tq = SLURMTaskQueue(args.infile)
    logging.info(f"Running Task Queue")
    tq.run()
    logging.info(f"Done Running Tasks in Queue")
