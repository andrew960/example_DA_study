# Some simple steps

### Installation instructions

Do 
```
git clone git@github.com:andrew960/example_DA_study.git -b release/v0.1.1
```

then 

```
cd example_DA_study
source make_miniconda.sh
```

This will install the needed python distributions.

### OLD


```bash 
# install miniconda, then:
pip install xsuite
pip install cpymad
git clone https://github.com/xsuite/tree_maker.git
pip install -e tree_maker/
git clone https://github.com/lhcopt/lhcmask.git
pip install -e lhcmask/
git clone https://github.com/lhcopt/lhctoolkit.git
git clone https://github.com/lhcopt/lhcerrors.git
```

### Connecting to the cnaf.infn.it
Go to the the hpc-201-11-01-a machine.
I do it by 
```bash
ssh bologna
```

but you need to configure your `~/.ssh/config` by adding
```bash
Host bastion
 HostName bastion.cnaf.infn.it
 User sterbini
 ForwardX11 yes

Host bologna
 ProxyCommand ssh -q bastion nc hpc-201-11-01-a 22
 ForwardX11 yes
```

As you can see, `bologna` (hpc-201-11-01-a) host is passing via the `bastion` (bastion.cnaf.infn.it) connection.

### Activate the environment
I sugget sto open a `tmux` terminal and 

```bash
source /home/HPC/sterbini/py38/bin/activate
```
then move where you want to start lauch this DA study and make a clone of this repository

```
git clone https://github.com/sterbini/DA_study_example.git
cd DA_study_example
```

### Tree creation
With the command
```bash
python 001_make_folders.py
```
one creates the `study_000` tree in the `DA_study_example`.
It consists of a 21x21 folders (a tune scan using the pymask in `000_machine_model`).
Each folders has 15 subfolders (each is launching an `xtrack` jobs of 42 or 43 particles, in total the particles are 640).

All the details are in the code `001_make_folders.py`.

### Launching the simulation

One can launch the first generation of of jobs (441 pymasks) by 
```
python 002_chronjob.py
```
And you can repeat the same command to advance in the tree genealogy and launch the tracking (the code knows when is ready to launch the second generation).
In fact this could be implemented in a chron job.

One can monitor the status of the jobs by
```
bjobs -sum -u user_name 
```












