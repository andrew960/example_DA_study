# Dynamics aperture study boilerplate

This repository contains a boilerplate that allows users to compute the dynamics aperture of a collider
under different parametric scenarios.

Jobs can be efficiently stored and parallelized using the
[Tree Maker](https://github.com/xsuite/tree_maker) package, while collider generation and particle tracking harness the power of [XSuite](https://github.com/xsuite/xsuite).

ℹ️ If you do not need to do parametric scans, this repository is probably not what you're looking for.

## Installation instructions

The simplest way to start is to clone the repository and install the dependencies using conda:

```bash
git clone git@github.com:xsuite/example_DA_study.git
```

Then:

```bash
cd example_DA_study
source make_miniforge.sh
```

This should install conda along with the required python modules. If something goes wrong, you can execute the commands in the ```make_miniforge.sh``` script manually, one line after the other. Git may trigger an error after ```git submodule init```, in which case you can mark the directory as safe using the command suggested by git, and enter ```git submodule init``` and ```git submodule update``` again. If this still doesn't work, you can try to manually get into ```modules/xmask/xmask/lhc``` , manually remove ```lhcerrors``` with ```rm -rf lhcerrors``` (which is potentially empty), and finally git clone ```https://github.com/lhcopt/lhcerrors.git```.

## Running a simple parameter scan simulation

This section introduces the basic steps to run a simple parameter scan simulation. The simulation consists in tracking a set of particles for a given number of turns, and computing the dynamics aperture for each particle. To get a more refined understanding of what the scripts used below are actually doing, please check the section [What happens under the hood](#what-happens-under-the-hood).

### Setting up the study

You can select the range of parameters you want to scan by editing the ```master_study/001_make_folders.py``` script, under the section ```Machine parameters being scanned```. For example, you can edit the following lines to do a tune scan of your liking (here, only the first 6 tunes are selected, in order not to create too many jobs):

```python
array_qx = np.round(np.arange(62.305, 62.330, 0.001), decimals=4)[:6]
array_qy = np.round(np.arange(60.305, 60.330, 0.001), decimals=4)[:6]
```

Note that, if the parameter ```only_keep_upper_triangle``` is set to True, most of the jobs in the grid defined above will be automatically skipped as the corresponding working points are too close to resonance, or are unreachable in the LHC.

In addition, since this is a toy simulation, you also want to keep a low number of turns simulated (e.g. 200 instead of 1000000):

```python
n_turns = 200
```

you can give the study you're doing the name of your choice by editing the following line:

```python
study_name = "example_HL_tunescan"
```

### Building the tree

You are now ready to create the folder structure (the _tree_) of your study. The tree structure can be checked in ```master_study/config.yaml```. As you can see, there are only 2 generations here :

- the first generation generates the particles distribution and build a collider with just the optics (which we call "base collider" from now on).
- the second generation sets all the other collider parameters, including the ones that are being scanned, and tracks the particles for a given number of turns.

For now, you might want to keep the jobs running on your local machine to ensure everything runs fine. To do so, ensure that in the file ```master_study/config.yaml```, there is, for both generations:

```yaml
run_on: 'local_pc'
```

If not already done, activate the conda environment:

```bash
source miniforge/bin/activate
```

Now, move to the master_study folder, and run to script to build the tree and write it on disk:

```bash
cd master_study
python 001_make_folders.py
```

This should create a folder named after ```study_name``` in ```master_study/scans```. This folder contains the tree structure of your study: the parent generation is in the subfolder ```base_collider```, while the subsequent children are in the ```xtrack_iiii```. The tree_maker ```.json``` and ```.log``` files are used by tree_maker to keep track of the jobs that have been run and the ones that are still to be run.

Each node of each generation contains a ```config.yaml``` file that contains the parameters used to run the corresponding job (e.g. the particle distributions parameters or the collider crossing-angle for the first generation, and, e.g. the tunes and number of turns simulated for the second generation).

You should be able to run each job individually by executing the following command in the corresponding folder:

```bash
source run.sh
```

Note that, to run without errors, children nodes will most likley need the files output by the parent nodes. Therefore, you should run the parent nodes first, and then the children nodes. However, this is all done automatically by the ```master_study/002_chronjob.py``` script (cf. upcoming section), such that running manually the ```run.sh``` should be done only for debugging purposes.

### Running the jobs

First, update the study name in ```master_study/002_chronjob.py```. You can now execute the script:

```bash
python 002_chronjob.py
```

Here, this will run the first generation (```base_collider```), which consists of only one job (building the particles distribution and the base collider).

In a general way, once the script is finished running, executing it again will check that the jobs have been run successfully, and re-run the ones that failed. If no jobs have failed, it will run the jobs from the next generation. Therefore, executing it again should launch all the tracking jobs (several for each tune, as the particle distribution is split in several files).

⚠️ **If the generation of the simulation you're launching comprises many jobs, ensure that you're not running them on your local machine (i.e. you don't use ```run_on: 'local_pc'``` in ```master_study/config.yaml```). Otherwise, as the jobs are run in parallel, you will most likely saturate the cores and/or the RAM of your machine.**

### Analyzing the results

Once all jobs of all generations have been computed, the results from the simulations can be gathered in a single dataframe by running the ```master_study/003_postprocessing.py``` script. First, make sure to update the study name in the script. Then, ensure that the jobs will be grouped by the variable that have been scanned (here, the tunes) by editing the following line:

```python
groupby = ["qx", "qy"]
```

Finally, run the script:

```bash
python 003_postprocessing.py
```

This should output a parquet dataframe in ```master_study/scans/study_name/```. This dataframe contains the results of the simulations (e.g. dynamics aperture for each tune), and can be used for further analysis. Note that, in the toy example above, since we simulate for a very small number of turns, the resulting dataframe will be empty as no particles will be lost during the simulation.

## What happens under the hood

The aim of this set of scripts is to run sets of simulations in a fast and automated way, while keeping the possibility to run each simulation individually.

### Tree structure

Since simulations all make use of the same base collider, the base collider only needs to be built once, as the optics is never "scanned". However, each simulation corresponds to a different set of parameters, meaning that the base collider needs to be tailored ("tuned") to each simulation. Therefore, the base collider will correspond to generation 1, and the subsequent tracking simulations with different parameters will correspond to generation 2.

This is described in the file ```master_study/config.yaml```:

```yaml
'root':
  setup_env_script: 'none'
  generations:
    1: # Build the particle distribution and base collider
      job_folder: '../../master_jobs/1_build_distr_and_collider'
      job_executable: 1_build_distr_and_collider.py # has to be a python file
      files_to_clone: # relative to the template folder
        - gen_config_orbit_correction.py
        - optics_specific_tools_hlhc15.py
      run_on: 'local_pc'
    2: # Launch the pymask and prepare the colliders
      job_folder: '../../master_jobs/2_configure_and_track'
      job_executable: 2_configure_and_track.py # has to be a python file
      run_on: 'htc' #'local_pc' #'htc' #'local_pc' 
      htc_job_flavor: 'tomorrow' # optional parameter to define job flavor
  # Children will be added below in the script 001_make_folders.py
  children:
```

This file defines the structure of the tree of jobs, which python files must be called at each generation, with which parameters, with which python distribution, and on which machine/cluster. More precisely:

- ```setup_env_script``` is the path to the conda environment that will be used to run Python in the simulations. By default, is set to ```none```, but is updated by the ```001_make_folders.py``` script to point to the conda environment in the ```master_study``` folder.
- ```job_folder```: for each generation, this describes the folder containing the files used to run the simulation. There should be at least a python script, and a ```config.yaml``` file. The python script then reads the parameters of the currunt simulation in the ```config.yaml file```, and runs the simulation accordingly.
- ```job_executable```, this is the name of the python script that will be run at each generation.
- ```files_to_clone```: this is a list of files that will be copied from the ```job_folder``` to the simulation folder. This is useful to copy files that are common to all simulations.
- ```run_on```: this is the machine/cluster on which the simulations will be run. At the moment, the following options are available:
  - ```local_pc```: the simulations will be run on the local machine. This is useful when running small number of jobs, or debugging purposes.
  - ```htc```: the simulations will be run on the HTCondor cluster at CERN. This is useful to run large sets of simulations.
  - ```slurm```: the simulations will be run on the Slurm cluster at CNAF.INFN. This is useful to run large sets of simulations.
  See section [Using computing clusters](#using-computing-clusters) below for more details.
- ```htc_job_flavor```: this is an optional parameter that can be used to define the job flavor on HTCondor. Long jobs (>8h, <24h) should most likely use ```tomorrow```. See all flavours [here](https://batchdocs.web.cern.ch/local/submit.html).
- ```children```: this is a list of children for each generation. More precisely, this contains the set of parameters used by each job of each generation. This is generated by the ```001_make_folders.py``` script, and should not be modified manually.

To get a better idea of how this file is used, you can check the json mutated version at the root of each scan (i.e. ```master_study/scans/study_name/tree_maker_study_name.json```).

Here, only two generations are used, but it is possible to add more generations if needed. For instance, if one wants to run intricated grid searches, e.g. check the dynamics aperture for each tune and each crossing angle, one could build the tree such that the tunes are scanned at generation 2, and the crossing angles at generation 3. However, this is not implemented yet, you will have to modify the scripts yourself.

### Overall pipeline

When doing a parameter scan, the following steps are performed:

1. Running the ```001_make_folders.py``` script. This creates the folder structure for the simulations, and generate the ```config.yaml``` files for each simulation. It also generates the ```tree_maker_study_name.json``` file at each generation of the scan. In this file, the following parameters are set:
    - the base parameters for the collider, that is, the parameters of the collider that might be changed from one study to the other (e.g. optics being used), but that will be the same for all simulations of the study (parameters being scanned excluded).
    ⚠️ **It is possible that you need to update other collider parameters (e.g. ```on_a5```). In this case, you can either update directly the master configuration file in ```master_study/master_jobs/1_build_distr_and_collider/config.yaml```, or adapt the ```001_make_folders.py``` script to update the collider parameters you need.**
    - the parameters for the initial particles distribution. One parameter that is important here is ```n_split```, as it sets how much a given working point will be split into different simulations, each containing a subset of the inital particles distribution. That is, ```n_split``` is actually responsible to a large extent for the parallelization of the simulations.
  
    All these parameters are added to the root of the main configuration file (```master_study/config.yaml```). The tree_maker package then takes care of providing the right set of parameters to the right python file for each generation. In practice, the master jobs (located in ```master_study/master_jobs```) are copied to the simulation folders, and the corresponding ```config.yaml``` (e.g. ```master_study/master_jobs/1_build_distr_and_collider/config.yaml```) file is adapted (mutated) for each generation and each simulation, according to the main tree_maker configuration file, which as been generated at the same time as the simulation folders (e.g. in ```master_study/scans/study_name/tree_maker_study_name.json```).
2. Running the ```002_run_simulations.py``` script. This script will run the simulations in parallel, and output a file (e.g. a collider json file, or a dataframe containing the result of the tracking) for each simulation. In practice, it calls each ```run.sh``` script in each simulation folder, which in turn calls the python script defined in the ```job_executable``` parameter of the ```master_study/config.yaml``` file. The python script makes use of the proper set of parameters, set in the mutated ```config.yaml``` files (one per job, e.g. ```master_study/scans/study_name/base_collider/xtrack_0001/config.yaml```).
3. Running the ```003_analyse_simulations.py``` script. This script will analyse the results of the simulations, and output a summary dataframe at the root of the study.

### More information

More information, although outdated, can be gathered from the explanations provided in previous versions of this repository, e.g. [in the previous README](https://github.com/xsuite/example_DA_study/blob/release/v0.1.1/README.md) and [the Tree Maker tutorial](https://github.com/xsuite/example_DA_study/blob/release/v0.1.1/tree_tutorial.md).

The code is now well formatted and well commented, such that any question should be relatively easily answered by looking at the code itself. If you have any question, do not hesitate to open an issue.

## Parameters that can be scanned

At the moment, all the collider parameters can be scanned without requiring extensive scripts modifications. This includes (but is not limited to):

- intensity (```num_particles_per_bunch```)
- crossing-angle (```on_x1, on_x5```)
- tune (```qx, qy```)
- chromaticity (```dqx, dqy```)
- octupole current (```i_oct_b1, i_oct_b2```)
- bunch being tracked (```i_bunch_b1, i_bunch_b2```)

At generation 1, the base collider is built with a default set of parameters for the optics (which are explicitely set in ```001_make_folder.py```). At generation 2, the base collider is tailored to the parameters being scanned. That is,
 the tune and chroma are matched, the luminosity leveling is computed (if leveling is required), and the beam-beam lenses
 are configured.

It should be relatively easy to accomodate the scripts for other parameters. In addition, to prevent any complication, only the simulation of beam 1 is possible, but this should also be relatively easy to adapt.

## Using computing clusters

### General procedure

The scripts in the repository allows for an easy deployment of the simulations on HTCondor (CERN cluster) and Slurm (CNAF.INFN cluster). Please consult the corresponding tutorials ([here](https://abpcomputing.web.cern.ch/guides/htcondor/), and [here](https://abpcomputing.web.cern.ch/computing_resources/hpc_cnaf/)) to set up the clusters on your machine.

Once, this is done, jobs can be executed on HTCondor by setting ```run_on: 'htc'``` instead of ```run_on: 'local_pc'``` in ```master_study/config.yaml```. Similarly, jobs can be executed on the CNAF cluster by setting ```run_on: 'slurm'```.

⚠️ **Be careful of not running the ```master_study/002_chronjob.py``` script several times, as this will submit the same jobs several times.** In the future, this will hopefully be fixed by adding a check in the script to see if the jobs have already been submitted.

### Using Docker images

For reproducibility purposes and/or limiting the load on AFS or EOS drive, one can use Docker images to run the simulations. A registry of Docker images is available at "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/", and some ready-to-use for DA simulations Docker images are available at ""/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cdroin/da-study-docker" (this is the default directory for images in the ```002_chronjob.py``` file). To learn more about building Docker images and hosting them on the CERN registry, please consult the [corresponding tutorial](https://abpcomputing.web.cern.ch/guides/docker_on_htcondor/) abd the [corresponding repository](https://gitlab.cern.ch/unpacked/sync).

#### HTCondor
When running simulations on HTCondor, Docker images are be pulled directly from CVMFS through the submit file. No additional configuration is required, except for setting ```run_on: 'htc_docker'``` in ```master_study/config.yaml```.

#### Slurm

Things are a bit tricker with Slurm, as the Docker image must first be manually pulled from CVMFS, and then loaded on the node after Singularity-ize it. The pulling of the image is only needed the first time, and can be done with e.g. (for the image ```cdroin/da-study-docker```):
  
  ```bash
  pull docker://gitlab-registry.cern.ch/cdroin/da-study-docker
  ```

However, due to unknown reason, only some nodes of INFN-CNAF will correctly execute this command. For instance, it didn't work on the default CPU CERN node (```hpc-201-11-01-a```), but it did on an alternative one (```hpc-201-11-02-a```). We recommand using either ```hpc-201-11-02-a``` or a GPU node (e.g. ```hpc-201-11-35```) to pull the image. Once the image is pulled, it will be accessible from any node. 

For testing purposes, one can then run the image with Singularity directly on the node (not required):
  
  ```bash
  singularity run da-study-docker_latest.sif
  ```

In practice, the ```002_chronjob.py``` script will take care of mouting the image and running it on the node with the correct miniforge distribution. All you have to do is change the ```run_on``` parameter in ```master_study/config.yaml``` to ```run_on: 'slurm_docker'```.

⚠️ **The slurm docker scripts are a still experimental**. For instance, simulations seem to run more slowly when containerized. In addition, you may need to adapt the ```002_chronjob.py``` script and the ```make_miniforge.sh``` script to your needs. For instance, at the time of writing this documentation, symlinks path in the front-end node of INFN-CNAF are currently broken, meaning that some temporary fixs are implemented. This will hopefully be fixed in the future, and the fixs should not prevent the scripts from running anyway.
