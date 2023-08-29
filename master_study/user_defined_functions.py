import numpy as np
import json


def generate_run_sh(node, generation_number):
    python_command = node.root.parameters["generations"][generation_number]["job_executable"]
    return (
        f"#!/bin/bash\n"
        + f"source {node.root.parameters['setup_env_script']}\n"
        + f"cd {node.get_abs_path()}\n"
        + f"python {python_command} > output.txt 2> error.txt\n"
        + f"rm -rf final_* modules optics_repository optics_toolkit tools tracking_tools temp"
        f" mad_collider.log __pycache__ twiss* errors fc* optics_orbit_at*\n"
    )


def _compute_LR_per_bunch(
    _array_b1, _array_b2, _B1_bunches_index, _B2_bunches_index, numberOfLRToConsider, beam="beam_1"
):
    # Reverse beam order if needed
    if beam == "beam_1":
        factor = 1
    elif beam == "beam_2":
        _array_b1, _array_b2 = _array_b2, _array_b1
        _B1_bunches_index, _B2_bunches_index = _B2_bunches_index, _B1_bunches_index
        factor = -1
    else:
        raise ValueError("beam must be either 'beam_1' or 'beam_2'")

    B2_bunches = np.array(_array_b2) == 1.0

    # Define number of LR to consider
    if isinstance(numberOfLRToConsider, int):
        numberOfLRToConsider = [numberOfLRToConsider, numberOfLRToConsider, numberOfLRToConsider]

    l_long_range_per_bunch = []
    for n in _B1_bunches_index:
        # First check for collisions in ALICE

        # Formula for head on collision in ALICE is
        # (n + 891) mod 3564 = m
        # where n is number of bunch in B1, and m is number of bunch in B2

        # Formula for head on collision in ATLAS/CMS is
        # n = m
        # where n is number of bunch in B1, and m is number of bunch in B2

        # Formula for head on collision in LHCb is
        # (n + 2670) mod 3564 = m
        # where n is number of bunch in B1, and m is number of bunch in B2

        colide_factor_list = [891, 0, 2670]
        number_of_bunches = 3564

        # i == 0 for ALICE
        # i == 1 for ATLAS and CMS
        # i == 2 for LHCB
        num_of_long_range = 0
        l_HO = [False, False, False]
        for i in range(0, 3):
            collide_factor = colide_factor_list[i]
            m = (n + factor * collide_factor) % number_of_bunches

            # if this bunch is true, then there is head on collision
            l_HO[i] = B2_bunches[m]

            ## Check if beam 2 has bunches in range  m - numberOfLRToConsider to m + numberOfLRToConsider
            ## Also have to check if bunches wrap around from 3563 to 0 or vice versa

            bunches_ineraction_temp = np.array([])
            positions = np.array([])

            first_to_consider = m - numberOfLRToConsider[i]
            last_to_consider = m + numberOfLRToConsider[i] + 1

            numb_of_long_range = 0

            if first_to_consider < 0:
                bunches_ineraction_partial = np.flatnonzero(
                    _array_b2[(number_of_bunches + first_to_consider) : (number_of_bunches)]
                )

                # This represents the relative position to the head-on bunch
                positions = np.append(positions, first_to_consider + bunches_ineraction_partial)

                # Set this varibale so later the normal syntax wihtout the wrap around checking can be used
                first_to_consider = 0

            if last_to_consider > number_of_bunches:
                bunches_ineraction_partial = np.flatnonzero(
                    _array_b2[0 : last_to_consider - number_of_bunches]
                )

                # This represents the relative position to the head-on bunch
                positions = np.append(positions, number_of_bunches - m + bunches_ineraction_partial)

                last_to_consider = number_of_bunches

            bunches_ineraction_partial = np.append(
                bunches_ineraction_temp,
                np.flatnonzero(_array_b2[first_to_consider:last_to_consider]),
            )

            # This represents the relative position to the head-on bunch
            positions = np.append(positions, bunches_ineraction_partial - (m - first_to_consider))

            # Substract head on collision from number of secondary collisions
            num_of_long_range_curren_ip = len(positions) - _array_b2[m]

            # Add to total number of long range collisions
            num_of_long_range += num_of_long_range_curren_ip

        # If a head-on collision is missing, discard the bunch by setting LR to 0
        if False in l_HO:
            num_of_long_range = 0

        # Add to list of long range collisions per bunch
        l_long_range_per_bunch.append(num_of_long_range)
    return l_long_range_per_bunch


def get_worst_bunch(filling_scheme_path, numberOfLRToConsider=26, beam="beam_1"):
    """
    # Adapted from https://github.com/PyCOMPLETE/FillingPatterns/blob/5f28d1a99e9a2ef7cc5c171d0cab6679946309e8/fillingpatterns/bbFunctions.py#L233
    Given a filling scheme, containing two arrays of booleans representing the trains of bunches for
    the two beams, this function returns the worst bunch for each beam, according to their collision
    schedule.
    """

    # Load the filling scheme directly if json
    if filling_scheme_path.endswith(".json"):
        with open(filling_scheme_path, "r") as fid:
            filling_scheme = json.load(fid)

    # Extract booleans beam arrays
    array_b1 = np.array(filling_scheme["beam1"])
    array_b2 = np.array(filling_scheme["beam2"])

    # Get bunches index
    B1_bunches_index = np.flatnonzero(array_b1)
    B2_bunches_index = np.flatnonzero(array_b2)

    # Compute the number of long range collisions per bunch
    l_long_range_per_bunch = _compute_LR_per_bunch(
        array_b1, array_b2, B1_bunches_index, B2_bunches_index, numberOfLRToConsider, beam=beam
    )

    # Get the worst bunch for both beams
    if beam == "beam_1":
        worst_bunch = B1_bunches_index[np.argmax(l_long_range_per_bunch)]
    elif beam == "beam_2":
        worst_bunch = B2_bunches_index[np.argmax(l_long_range_per_bunch)]

    # Need to explicitly convert to int for json serialization
    return int(worst_bunch)


def reformat_filling_scheme_from_lpc(filling_scheme_path, fill_number=None):
    """
    Adapted from a function provided by Matteo Ruffolo, matteo.rufolo@cern.ch
    This function converts a .json file downloaded from the url link of LPC to the appropriate
    format for xmask. If a fill number is not provided, it is assumed that the first fill in the
    json file is the one of interest.
    When computing the filling scheme in case of a hybrid scheme, the following hypotheses are done:
    - There must be only one PS batch composed by 8b4e at the beginning of every SPS batch
    - After that 8b4e PS batch, all the remaining PS batches inside that SPS batch must be BCMS composed by 36 bunches
    - All the SPS batches composed by more than one PS batch have to respect the rules above
    """

    # Load the filling scheme directly if json
    with open(filling_scheme_path, "r") as fid:
        data = json.load(fid)

    # If the fill number has not been provided, take the first one
    if fill_number is None:
        fill_number = list(data["fills"].keys())[0]

    # Do the conversion (Matteo's code)
    string = ""
    B1 = np.zeros(3564)
    B2 = np.zeros(3564)
    if data["fills"][f"{fill_number}"]["name"][0:1000].split("_")[7] == "hybrid":
        n_injection = int(
            data["fills"][f"{fill_number}"]["name"][0:1000].split("_")[6].split("inj")[0]
        )
        beam = np.fromstring(
            string.join(
                string.join(
                    data["fills"][f"{fill_number}"]["csv"].split("\t")[
                        3 : n_injection * 2 * 10 : 10
                    ]
                ).split("ring_")[0 : n_injection * 2 + 1]
            ),
            dtype=int,
            sep=",",
        )

        n_bunches = np.fromstring(
            string.join(
                data["fills"][f"{fill_number}"]["csv"].split("\t")[8 : n_injection * 2 * 10 : 10]
            ),
            dtype=int,
            sep=",",
        )

        initial = np.fromstring(
            string.join(
                data["fills"][f"{fill_number}"]["csv"].split("\t")[4 : n_injection * 2 * 10 : 10]
            ),
            dtype=int,
            sep=",",
        )

        n_batches = [
            int(ii.split("\n")[0])
            for ii in data["fills"][f"{fill_number}"]["csv"].split("\t")[
                11 : (n_injection) * 2 * 10 : 10
            ]
        ]
        n_batches = np.append(n_batches, max(n_batches))
        initial = [int(ii) for ii in (initial - 1) / 10]
        for i in np.arange(n_injection * 2):
            if n_batches[i] > 1:
                counter = 1
                if beam[i] == 1:
                    for k in np.arange(n_bunches[i] / 8):
                        init_batch = int(initial[i] + (k * 8 + k * 4))
                        # print(init_batch)
                        B1[init_batch : init_batch + 8] = np.ones(8)
                    for j in np.arange(n_batches[i] - 1):
                        init_batch = (
                            initial[i] + (counter - 1) * (36 + 7) + (n_bunches[i] + 6 * 4 + 7)
                        )
                        # print(init_batch)
                        B1[init_batch : init_batch + 36] = np.ones(36)
                        counter += 1
                else:
                    for k in np.arange(n_bunches[i] / 8):
                        init_batch = int(initial[i] + (k * 8 + k * 4))
                        B2[init_batch : init_batch + 8] = np.ones(8)
                    for j in np.arange(n_batches[i] - 1) + 1:
                        init_batch = (
                            initial[i] + (counter - 1) * (36 + 7) + (n_bunches[i] + 6 * 4 + 7)
                        )
                        B2[init_batch : init_batch + 36] = np.ones(36)
                        counter += 1
            else:
                counter = 0
                if beam[i] == 1:
                    for j in np.arange(n_batches[i]):
                        init_batch = initial[i] + counter * (n_bunches[i] + 7)
                        B1[init_batch : init_batch + n_bunches[i]] = np.ones(n_bunches[i])
                        counter += 1
                else:
                    for j in np.arange(n_batches[i]):
                        init_batch = initial[i] + counter * (n_bunches[i] + 7)
                        B2[init_batch : init_batch + n_bunches[i]] = np.ones(n_bunches[i])
                        counter += 1
    else:
        n_injection = int(
            data["fills"][f"{fill_number}"]["name"][0:1000].split("_")[6].split("inj")[0]
        )
        beam = np.fromstring(
            string.join(
                string.join(
                    data["fills"][f"{fill_number}"]["csv"].split("\t")[
                        3 : n_injection * 2 * 10 : 10
                    ]
                ).split("ring_")[0 : n_injection * 2 + 1]
            ),
            dtype=int,
            sep=",",
        )

        n_bunches = np.fromstring(
            string.join(
                data["fills"][f"{fill_number}"]["csv"].split("\t")[8 : n_injection * 2 * 10 : 10]
            ),
            dtype=int,
            sep=",",
        )

        initial = np.fromstring(
            string.join(
                data["fills"][f"{fill_number}"]["csv"].split("\t")[4 : n_injection * 2 * 10 : 10]
            ),
            dtype=int,
            sep=",",
        )

        n_batches = [
            int(ii.split("\n")[0])
            for ii in data["fills"][f"{fill_number}"]["csv"].split("\t")[
                11 : (n_injection) * 2 * 10 : 10
            ]
        ]
        n_batches = np.append(n_batches, max(n_batches))
        initial = [int(ii) for ii in (initial - 1) / 10]
        for i in np.arange(n_injection * 2):
            counter = 0
            if beam[i] == 1:
                for j in np.arange(n_batches[i]):
                    init_batch = initial[i] + counter * (n_bunches[i] + 7)
                    B1[init_batch : init_batch + n_bunches[i]] = np.ones(n_bunches[i])
                    counter += 1
            else:
                for j in np.arange(n_batches[i]):
                    init_batch = initial[i] + counter * (n_bunches[i] + 7)
                    B2[init_batch : init_batch + n_bunches[i]] = np.ones(n_bunches[i])
                    counter += 1
    data_json = {"beam1": [int(ii) for ii in B1], "beam2": [int(ii) for ii in B2]}

    with open(filling_scheme_path.split(".json")[0] + "_converted.json", "w") as file_bool:
        json.dump(data_json, file_bool)
    return B1, B2


def reformat_filling_scheme_from_lpc_alt(filling_scheme_path):
    """
    Alternative to the function above, as sometimes the injection information is not present in the
    file. Not optimized but works.
    """

    # Load the filling scheme directly if json
    with open(filling_scheme_path, "r") as fid:
        data = json.load(fid)

    # Take the first fill number
    fill_number = list(data["fills"].keys())[0]

    # Do the conversion (Matteo's code)
    string = ""
    B1 = np.zeros(3564)
    B2 = np.zeros(3564)
    l_lines = data["fills"][f"{fill_number}"]["csv"].split("\n")
    for idx, line in enumerate(l_lines):
        # First time one encounters a line with 'Slot' in it, start indexing
        if "Slot" in line:
            # B1 is initially empty
            if np.sum(B1) == 0:
                for idx_2, line_2 in enumerate(l_lines[idx + 1 :]):
                    l_line = line_2.split(",")
                    if len(l_line) > 1:
                        slot = l_line[1]
                        B1[int(slot)] = 1
                    else:
                        break

            # Same with B2
            elif np.sum(B2) == 0:
                for idx_2, line_2 in enumerate(l_lines[idx + 1 :]):
                    l_line = line_2.split(",")
                    if len(l_line) > 1:
                        slot = l_line[1]
                        B2[int(slot)] = 1
                    else:
                        break
            else:
                break

    data_json = {"beam1": [int(ii) for ii in B1], "beam2": [int(ii) for ii in B2]}

    with open(filling_scheme_path.split(".json")[0] + "_converted.json", "w") as file_bool:
        json.dump(data_json, file_bool)
    return B1, B2


if __name__ == "__main__":
    # get_worst_bunch(
    #     "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/master_jobs/filling_scheme/8b4e_1972b_1960_1178_1886_224bpi_12inj_800ns_bs200ns.json"
    # )
    reformat_filling_scheme_from_lpc(
        "/afs/cern.ch/work/c/cdroin/private/example_DA_study/master_study/master_jobs/filling_scheme/25ns_2374b_2361_1730_1773_236bpi_13inj_hybrid_2INDIV.json"
    )
