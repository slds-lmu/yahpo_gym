from yahpo_gym.configuration import cfg
import onnxruntime as rt
import time
import json
import copy
import os

from pathlib import Path
from typing import Union, Dict, List
import numpy as np
from ConfigSpace.read_and_write import json as CS_json
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import pandas as pd


class BenchmarkSet:
    def __init__(
        self,
        scenario: str = None,
        instance: str = None,
        active_session: bool = True,
        session: Union[rt.InferenceSession, None] = None,
        multithread: bool = True,
        check: bool = True,
        noisy: bool = False,
    ):
        """
        Interface for a benchmark scenario.
        Initialized with a valid key for a valid scenario and optinally an `onnxruntime.InferenceSession`.

        Parameters
        ----------
        scenario: str
            (Required) A key for `ConfigDict` pertaining to a valid benchmark scenario (e.g. `lcbench`).
        instance: str
            (Optional) A key for `ConfigDict` pertaining to a valid instance (e.g. `3945`).
            See `BenchmarkSet(<key>).instances` for a list of available instances.
        active_session: bool
            Should the benchmark run in an active `onnxruntime.InferenceSession`? Initialized to `Trtue`.
        session: onnx.Session
            A ONNX session to use for inference. Overwrite `active_session` and sets the provided `onnxruntime.InferenceSession` as the active session.
            Initialized to `None`.
        multithread: bool
            Should the ONNX session be allowed to leverage multithreading capabilities?
            Initialized to `True` but on some HPC clusters it may be needed to set this to `False`, depending on your setup.
            Only relevant if no session is given.
        check: bool
            Should input to objective_function be checked for validity? Initialized to `True`, but can be disabled for speedups.
        noisy: bool
            Use stochastic surrogate models? Initialized to `False`.
        """

        assert scenario is not None, "Please provide a valid scenario."
        self.config = cfg(scenario)
        self.encoding = self._get_encoding()
        self.config_space = self._get_config_space()
        self.active_session = active_session
        self.noisy = noisy
        self.check = check
        self.quant = None
        self.constants = {}
        self.session = None
        self.archive = []

        if instance is not None:
            self.set_instance(instance)

        if self.active_session or (session is not None):
            self.set_session(session, multithread=multithread)

        if "citation" in self.config.config.keys():
            if self.config.config.get("citation") is not None:
                print("Please make sure to also cite:")
                print(*self.config.config.get("citation"), sep="\n")

    def objective_function(
        self,
        configuration: Union[Dict, List[Dict]],
        seed: int = None,
        logging: bool = False,
        multithread: bool = True,
    ):
        """
        Evaluate the surrogate for (a) given configuration(s).

        Parameters
        ----------
        configuration: Dict
            A valid dict or list of dicts containing hyperparameters to be evaluated.
            Attention: `configuration` is not checked for internal validity for speed purposes.
        logging: bool
            Should the evaluation be logged in the `archive`? Initialized to `False`.
        multithread: bool
            Should the ONNX session be allowed to leverage multithreading capabilities?
            Initialized to `True` but on some HPC clusters it may be needed to set this to `False`, depending on your setup.
            Only relevant if no active session has been set.
        """
        if not self.active_session or self.session is None:
            self.set_session(multithread=multithread)

        # Always work with a list of configurations
        if isinstance(configuration, dict) or isinstance(
            configuration, CS.Configuration
        ):
            configuration = [configuration]

        configuration = [
            cf.get_dictionary() if isinstance(cf, CS.Configuration) else cf
            for cf in configuration
        ]

        input_names = [x.name for x in self.session.get_inputs()]
        output_name = self.session.get_outputs()[0].name

        results_list = [None] * len(configuration)
        x_cont, x_cat = self._config_to_xs(configuration[0])
        for i in range(1, len(configuration)):
            x_cont_, x_cat_ = self._config_to_xs(configuration[i])
            x_cont = np.vstack((x_cont, x_cont_))
            x_cat = np.vstack((x_cat, x_cat_))

        # Set seed and run inference
        if seed is not None:
            rt.set_seed(seed)
        results = self.session.run(
            [output_name], {input_names[0]: x_cat, input_names[1]: x_cont}
        )[
            0
        ]  # batch predict
        for i in range(len(results)):
            results_dict = {k: v for k, v in zip(self.config.y_names, results[i])}
            if logging:
                timedate = time.strftime("%D|%H:%M:%S", time.localtime())
                self.archive.append(
                    {"time": timedate, "x": configuration[i], "y": results_dict}
                )
            results_list[i] = results_dict

        if not self.active_session:
            self.session = None

        return results_list

    def objective_function_timed(
        self,
        configuration: Union[Dict, List[Dict]],
        seed: int = None,
        logging: bool = False,
        multithread: bool = True,
    ):
        """
        Evaluate the surrogate for (a) given configuration(s) and sleep for 'self.quant' * predicted runtime(s).
        The quantity 'self.quant' is automatically inferred if it is not set manually.
        If configuration is a list of dicts, sleep is done after all evaluations.
        Note, that this assumes that the predicted runtime is in seconds.

        Parameters
        ----------
        configuration: Dict
            A valid dict or list of dicts containing hyperparameters to be evaluated.
            Attention: `configuration` is not checked for internal validity for speed purposes.
        logging: bool
            Should the evaluation be logged in the `archive`? Initialized to `False`.
        multithread: bool
            Should the ONNX session be allowed to leverage multithreading capabilities?
            Initialized to `True` but on some HPC clusters it may be needed to set this to `False`, depending on your setup.
            Only relevant if no active session has been set.
        """
        if self.quant is None:
            self.quant = self._infer_quant()

        start_time = time.time()

        # Always work with a list of results
        results = self.objective_function(
            configuration, seed=seed, logging=logging, multithread=multithread
        )
        if isinstance(results, dict):
            results = [results]

        runt = sum([result.get(self.config.runtime_name) for result in results])
        offset = time.time() - start_time
        sleepit = max(runt - offset, 0) * self.quant
        time.sleep(sleepit)
        return results

    def set_constant(self, param: str, value=None):
        """
        Set a given hyperparameter to a constant.

        Parameters
        ----------
        param: str
            A valid parameter name.
        value: int | str | any
            A valid value for the parameter `param`.
        """
        if param is not None:
            hpar = self.config_space.get_hyperparameter(param)
            if not hpar.is_legal(value):
                raise Exception(f"Value {value} not allowed for parameter {param}!")
            self.constants[param] = value

    def set_instance(self, value):
        """
        Set an instance.

        Parameters
        ----------
        value: int | str | any
            A valid value for the parameter pertaining to the configuration. See `instances`.
        """
        self.set_constant(self.config.instance_names, value)

    def get_opt_space(self, drop_fidelity_params: bool = False, seed: int = None):
        """
        Get the search space to be optimized.
        Sets 'instance' as a constant instance and removes all fidelity parameters if 'drop_fidelity_params = True'.

        Parameters
        ----------
        drop_fidelity_params: bool
            Should fidelity params be dropped from the `opt_space`? Defaults to `False`.
        seed : int
            Seed for the ConfigSpace. Optional, initialized to None.
        """
        csn = copy.deepcopy(self.config_space)
        hps = csn.get_hyperparameters()

        # Set constants (e.g. instance)
        for p, v in self.constants.items():
            param_idx = csn.get_hyperparameter_names().index(p)
            hps[param_idx] = CSH.Constant(p, v)

        # Drop fidelity parameters
        if drop_fidelity_params:
            fidelity_params_idx = [
                csn.get_hyperparameter_names().index(fidelity_param)
                for fidelity_param in self.config.fidelity_params
            ]
            fidelity_params_idx.sort()
            fidelity_params_idx.reverse()
            for idx in fidelity_params_idx:
                del hps[idx]

        # Rebuild ConfigSpace
        cnds = csn.get_conditions()
        fbds = csn.get_forbiddens()
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(hps)
        cs.add_conditions(cnds)
        cs.add_forbidden_clauses(fbds)
        return cs

    def get_fidelity_space(self, seed: int = None):
        """
        Get the fidelity space to be optimized for.

        Parameters
        ----------
        seed : int
            Seed for the ConfigSpace. Optional, initialized to None.
        """
        csn = copy.deepcopy(self.config_space)
        hps = csn.get_hyperparameters()
        fidelity_params_idx = [
            csn.get_hyperparameter_names().index(fidelity_param)
            for fidelity_param in self.config.fidelity_params
        ]
        hps = [hps[idx] for idx in fidelity_params_idx]
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters(hps)
        return cs

    def set_session(
        self, session: Union[rt.InferenceSession, None] = None, multithread: bool = True
    ):
        """
        Set the session for inference on the surrogate model.

        Parameters
        ----------
        session: onnxruntime.InferenceSession
            A ONNX session to use for inference. Overwrite `active_session` and sets the provided `onnxruntime.InferenceSession` as the active session.
            Initialized to `None`.
        multithread: bool
            Should the ONNX session be allowed to leverage multithreading capabilities?
            Initialized to `True` but on some HPC clusters it may be needed to set this to `False`, depending on your setup.
            Only relevant if no session is given.
        """
        # Either overwrite session or instantiate a new one if no active session exists
        if session is not None:
            self.session = session
        elif self.session is None:
            model_path = self._get_model_path()
            if not Path(model_path).is_file():
                raise Exception(f"ONNX file {model_path} not found!")
            options = rt.SessionOptions()
            if not multithread:
                options.inter_op_num_threads = 1
                options.intra_op_num_threads = 1
            self.session = rt.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])

    @property
    def instances(self):
        """
        A list of valid instances for the scenario.
        This usually refers to the dataset the selected ML algorithm (=`scenario`) the model
        should be fit on.
        Note: The `rbv2_*`, `lcbench`, and `iaml_*` scenarios contain instances based on OpenML datasets.
        Parameters of the `ConfigSpace` correspond to the `dataset` identifier, not the `task` identifier even though the
        `instance_name` is `OpenMLTaskId`.

        """
        if self.config.instance_names is None:
            return self.config.config["instances"]
        return [
            *self.config_space.get_hyperparameter(self.config.instance_names).choices
        ]

    @property
    def instance(self):
        """
        The selected instance. Returns none if no instance was selected.
        """
        return self.constants.get(self.config.instance_names)

    @property
    def targets(self):
        """
        A list of available targets for the scenario.
        """
        return self.config.y_names

    @property
    def target_stats(self):
        """
        Empirical minimum and maximum of the target.
        Obtained through random search with 0.5M points.

        Returns a data.frame with the following columns:
            * metric :: target
            * statistic :: min or max
            * value :: value of minimum/maximum
            * scenario :: the scenario
            " instance :: the instance
        If no instance is set, all instances for a given scenario are returned.
        """
        df = pd.read_csv(
            os.path.join(
                self.config.config["basedir"],
                "global_statistics",
                "instance_target_statistics.csv",
            )
        )
        df = df[(df.scenario == self.config.config_id)]
        if self.instance is not None:
            df = df[(df.instance == self.instance)]
        return df

    @property
    def properties(self):
        """
        List of properties of the benchmark scenario:
        Describes the type of the search space: ('continuous', 'mixed', 'categorical', 'hierarchical')
        and availability of metadata (e.g. 'memory': memory measurements are available).
        Note that the instance name or id variable is not counted as a categorical hyperparameter.
        """
        props = []

        cat = len(self.config.cat_names) > 1  # not >=1 because for all benchmarks expecept nb301 there is at least the categorical instance id
        cont = len(self.config.cont_names) >= 1
        props += ["mixed" if cat & cont else "categorical" if cat else "continuous"]
        if self.config.hierarchical:
            props += ["hierarchical"]
        if self.config.memory_name != "":
            props += ["memory"]

        return props

    def __repr__(self):
        return f"BenchmarkSet({self.config.config_id})"

    def _config_to_xs(self, configuration):
        if type(configuration) == CS.Configuration:
            configuration = configuration.get_dictionary()

        # Re-order:
        self.config_space._sort_hyperparameters()
        configuration = configuration.copy()
        configuration = {
            k: configuration.get(k)
            for k in self.config_space.get_hyperparameter_names()
            if configuration.get(k) is not None
        }

        if self.check:
            self.config_space.check_configuration(
                CS.Configuration(
                    self.config_space,
                    values=configuration,
                    allow_inactive_with_values=False,
                )
            )

        # Update with constants (constants overwrite configuration values)
        if len(self.constants):
            [configuration.update({k: v}) for k, v in self.constants.items()]

        # FIXME: check NA handling below
        all = self.config_space.get_hyperparameter_names()
        missing = list(set(all).difference(set(configuration.keys())))
        for hp in missing:
            value = (
                "#na#" if hp in self.config.cat_names else 0
            )  # '#na#' for cats, see _integer_encode below
            configuration.update({hp: value})

        x_cat = (
            np.array(
                [
                    self._integer_encode(configuration[x], x)
                    for x in self.config.cat_names
                    if x not in self.config.drop_predict
                ]
            )
            .reshape(1, -1)
            .astype(np.int32)
        )
        x_cont = (
            np.array([configuration[x] for x in self.config.cont_names])
            .reshape(1, -1)
            .astype(np.float32)
        )
        return x_cont, x_cat

    def _integer_encode(self, value, name):
        """
        Integer encode categorical variables.
        """
        # See model.py dl_from_config on how the encoding was generated and stored
        return self.encoding.get(name).get(value)

    def _get_encoding(self):
        with open(self.config.get_path("encoding"), "r") as f:
            encoding = json.load(f)
        return encoding

    def _get_config_space(self):
        with open(self.config.get_path("config_space"), "r") as f:
            json_string = f.read()
            cs = CS_json.read(json_string)
        return cs

    def _eval_random(self):
        cfg = self.config_space.sample_configuration().get_dictionary()
        return self.objective_function(cfg, logging=False, multithread=False)[0]

    def _infer_quant(self):
        offsets = []
        runtimes = []
        for i in range(15):
            start_time = time.time()
            results = self._eval_random()
            runtimes += [results[self.config.runtime_name]]
            offsets += [time.time() - start_time]

        # Compute average predicted runtime
        rt = np.mean(np.maximum(np.array(runtimes), 0.0))
        # Set the quantization factor as X offsets
        quant = np.minimum(20 * np.max(np.array(offsets)) / rt, 1.0)
        return quant

    def _get_model_path(self):
        path = self.config.get_path("model")
        if self.noisy:
            self.config.get_path("model_noisy")
        return path
