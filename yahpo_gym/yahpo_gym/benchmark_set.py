from yahpo_gym.configuration import cfg
import onnxruntime as rt
import time
import json
import copy

from pathlib import Path
from typing import Union, Dict, List
import numpy as np
from ConfigSpace.read_and_write import json as CS_json
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

class BenchmarkSet():

    def __init__(self, config_id: str = None, download: bool = True, active_session: bool = False,
        session: Union[rt.InferenceSession, None] = None, check: bool = True):
        """
        Interface for a benchmark scenario. 
        Initialized with a valid key for a valid scenario and optinally an `onnxruntime.InferenceSession`.

        Parameters
        ----------
        config_id: str
            (Required) A key for `ConfigDict` pertaining to a valid benchmark scenario (e.g. `lcbench`).
        active_session: bool
            Should the benchmark run in an active `onnxruntime.InferenceSession`? Initialized to `False`.
        session: onnx.Session
            A ONNX session to use for inference. Overwrite `active_session` and sets the provided `onnxruntime.InferenceSession` as the active session.
            Initialized to `None`.
        check: bool
            Should input to objective_function* be checked for validity? Initialized to True, can be
            disabled for speedups.
        """
        self.config = cfg(config_id, download=download)
        self.encoding = self._get_encoding()
        self.config_space = self._get_config_space()
        self.active_session = active_session
        self.check = check
        self.quant = 0.1
        self.constants = {}
        self.session = None
        self.archive = []

        if self.active_session or (session is not None):
            self.set_session(session)

    def objective_function(self, configuration: Union[Dict, List[Dict]], logging: bool = False):
        """
        Evaluate the surrogate for a given configuration.

        Parameters
        ----------
        configuration: Dict
            A valid dict containing hyperparameters to be evaluated. 
            Attention: `configuration` is not checked for internal validity for speed purposes.
        logging: bool
            Should the evaluation be logged in the `archive`? Initialized to `False`.
        """
        if not self.active_session:
            self.set_session()

        x_cont, x_cat = self._config_to_xs(configuration)
        # input & output names and dims
        input_names = [x.name for x in self.session.get_inputs()]
        output_name = self.session.get_outputs()[0].name
        results = self.session.run([output_name], {input_names[0]: x_cat, input_names[1]: x_cont})[0][0]

        results_dict = {k:v for k,v in zip(self.config.y_names, results)}
        if logging:
            timedate = time.strftime("%D|%H:%M:%S", time.localtime())
            self.archive.append({'time':timedate, 'x':configuration, 'y':results_dict})

        return results_dict

    def objective_function_timed(self, configuration: Union[Dict, List[Dict]], logging: bool = False):
        """
        Evaluate the surrogate for a given configuration and sleep for quant * predicted runtime.
        Note, that this assumes that the predicted runtime is in seconds.

        Parameters
        ----------
        configuration: Dict
            A valid dict containing hyperparameters to be evaluated. 
            Attention: `configuration` is not checked for internal validity for speed purposes.
        logging: bool
            Should the evaluation be logged in the `archive`? Initialized to `False`.
        """
        start_time = time.time()
        results = self.objective_function(configuration)
        rt = results[self.config.runtime_name]
        offset = time.time() - start_time
        sleepit = max(rt - offset, 0) * self.quant
        time.sleep(sleepit)
        return results

    def set_constant(self, param: str, value = None):
        """
        Set a given hyperparameter to a constant.

        Parameters
        ----------
        param: str
            A valid parameter name.
        value: int | str | any
            A valid value for the parameter `param`.
        """
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

    def get_opt_space(self, instance:str, drop_fidelity_params:bool = True):
        """
        Get the search space to be optimized.
        Sets 'instance' as a constant instance and removes all fidelity parameters if 'drop_fidelity_params = True'.
        
        Parameters
        ----------
        instance: str
            A valid instance. See `instances`.
        drop_fidelity_params: bool
            Should fidelity params be dropped from the `opt_space`? Defaults to `True`.
        """
        # FIXME: assert instance is a valid choice
        csn = copy.deepcopy(self.config_space)
        hps = csn.get_hyperparameters()
        instance_names_idx = csn.get_hyperparameter_names().index(self.config.instance_names)
        hps[instance_names_idx] = CSH.Constant(self.config.instance_names, instance)
        if drop_fidelity_params:
            fidelity_params_idx = [csn.get_hyperparameter_names().index(fidelity_param) for fidelity_param in self.config.fidelity_params]
            for idx in fidelity_params_idx:
                del hps[idx]
        cnds = csn.get_conditions()
        fbds = csn.get_forbiddens()
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        cs.add_conditions(cnds)
        cs.add_forbidden_clauses(fbds)
        return cs

    def get_fidelity_space(self):
        """
        Get the fidelity space to be optimized for.
        """
        csn = copy.deepcopy(self.config_space)
        hps = csn.get_hyperparameters()
        fidelity_params_idx = [csn.get_hyperparameter_names().index(fidelity_param) for fidelity_param in self.config.fidelity_params]
        hps = [hps[idx] for idx in fidelity_params_idx]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters(hps)
        return cs


    def set_session(self, session: Union[rt.InferenceSession, None] = None):
        """
        Set the session for inference on the surrogate model.

        Parameters
        ----------
        session: onnxruntime.InferenceSession
            A ONNX session to use for inference. Overwrite `active_session` and sets the provided `onnxruntime.InferenceSession` as the active session.
            Initialized to `None`.
        """
        # Either overwrite session or instantiate a new one if no active session exists
        if (session is not None):
            self.session = session
        elif (self.session is None):
            model_path = self.config.get_path("model")
            if not Path(model_path).is_file():
                raise Exception(f("ONNX file {model_path} not found!"))
            options = rt.SessionOptions()
            options.inter_op_num_threads = 1L
            optionss.intra_op_num_threads = 1L
            self.session = rt.InferenceSession(model_path, sess_options = options)

    @property
    def instances(self):
        """
        A list of valid instances for the scenario.
        """
        if self.config.instance_names is None:
            return []
        return [*self.config_space.get_hyperparameter(self.config.instance_names).choices]


    def __repr__(self):
        return f"BenchmarkSet ({self.config.config_id})"

    def _config_to_xs(self, configuration):
        if type(configuration) == CS.Configuration:
            configuration = configuration.get_dictionary()

        # Re-order:
        self.config_space._sort_hyperparameters()
        configuration = configuration.copy()
        configuration = {k: configuration.get(k) for k in self.config_space.get_hyperparameter_names() if configuration.get(k) is not None}

        if self.check:
            self.config_space.check_configuration(CS.Configuration(self.config_space, values = configuration, allow_inactive_with_values = False))

        # FIXME: This should work with configuration as a `List` of Dicts
        # Update with constants (constants overwrite configuration values)
        if len(self.constants):
            [configuration.update({k : v}) for k,v in self.constants.items()]

        # FIXME: check NA handling below
        all = self.config_space.get_hyperparameter_names()
        missing = list(set(all).difference(set(configuration.keys())))
        for hp in missing:
            value = '#na#' if hp in self.config.cat_names else 0  # '#na#' for cats, see _integer_encode below
            configuration.update({hp:value})


        x_cat = np.array([self._integer_encode(configuration[x], x) for x in self.config.cat_names if x not in self.config.drop_predict]).reshape(1, -1).astype(np.int32)
        x_cont = np.array([configuration[x] for x in self.config.cont_names]).reshape(1, -1).astype(np.float32)
        return x_cont, x_cat

    def _integer_encode(self, value, name):
        """
        Integer encode categorical variables.
        """
        # see model.py dl_from_config on how the encoding was generated and stored
        return self.encoding.get(name).get(value)

    def _get_encoding(self):
        with open(self.config.get_path("encoding"), 'r') as f:
            encoding = json.load(f)
        return encoding

    def _get_config_space(self):
        with open(self.config.get_path("config_space"), 'r') as f:
            json_string = f.read()
            cs = CS_json.read(json_string)
        return cs

    def _eval_random(self):
        cfg = self.config_space.sample_configuration().get_dictionary()
        return self.objective_function_timed(cfg)
