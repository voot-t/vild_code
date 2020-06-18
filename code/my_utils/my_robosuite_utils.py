from my_utils import *

import h5py
import robosuite
from robosuite.wrappers import Wrapper
from robosuite.utils.mjcf_utils import postprocess_model_xml
from gym import spaces

def make_robosuite_env(args):
    demo_dict = {
                ## Robosuite
                21 : "pegs-RoundNut",   #
                22 : "pegs-SquareNut",  #
                23 : "pegs-full",
                24 : "bins-Bread",      #
                25 : "bins-Can",        # 
                26 : "bins-Cereal",
                27 : "bins-Milk",        # 
                28 : "bins-full",
    }
    demo_name = demo_dict[args.env_id]

    hd5_demo_path = "../imitation_data/RoboTurkPilot/%s" % (demo_name)

    env_name = args.env_name + "_reach"
    # total_traj = len(os.listdir(demo_path))
    demo_list = []
    filename = "../imitation_data/TRAJ_h5/%s/%s_chosen.txt" % (env_name, env_name)
    demo_idx = -1

    sort_list = np.arange(0, 10)   # 10 demo. 
    with open(filename, 'r') as ff:
        i = 0
        for line in ff:
            line = line.replace(":", " ").replace("(", " ").replace(")", " ").replace(",", " ").split()
            if demo_idx == -1:
                demo_idx = line.index("demo") + 1
            if np.any(sort_list == i):
                demo_list += [int(line[demo_idx])]
            i = i + 1
 
    env = RobosuiteReacherWrapper(robosuite.make(
        args.env_name,
        has_offscreen_renderer=False,  # not needed since we do not use pixel obs
        has_renderer=args.render,
        ignore_done=False,  
        use_camera_obs=False,
        gripper_visualization=False,
        reward_shaping=False,   # 
        control_freq=100,
        horizon = args.t_max  # default is 500 
        ),
        demo_path=hd5_demo_path,
        demo_list=demo_list,
        need_xml=True,
        sampling_schemes=["uniform_first", "random"], # initial state sampler. Default from the repo is uniform + random, but initilizing states at the middle of trajectory is possible only in simulations.. 
        scheme_ratios=[0.9, 0.1],
        robo_task = "reach",
    )
    return env 

""" Mostly from the robosuite github repo. This code simply changes reward, makes gripper action constant, and uses gym interface. 
    ## are (mostly) comments for additional/change parts. 
"""
class RobosuiteReacherWrapper(Wrapper):
    env = None

    def __init__(
        self,
        env,
        demo_path,
        need_xml=False,
        # num_traj=-1,
        demo_list=[None],
        sampling_schemes=["uniform", "random"],
        scheme_ratios=[0.9, 0.1],
        open_loop_increment_freq=100,
        open_loop_initial_window_width=25,
        open_loop_window_increment=25,
        keys=None,
        robo_task="reach"
    ):
        """
        Initializes a wrapper that provides support for resetting the environment
        state to one from a demonstration. It also supports curriculums for
        altering how often to sample from demonstration vs. sampling a reset
        state from the environment.
        Args:
            env (MujocoEnv instance): The environment to wrap.
            demo_path (string): The path to the folder containing the demonstrations.
                There should be a `demo.hdf5` file and a folder named `models` with 
                all of the stored model xml files from the demonstrations.
            
            need_xml (bool): If True, the mujoco model needs to be reloaded when
                sampling a state from a demonstration. This could be because every
                demonstration was taken under varied object properties, for example.
                In this case, every sampled state comes with a corresponding xml to
                be used for the environment reset.
            preload (bool): If True, fetch all demonstrations into memory at the
                beginning. Otherwise, load demonstrations as they are needed lazily.
            num_traj (int): If provided, subsample @number demonstrations from the 
                provided set of demonstrations instead of using all of them.
            sampling_schemes (list of strings): A list of sampling schemes
                to be used. The following strings are valid schemes:
                    "random" : sample a reset state directly from the wrapped environment
                    "uniform" : sample a state from a demonstration uniformly at random
                    "forward" : sample a state from a window that grows progressively from
                        the start of demonstrations
                    "reverse" : sample a state from a window that grows progressively from
                        the end of demonstrations
            scheme_ratios (list of floats): A list of probability values to
                assign to each member of @sampling_schemes. Must be non-negative and
                sum to 1.
            open_loop_increment_freq (int): How frequently to increase
                the window size in open loop schemes ("forward" and "reverse"). The
                window size will increase by @open_loop_window_increment every
                @open_loop_increment_freq samples. Only samples that are generated
                by open loop schemes contribute to this count.
            open_loop_initial_window_width (int): The width of the initial sampling
                window, in terms of number of demonstration time steps, for
                open loop schemes.
            open_loop_window_increment (int): The window size will increase by
                @open_loop_window_increment every @open_loop_increment_freq samples.
                This number is in terms of number of demonstration time steps.
        """
        
        super().__init__(env)

        ## Gym part 
        if keys is None:
            assert self.env.use_object_obs, "Object observations need to be enabled."
            keys = ["robot-state", "object-state"]
        self.keys = keys

        # set up observation and action spaces
        flat_ob = self._flatten_obs(self.env.reset(), verbose=True)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

        self.robo_task = robo_task 

        self.demo_path = demo_path
        hdf5_path = os.path.join(self.demo_path, "demo.hdf5")
        self.demo_file = h5py.File(hdf5_path, "r")

        # ensure that wrapped env matches the env on which demonstrations were collected
        env_name = self.demo_file["data"].attrs["env"]
        assert (
            env_name == self.unwrapped.__class__.__name__
        ), "Wrapped env {} does not match env on which demos were collected ({})".format(
            env.__class__.__name__, env_name
        )

        # list of all demonstrations episodes
        # self.demo_list = list(self.demo_file["data"].keys())

        # subsample a selection of demonstrations if requested
        if demo_list[0] is None :
            # random.seed(3141)  # ensure that the same set is sampled every time
            # self.demo_list = random.sample(self.demo_list, num_traj)
            self.demo_list = list(self.demo_file["data"].keys())
        else:
            self.demo_list = [ "demo_%d" % i for i in demo_list ]

        self.need_xml = need_xml
        self.demo_sampled = 0

        self.sample_method_dict = {
            "random": "_random_sample",
            "uniform": "_uniform_sample",
            "uniform_half": "_uniform_sample_half",
            "uniform_first": "_uniform_sample_first",
            "forward": "_forward_sample_open_loop",
            "reverse": "_reverse_sample_open_loop",
        }

        self.sampling_schemes = sampling_schemes
        self.scheme_ratios = np.asarray(scheme_ratios)

        # make sure the list of schemes is valid
        schemes = self.sample_method_dict.keys()
        assert np.all([(s in schemes) for s in self.sampling_schemes])

        # make sure the distribution is the correct size
        assert len(self.sampling_schemes) == len(self.scheme_ratios)

        # make sure the distribution lies in the probability simplex
        assert np.all(self.scheme_ratios > 0.)
        assert sum(self.scheme_ratios) == 1.0

        # open loop configuration
        self.open_loop_increment_freq = open_loop_increment_freq
        self.open_loop_window_increment = open_loop_window_increment

        # keep track of window size
        self.open_loop_window_size = open_loop_initial_window_width

    ## Gym part 
    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict: ordered dictionary of observations
        """
        ob_lst = []
        for key in obs_dict:
            if key in self.keys:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(obs_dict[key])
        return np.concatenate(ob_lst)


    def step(self, action):
        ## reacher case
        if self.robo_task == "reach":
            action = np.append(action, [0]) ## append null gripper command to input action 

        ob_dict, reward, done, info = self.env.step(action) 
        ## The returned done from step checks only horizon length, at least for the assemblyroundnut task. 
        ## Edit: This does not seem to matter for reacher case, since _check_success is for assembly tasks. 
        done = done or self._check_success()
    
        if self.robo_task == "reach":
            r_reach, _, _, _ = self.env.staged_rewards()
            reward = r_reach * 10   #otherwise the reward is too small given the current float printout format. 
            
        return self._flatten_obs(ob_dict), reward, done, info

    def reset(self):
        """
        Logic for sampling a state from the demonstration and resetting
        the simulation to that state. 
        """
        state = self.sample()
        if state is None:
            # None indicates that a normal env reset should occur
            #return self.env.reset()
            ## Gym version
            return self._flatten_obs(self.env.reset())

        else:
            if self.need_xml:
                # reset the simulation from the model if necessary
                state, xml = state
                self.env.reset_from_xml_string(xml)

            if isinstance(state, tuple):
                state = state[0]

            # force simulator state to one from the demo
            self.sim.set_state_from_flattened(state)
            self.sim.forward()

            # return self.self.env._get_observation()
            ## Gym version 
            return self._flatten_obs(self.env._get_observation())

    def sample(self):
        """
        This is the core sampling method. Samples a state from a
        demonstration, in accordance with the configuration.
        """

        # chooses a sampling scheme randomly based on the mixing ratios
        seed = random.uniform(0, 1)
        ratio = np.cumsum(self.scheme_ratios)
        ratio = ratio > seed
        for i, v in enumerate(ratio):
            if v:
                break

        sample_method = getattr(self, self.sample_method_dict[self.sampling_schemes[i]])
        return sample_method()

    def _random_sample(self):
        """
        Sampling method.

        Return None to indicate that the state should be sampled directly
        from the environment.
        """
        return None

    def _uniform_sample(self):
        """
        Sampling method.

        First uniformly sample a demonstration from the set of demonstrations.
        Then uniformly sample a state from the selected demonstration.
        """

        # get a random episode index
        ep_ind = random.choice(self.demo_list)

        # select a flattened mujoco state uniformly from this episode
        states = self.demo_file["data/{}/states".format(ep_ind)].value
        state = random.choice(states)

        if self.need_xml:
            model_xml = self._xml_for_episode_index(ep_ind)
            xml = postprocess_model_xml(model_xml)
            return state, xml
        return state

    def _uniform_sample_half(self):
        """
        ## Uniformly sample states from first half of demo.
        """

        # get a random episode index
        ep_ind = random.choice(self.demo_list)

        # select a flattened mujoco state uniformly from this episode
        states = self.demo_file["data/{}/states".format(ep_ind)].value
        state = random.choice(states[0:len(states)//2])

        if self.need_xml:
            model_xml = self._xml_for_episode_index(ep_ind)
            xml = postprocess_model_xml(model_xml)
            return state, xml
        return state

    def _uniform_sample_first(self):
        """
        ## Uniformly sample states from first state of demo.
        """

        # get a random episode index
        ep_ind = random.choice(self.demo_list)

        # select a flattened mujoco state uniformly from this episode
        states = self.demo_file["data/{}/states".format(ep_ind)].value

        state = states[0] # random.choice(states[0:len(states)//2])

        if self.need_xml:
            model_xml = self._xml_for_episode_index(ep_ind)
            xml = postprocess_model_xml(model_xml)
            return state, xml
        return state

    def _reverse_sample_open_loop(self):
        """
        Sampling method.

        Open loop reverse sampling from demonstrations. Starts by 
        sampling from states near the end of the demonstrations.
        Increases the window backwards as the number of calls to
        this sampling method increases at a fixed rate.
        """

        # get a random episode index
        ep_ind = random.choice(self.demo_list)

        # sample uniformly in a window that grows backwards from the end of the demos
        states = self.demo_file["data/{}/states".format(ep_ind)].value
        eps_len = states.shape[0]
        index = np.random.randint(max(eps_len - self.open_loop_window_size, 0), eps_len)
        state = states[index]

        # increase window size at a fixed frequency (open loop)
        self.demo_sampled += 1
        if self.demo_sampled >= self.open_loop_increment_freq:
            if self.open_loop_window_size < eps_len:
                self.open_loop_window_size += self.open_loop_window_increment
            self.demo_sampled = 0

        if self.need_xml:
            model_xml = self._xml_for_episode_index(ep_ind)
            xml = postprocess_model_xml(model_xml)
            return state, xml

        return state

    def _forward_sample_open_loop(self):
        """
        Sampling method.

        Open loop forward sampling from demonstrations. Starts by
        sampling from states near the beginning of the demonstrations.
        Increases the window forwards as the number of calls to
        this sampling method increases at a fixed rate.
        """

        # get a random episode index
        ep_ind = random.choice(self.demo_list)

        # sample uniformly in a window that grows forwards from the beginning of the demos
        states = self.demo_file["data/{}/states".format(ep_ind)].value
        eps_len = states.shape[0]
        index = np.random.randint(0, min(self.open_loop_window_size, eps_len))
        state = states[index]

        # increase window size at a fixed frequency (open loop)
        self.demo_sampled += 1
        if self.demo_sampled >= self.open_loop_increment_freq:
            if self.open_loop_window_size < eps_len:
                self.open_loop_window_size += self.open_loop_window_increment
            self.demo_sampled = 0

        if self.need_xml:
            model_xml = self._xml_for_episode_index(ep_ind)
            xml = postprocess_model_xml(model_xml)
            return state, xml

        return state

    def _xml_for_episode_index(self, ep_ind):
        """
        Helper method to retrieve the corresponding model xml string
        for the passed episode index.
        """

        # read the model xml, using the metadata stored in the attribute for this episode
        model_file = self.demo_file["data/{}".format(ep_ind)].attrs["model_file"]
        model_path = os.path.join(self.demo_path, "models", model_file)
        with open(model_path, "r") as model_f:
            model_xml = model_f.read()
        return model_xml
