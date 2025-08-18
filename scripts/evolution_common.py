import numpy as np
from pxr import Usd, UsdGeom, Gf
from isaacsim.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat, gf_quat_to_np_array
import omni.usd
from stable_baselines3 import PPO
import torch


class Agent:
    """Represents an agent with its policy and parameters."""
    
    PARAMETERIZED_PRIM_NAMES = [
        "/World/robot/right_lower_digit",
        "/World/robot/left_lower_digit",
        "/World/robot/back_lower_digit",
    ]
    PARAMETERIZED_JOINT_NAMES= [
        "/World/robot/left_lower_digit/left_lower_fixed",
        "/World/robot/right_lower_digit/right_lower_fixed",
        "/World/robot/back_lower_digit/back_lower_fixed",
    ]
    PARAMETERIZED_FIXED_JOINT_NAMES = [
        "/World/robot/palm/joint_left",
        "/World/robot/palm/joint_right",
        "/World/robot/palm/joint_back",
    ]
    PARAM_OFFSET = 1/(2*np.sqrt(2))
    MIN_SCALE_LENGTH = 0.3
    MAX_SCALE_LENGTH = 2.0
    MIN_SCALE_WIDTH = 0.1
    MAX_SCALE_WIDTH = 1.0
    MIN_JOINT_ANGLE = 0
    MAX_JOINT_ANGLE = 2*np.pi
    NUM_PARAMS = len(PARAMETERIZED_PRIM_NAMES)


    def __init__(self, agent_id: int, params: np.ndarray, policy_arch: str, agent_cfg: dict, generation: int = 0, parent_id: int = None):
        self.id = agent_id
        self.params = params
        self.policy_arch = policy_arch
        self.agent_cfg = agent_cfg.copy()
        self.model = None
        self.fitness_history = dict()
        self.generation = generation
        self.parent_id = parent_id
        self.family_tree = []  # List of (generation, parent_id) tuples
        self._parent_model_params = None  # Store parent's model parameters for inheritance
        
        # Build family tree if this is a mutated agent
        if parent_id is not None:
            self.family_tree.append((generation, parent_id))
    
    def mutate(self, new_id: int, generation: int, mutation_strength: float, rng):
        """Create a mutated copy of this agent."""

        # Add gaussian noise to parameters
        range = np.array([Agent.MAX_SCALE_LENGTH, Agent.MAX_SCALE_WIDTH, Agent.MAX_JOINT_ANGLE]) - np.array([Agent.MIN_SCALE_LENGTH, Agent.MIN_SCALE_WIDTH, Agent.MIN_JOINT_ANGLE])
        noise = rng.normal(0, mutation_strength, size=self.params.shape) * range
        length_params = [
            np.clip(param + noise_val, Agent.MIN_SCALE_LENGTH, Agent.MAX_SCALE_LENGTH) 
            for param, noise_val in zip(self.params[:, 0], noise[:, 0])
        ]
        width_params = [
            np.clip(param + noise_val, Agent.MIN_SCALE_WIDTH, Agent.MAX_SCALE_WIDTH) 
            for param, noise_val in zip(self.params[:, 1], noise[:, 1])
        ]
        rotation_params = [
            ((param + noise_val) % Agent.MAX_JOINT_ANGLE)
            for param, noise_val in zip(self.params[:, 2], noise[:, 2])
        ]
        new_params = np.array([length_params, width_params, rotation_params]).T
        
        # Create new agent with family tree
        mutated_agent = Agent(new_id, new_params, self.policy_arch, self.agent_cfg, generation, self.id)
        
        # Copy family tree and add this mutation
        mutated_agent.family_tree = self.family_tree.copy()
        mutated_agent.family_tree.append((generation, self.id))
        
        # Copy parent's model parameters if parent has a trained model
        if self.model is not None:
            mutated_agent._parent_model_params = self.model.get_parameters()
        
        return mutated_agent
    
    def update_model(self, env):
        if self.model is None:
            self.model = PPO(self.policy_arch, env, **self.agent_cfg)
            # If this agent has inherited parameters from a parent, load them
            if self._parent_model_params is not None:
                self.model.set_parameters(self._parent_model_params)
                # Clear the stored parameters to save memory
                self._parent_model_params = None
        else:
            new_model = PPO(self.policy_arch, env, **self.agent_cfg)
            new_model.set_parameters(self.model.get_parameters())
            del self.model
            self.model = new_model

    def train(self, timesteps: int, callback=None):
        """Train the agent for specified timesteps."""
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        self.model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=False)
    
    def evaluate_fitness(self, env, generation, num_episodes: int = 1):
        """Evaluate agent fitness by running episodes and measuring performance."""
        if self.model is None:
            return 0.0
        
        total_reward = 0.0
        for _ in range(num_episodes):
            obs = env.reset()
            done_count = 0
            episode_reward = 0.0
            
            while done_count < env.num_envs * num_episodes:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward.mean() if hasattr(reward, 'mean') else reward
                done_count += done.sum().item()
            
            total_reward += episode_reward
        
        fitness = total_reward / num_episodes
        self.fitness_history[generation] = fitness
        return fitness
    

def modify_params_usd(task_folder: str, parameters: np.ndarray):
    """Modify the USD file with new parameters."""
    stage = Usd.Stage.Open(f"source/IsaacEnv/IsaacEnv/tasks/direct/{task_folder}/Grasp3D.usd")
    prims = [stage.GetPrimAtPath(prim_name) for prim_name in Agent.PARAMETERIZED_PRIM_NAMES]
    joints = [stage.GetPrimAtPath(joint_name) for joint_name in Agent.PARAMETERIZED_FIXED_JOINT_NAMES]
    
    # Scale each prim to the new scale factor
    for param, prim, joint in zip(parameters, prims, joints):
        prim_xform = UsdGeom.Xformable(prim)

        # Find or create scale operation
        scale_op = None
        for op in prim_xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break
        if scale_op is None:
            scale_op = prim_xform.AddScaleOp()

        # Set the scale
        scale = scale_op.Get()
        scale[0] = param[1]  # length
        scale[2] = param[0]  # width
        scale_op.Set(scale)
        
        # Rotate the joint around the palm
        attr = joint.GetAttribute('physics:localRot0')
        current_q = attr.Get()
        quat_np = gf_quat_to_np_array(current_q)
        euler = quat_to_euler_angles(quat_np, degrees=False, extrinsic=True)
        euler[2] = param[2]
        new_quat_np = euler_angles_to_quat(euler, degrees=False, extrinsic=True)
        new_quat = Gf.Quatf(new_quat_np[0], new_quat_np[1], new_quat_np[2], new_quat_np[3])
        attr.Set(new_quat)

    stage.GetRootLayer().Export(f"source/IsaacEnv/IsaacEnv/tasks/direct/{task_folder}/Grasp3D-temp.usd")

    # Reload the USD file using Omniverse context
    context = omni.usd.get_context()
    context.new_stage()
    context.get_stage().Reload()

    # Clear GPU vcache
    torch.cuda.empty_cache()