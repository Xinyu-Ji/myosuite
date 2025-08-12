from typing import Any, Dict, Optional, Union, List
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State
from mujoco_playground._src import mjx_env
import numpy as np
from myosuite.utils.quat_math import quat2mat

class MjxWalkEnvV0(mjx_env.MjxEnv):
    def __init__(
            self,
            config: config_dict.ConfigDict,
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        spec = mujoco.MjSpec.from_file(config.model_path.as_posix())
        spec = self.preprocess_spec(spec)
        self._mj_model = spec.compile()

        self._mj_model.geom_margin = np.zeros(self._mj_model.geom_margin.shape)
        print(f"All margins set to 0")

        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mj_model.opt.iterations = 6
        self._mj_model.opt.ls_iterations = 6
        self._mj_model.opt.disableflags = self._mj_model.opt.disableflags | mjx.DisableBit.EULERDAMP

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = config.model_path.as_posix()

        # Walk-specific parameters
        self.min_height = config.get('min_height', 0.8)
        self.max_rot = config.get('max_rot', 0.8)
        self.hip_period = config.get('hip_period', 100)
        self.reset_type = config.get('reset_type', 'init')
        self.target_x_vel = config.get('target_x_vel', 0.0)
        self.target_y_vel = config.get('target_y_vel', 1.2)
        self.target_rot = config.get('target_rot', None)
        self.steps = 0

    def preprocess_spec(self, spec:mujoco.MjSpec):
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom.conaffinity = 0
                geom.contype = 0
                print(f"Disabled contacts for cylinder geom named \"{geom.name}\"")
        return spec

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1 = jax.random.split(rng, 2)
        # qpos = self.mjx_model.key_qpos[3]
        # qvel = self.mjx_model.key_qvel[3]
        # if self.reset_type == 'random':
        #     # Randomly start with flexed left or right knee
        #     if jax.random.uniform(rng1) < 0.5:
        #         qpos = self.mjx_model.key_qpos[2]
        #         qvel = self.mjx_model.key_qvel[2]
        #     else:
        #         qpos = self.mjx_model.key_qpos[3]
        #         qvel = self.mjx_model.key_qvel[3]

        #     # Randomize qpos coordinates but don't change height or rot state
        #     rot_state = qpos[3:7]
        #     height = qpos[2]
        #     qpos = qpos + jax.random.normal(rng1, shape=qpos.shape) * 0.02
        #     qpos = qpos.at[3:7].set(rot_state)
        #     qpos = qpos.at[2].set(height)
        # elif self.reset_type == 'init':
        #     qpos = self.mjx_model.key_qpos[2]
        #     qvel = self.mjx_model.key_qvel[2]
        # else:
        #     qpos = self.mjx_model.key_qpos[0]
        #     qvel = self.mjx_model.key_qvel[0]

        qpos = jax.random.uniform(
            rng1, (self.mjx_model.nq,),
            minval=0,
            maxval=0.01
        )
        qvel = jp.zeros(self.mjx_model.nv)

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=jp.zeros((self.mjx_model.nu,)))
        obs = self._get_obs(data)

        reward, done = jp.zeros(2)
        metrics = {
            'vel_reward': 0.0,
            'cyclic_hip': 0.0,
            'ref_rot': 0.0,
            'joint_angle_rew': 0.0,
        }
        info = {'rng': rng}
        return State(data, {"state": obs}, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data = mjx_env.step(self.mjx_model, state.data, action)
        obs = self._get_obs(data)

        # Calculate rewards
        vel_reward = self._get_vel_reward(data)
        cyclic_hip = self._get_cyclic_rew(data)
        ref_rot = self._get_ref_rotation_rew(data)
        joint_angle_rew = self._get_joint_angle_rew(data, ['hip_adduction_l', 'hip_adduction_r', 'hip_rotation_l',
                                                           'hip_rotation_r'])

        reward = vel_reward * 5.0 + cyclic_hip * (-10) + ref_rot * 10.0 + joint_angle_rew * 5.0
        done = self._get_done(data)

        state.metrics.update(
            vel_reward=vel_reward,
            cyclic_hip=cyclic_hip,
            ref_rot=ref_rot,
            joint_angle_rew=joint_angle_rew,
        )

        self.steps += 1
        return state.replace(
            data=data, obs={"state": obs}, reward=reward, done=done
        )

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:
        """Observe qpos, qvel, com_vel, torso_angle, feet_heights, etc."""
        # Get center of mass velocity
        mass = jp.expand_dims(self.mjx_model.body_mass, -1)
        cvel = -data.cvel
        com_vel = (jp.sum(mass * cvel, axis=0) / jp.sum(mass))[3:5]

        # Get torso angle
        torso_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        torso_angle = data.xquat[torso_id]

        # Get feet heights
        foot_id_l = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, 'talus_l')
        foot_id_r = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, 'talus_r')
        feet_heights = jp.array([data.xpos[foot_id_l][2], data.xpos[foot_id_r][2]])

        # Get feet relative positions
        pelvis_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        feet_rel_pos = jp.array([
            data.xpos[foot_id_l] - data.xpos[pelvis_id],
            data.xpos[foot_id_r] - data.xpos[pelvis_id]
        ])

        # Get height
        height = self._get_height(data)

        # Phase variable
        phase_var = jp.array([(self.steps / self.hip_period) % 1])

        # Muscle states
        muscle_length = data.actuator_length
        muscle_velocity = jp.clip(data.actuator_velocity, -100, 100)
        muscle_force = jp.clip(data.actuator_force / 1000, -100, 100)

        obs = jp.concatenate([
            data.qpos[2:],  # qpos without xy
            data.qvel * self.mjx_model.opt.timestep,
            com_vel,
            torso_angle,
            feet_heights,
            jp.array([height]),
            feet_rel_pos.ravel(),
            phase_var,
            muscle_length,
            muscle_velocity,
            muscle_force
        ])
        return obs

    def _get_vel_reward(self, data: mjx.Data):
        """Gaussian that incentivizes a walking velocity."""
        vel = self._get_com_velocity(data)
        return jp.exp(-jp.square(self.target_y_vel - vel[1])) + jp.exp(-jp.square(self.target_x_vel - vel[0]))

    def _get_cyclic_rew(self,data: mjx.Data):
        """Cyclic extension of hip angles is rewarded to incentivize a walking gait."""
        phase_var = (self.steps / self.hip_period) % 1
        des_angles = jp.array([0.8 * jp.cos(phase_var * 2 * jp.pi + jp.pi),
                               0.8 * jp.cos(phase_var * 2 * jp.pi)])
        angles = self._get_angle(['hip_flexion_l', 'hip_flexion_r'],data)
        return jp.linalg.norm(des_angles - angles)

    def _get_ref_rotation_rew(self, data: mjx.Data):
        """Incentivize staying close to the initial reference orientation."""
        target_rot = self.target_rot if self.target_rot is not None else self.mjx_model.key_qpos[0, 3:7]
        return jp.exp(-jp.linalg.norm(5.0 * (data.qpos[3:7] - target_rot)))

    def _get_joint_angle_rew(self, data: mjx.Data, joint_names):
        """Get a reward proportional to the specified joint angles."""
        angles = jp.array([data.qpos[mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                           for name in joint_names])
        mag = jp.mean(jp.abs(angles))
        return jp.exp(-5 * mag)

    def _get_done(self, data: mjx.Data):
        """Check if episode should terminate."""
        height = self._get_height(data)
        rot_condition = self._get_rot_condition(data)
        return (height < self.min_height) | rot_condition

    def _get_rot_condition(self, data: mjx.Data):
        """Check if body is facing in the right direction."""
        quat = data.qpos[3:7]
        return jp.abs((quat2mat(quat) @ jp.array([1, 0, 0]))[0]) > self.max_rot

    def _get_com_velocity(self, data: mjx.Data):
        """Compute the center of mass velocity of the model."""
        mass = jp.expand_dims(self.mjx_model.body_mass, -1)
        cvel = -data.cvel
        return (jp.sum(mass * cvel, axis=0) / jp.sum(mass))[3:5]

    def _get_height(self, data: mjx.Data):
        """Get center-of-mass height."""
        mass = jp.expand_dims(self.mjx_model.body_mass, -1)
        com = jp.sum(mass * data.xipos, axis=0) / jp.sum(mass)
        return com[2]

    def _get_angle(self, names: List[str],data: mjx.Data) -> jp.ndarray:
        """
        Get the angles of a list of named joints (MJX version).

        Args:
            names: List of joint names to get angles for

        Returns:
            jp.ndarray: Array of joint angles in radians
        """


        angles = []
        for name in names:
            joint_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self._mj_model.jnt_qposadr[joint_id]
            angles.append(data.qpos[qpos_adr])
        return jp.array(angles)

    # Accessors
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model