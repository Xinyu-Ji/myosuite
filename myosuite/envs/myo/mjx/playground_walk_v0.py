from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State
from mujoco_playground._src import mjx_env
import numpy as np


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
        self.target_x_vel = config.get('target_x_vel', 0.0)
        self.target_y_vel = config.get('target_y_vel', 1.2)
        self.target_rot = config.get('target_rot', None)
        self.steps = 0

        # Move heightfield down if not used
        # self._mj_model.geom_rgba[self._mj_model.geom_name2id('terrain')][-1] = 0.0
        # self._mj_model.geom_pos[self._mj_model.geom_name2id('terrain')] = np.array([0, 0, -10])

    def preprocess_spec(self, spec:mujoco.MjSpec):
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom.conaffinity = 0
                geom.contype = 0
                print(f"Disabled contacts for cylinder geom named \"{geom.name}\"")
        return spec

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        self.steps = 0
        rng, rng1 = jax.random.split(rng, 2)

        # Initialize qpos and qvel
        qpos = jp.array(self.mj_model.key_qpos[0].copy())
        qvel = jp.zeros(self.mjx_model.nv)

        # Randomize initial state if needed
        # if self._config.get('reset_type', 'init') == 'random':
        #     qpos = qpos + jax.random.normal(rng1, (self._mjx_model.nq,)) * 0.02
        #     # Keep height and rotation state unchanged
        #     qpos = qpos.at[2].set(self._mj_model.key_qpos[0][2])
        #     qpos = qpos.at[3:7].set(self._mj_model.key_qpos[0][3:7])

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=jp.zeros((self.mjx_model.nu,)))

        obs = self._get_obs(data)
        reward, done = jp.zeros(2)
        metrics = {
            'vel_reward': jp.zeros(1),
            'cyclic_hip': jp.zeros(1),
            'ref_rot': jp.zeros(1),
            'joint_angle_rew': jp.zeros(1),
        }
        info = {'rng': rng}
        return State(data, {"state": obs}, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        self.steps += 1

        # Normalize action if needed
        norm_action = action  # Could add normalization like in pose_v0 if needed

        data = mjx_env.step(self.mjx_model, state.data, norm_action)

        # Compute rewards
        vel_reward = self._get_vel_reward(data)
        cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_rotation_rew(data)
        joint_angle_rew = self._get_joint_angle_rew(data, ['hip_adduction_l', 'hip_adduction_r',
                                                           'hip_rotation_l', 'hip_rotation_r'])

        obs = self._get_obs(data)
        reward = (vel_reward * 5.0 +
                  cyclic_hip * -10.0 +
                  ref_rot * 10.0 +
                  joint_angle_rew * 5.0)

        done = self._get_done(data)

        state.metrics.update(
            vel_reward=vel_reward,
            cyclic_hip=cyclic_hip,
            ref_rot=ref_rot,
            joint_angle_rew=joint_angle_rew,
        )

        return state.replace(
            data=data, obs={"state": obs}, reward=reward, done=done
        )

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:
        """Observe qpos (without xy), qvel, com_vel, torso_angle, feet heights, etc."""
        qpos_without_xy = data.qpos[2:]
        qvel = data.qvel * self.mjx_model.opt.timestep
        com_vel = self._get_com_velocity(data)
        torso_angle = self._get_torso_angle(data)
        feet_heights = self._get_feet_heights(data)
        height = self._get_height(data)
        feet_rel_pos = self._get_feet_relative_position(data)
        phase_var = jp.array([(self.steps / self.hip_period) % 1])

        if self.mjx_model.na > 0:
            act = data.act
        else:
            act = jp.zeros(0)

        return jp.concatenate([
            qpos_without_xy,
            qvel,
            com_vel,
            torso_angle,
            feet_heights,
            jp.array([height]),
            feet_rel_pos.flatten(),
            phase_var,
            act
        ])

    # Reward and helper functions
    def _get_vel_reward(self, data):
        vel = self._get_com_velocity(data)
        return jp.exp(-jp.square(self.target_y_vel - vel[1])) + jp.exp(-jp.square(self.target_x_vel - vel[0]))

    def _get_cyclic_rew(self):
        phase_var = (self.steps / self.hip_period) % 1
        des_angles = jp.array([0.8 * jp.cos(phase_var * 2 * jp.pi + jp.pi),
                               0.8 * jp.cos(phase_var * 2 * jp.pi)])
        angles = self._get_angle(['hip_flexion_l', 'hip_flexion_r'])
        return -jp.linalg.norm(des_angles - angles)

    def _get_ref_rotation_rew(self, data):
        target_rot = self.target_rot if self.target_rot is not None else self._mj_model.key_qpos[0][3:7]
        return jp.exp(-jp.linalg.norm(5.0 * (data.qpos[3:7] - target_rot)))

    def _get_joint_angle_rew(self, data, joint_names):
        angles = jp.array([data.qpos[self.mj_model.jnt_qposadr[mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, name)]]
                           for name in joint_names])
        return jp.exp(-5 * jp.mean(jp.abs(angles)))

    def _get_done(self, data):
        height = self._get_height(data)
        rot_condition = self._get_rot_condition(data)
        return jp.where((height < self.min_height) | (rot_condition > self.max_rot), 1.0, 0.0)

    def _get_feet_heights(self, data):
        foot_id_l = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'talus_l')
        foot_id_r = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'talus_r')
        return jp.array([data.xpos[foot_id_l][2], data.xpos[foot_id_r][2]])

    def _get_feet_relative_position(self, data):

        foot_id_l = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'talus_l')
        foot_id_r = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'talus_r')
        pelvis = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'pelvis')
        # foot_id_l = self._mj_model.body_name2id('talus_l')
        # foot_id_r = self._mj_model.body_name2id('talus_r')
        # pelvis = self._mj_model.body_name2id('pelvis')
        return jp.array([data.xpos[foot_id_l] - data.xpos[pelvis], data.xpos[foot_id_r] - data.xpos[pelvis]])

    def _get_torso_angle(self, data):
        # body_id = self._mj_model.body_name2id('torso')
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'torso')
        return data.xquat[body_id]

    def _get_com_velocity(self, data):
        mass = jp.expand_dims(self.mj_model.body_mass, -1)
        cvel = -data.cvel
        return (jp.sum(mass * cvel, 0) / jp.sum(mass))[3:5]

    def _get_height(self, data):
        mass = jp.expand_dims(self.mj_model.body_mass, -1)
        com = data.xipos
        return (jp.sum(mass * com, 0) / jp.sum(mass))[2]

    def _get_rot_condition(self, data):
        quat = data.qpos[3:7].copy()
        mat = jp.array([[1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
                         2.0 * (quat[1] * quat[2] - quat[0] * quat[3]),
                         2.0 * (quat[1] * quat[3] + quat[0] * quat[2])],
                        [2.0 * (quat[1] * quat[2] + quat[0] * quat[3]),
                         1.0 - 2.0 * (quat[1] ** 2 + quat[3] ** 2),
                         2.0 * (quat[2] * quat[3] - quat[0] * quat[1])],
                        [2.0 * (quat[1] * quat[3] - quat[0] * quat[2]),
                         2.0 * (quat[2] * quat[3] + quat[0] * quat[1]),
                         1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)]])
        return jp.abs((mat @ jp.array([1.0, 0.0, 0.0]))[0])

    def _get_angle(self, joint_names):
        return jp.array([self.mj_model.jnt_qposadr[mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, name)]
                         for name in joint_names])

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