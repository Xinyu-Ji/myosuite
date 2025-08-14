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

        # 从模型文件加载MJX模型
        spec = mujoco.MjSpec.from_file(config.model_path.as_posix())
        spec = self.preprocess_spec(spec)
        self._mj_model = spec.compile()

        # 设置仿真参数
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mj_model.opt.iterations = 6
        self._mj_model.opt.ls_iterations = 6
        self._mj_model.opt.disableflags = self._mj_model.opt.disableflags | mjx.DisableBit.EULERDAMP

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = config.model_path.as_posix()

        # 行走环境特定参数
        self.min_height = config.get('min_height', 0.8)  # 最小高度阈值
        self.max_rot = config.get('max_rot', 0.8)  # 最大旋转阈值
        self.hip_period = config.get('hip_period', 100)  # 髋关节运动周期
        self.target_x_vel = config.get('target_x_vel', 0.0)  # 目标X轴速度
        self.target_y_vel = config.get('target_y_vel', 1.2)  # 目标Y轴速度
        self.target_rot = config.get('target_rot', None)  # 目标旋转
        self.steps = 0  # 步数计数器

        # 隐藏地形(如果不需要)
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
        """重置环境到初始状态"""
        self.steps = 0
        rng, rng1 = jax.random.split(rng, 2)

        # 初始化关节位置和速度
        qpos = jp.array(self._mj_model.key_qpos[0].copy())
        # qpos = jp.zeros(self._mjx_model.nq) #初始化
        qvel = jp.zeros(self._mjx_model.nv)

        # 如果需要随机初始化状态
        # if self._config.get('reset_type', 'init') == 'random':
        #     qpos = qpos + jax.random.normal(rng1, (self._mjx_model.nq,)) * 0.02
        #     # 保持高度和旋转状态不变
            # qpos = qpos.at[2].set(self._mj_model.key_qpos[0][2])
            # qpos = qpos.at[3:7].set(self._mj_model.key_qpos[0][3:7])

        data = mjx_env.init(self._mjx_model, qpos=qpos, qvel=qvel, ctrl=jp.zeros((self._mjx_model.nu,)))

        # 获取初始观测值
        obs = self._get_obs(data)
        reward, done = jp.zeros(2)
        metrics = {
            'vel_reward': jp.zeros(1),  # 速度奖励
            'cyclic_hip': jp.zeros(1),  # 髋关节周期性奖励
            'ref_rot': jp.zeros(1),  # 旋转参考奖励
            'joint_angle_rew': jp.zeros(1),  # 关节角度奖励
        }
        info = {'rng': rng}
        return State(data, {"state": obs}, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """运行环境动力学的一个时间步"""
        self.steps += 1

        # 动作归一化(如果需要)
        norm_action = action * 0 # 可以像pose_v0那样添加归一化

        data = mjx_env.step(self._mjx_model, state.data, jp.zeros((self._mjx_model.nu,)))

        # 计算各种奖励
        vel_reward = self._get_vel_reward(data)  # 速度奖励
        cyclic_hip = self._get_cyclic_rew()  # 髋关节周期性奖励
        ref_rot = self._get_ref_rotation_rew(data)  # 旋转参考奖励
        joint_angle_rew = self._get_joint_angle_rew(data, ['hip_adduction_l', 'hip_adduction_r',
                                                           'hip_rotation_l', 'hip_rotation_r'])  # 关节角度奖励

        obs = self._get_obs(data)
        # 综合奖励计算(带权重)
        reward = (vel_reward * 5.0 +
                  cyclic_hip * -10.0 +
                  ref_rot * 10.0 +
                  joint_angle_rew * 5.0)

        done = self._get_done(data)  # 终止条件判断

        # 更新指标
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
        """获取观测值: 关节位置(不含xy)、速度、质心速度、躯干角度、脚高度等"""
        qpos_without_xy = data.qpos[2:]  # 去除x,y坐标的关节位置
        qvel = data.qvel * self._mjx_model.opt.timestep  # 关节速度
        com_vel = self._get_com_velocity(data)  # 质心速度
        torso_angle = self._get_torso_angle(data)  # 躯干角度
        feet_heights = self._get_feet_heights(data)  # 脚高度
        height = self._get_height(data)  # 高度
        feet_rel_pos = self._get_feet_relative_position(data)  # 脚相对位置
        phase_var = jp.array([(self.steps / self.hip_period) % 1])  # 相位变量

        if self._mjx_model.na > 0:
            act = data.act  # 动作
        else:
            act = jp.zeros(0)

        # 拼接所有观测值
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

    # 奖励计算和辅助函数
    def _get_vel_reward(self, data):
        """速度奖励: 高斯函数激励目标行走速度"""
        vel = self._get_com_velocity(data)
        return jp.exp(-jp.square(self.target_y_vel - vel[1])) + jp.exp(-jp.square(self.target_x_vel - vel[0]))

    def _get_cyclic_rew(self):
        """髋关节周期性奖励: 激励行走步态"""
        phase_var = (self.steps / self.hip_period) % 1
        des_angles = jp.array([0.8 * jp.cos(phase_var * 2 * jp.pi + jp.pi),
                               0.8 * jp.cos(phase_var * 2 * jp.pi)])
        angles = self._get_angle(['hip_flexion_l', 'hip_flexion_r'])
        return -jp.linalg.norm(des_angles - angles)

    def _get_ref_rotation_rew(self, data):
        """旋转参考奖励: 激励保持初始参考方向"""
        target_rot = self.target_rot if self.target_rot is not None else self._mj_model.key_qpos[0][3:7]
        return jp.exp(-jp.linalg.norm(5.0 * (data.qpos[3:7] - target_rot)))

    def _get_joint_angle_rew(self, data, joint_names):
        """关节角度奖励: 激励特定关节角度"""
        angles = jp.array([data.qpos[self._mj_model.jnt_qposadr[mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, name)]]
                           for name in joint_names])
        return jp.exp(-5 * jp.mean(jp.abs(angles)))

    def _get_done(self, data):
        """终止条件判断"""
        height = self._get_height(data)
        rot_condition = self._get_rot_condition(data)
        return jp.where((height < self.min_height) | (rot_condition > self.max_rot), 1.0, 0.0)

    def _get_feet_heights(self, data):
        """获取双脚高度"""
        foot_id_l = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value,'talus_l')
        foot_id_r = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value,'talus_r')
        # foot_id_l = self._mj_model.mj_name2id('talus_l')
        # foot_id_r = self._mj_model.mj_name2id('talus_r')
        return jp.array([data.xpos[foot_id_l][2], data.xpos[foot_id_r][2]])

    def _get_feet_relative_position(self, data):
        """获取脚相对于骨盆的位置"""
        foot_id_l = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'talus_l')
        foot_id_r = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'talus_r')
        pelvis = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'pelvis')
        # pelvis = self._mj_model.body_name2id('pelvis')

        return jp.array([data.xpos[foot_id_l] - data.xpos[pelvis], data.xpos[foot_id_r] - data.xpos[pelvis]])

    def _get_torso_angle(self, data):
        """获取躯干角度"""
        body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'torso')
        # body_id = self._mj_model.body_name2id('torso')
        return data.xquat[body_id]

    def _get_com_velocity(self, data):
        """计算模型质心速度"""
        mass = jp.expand_dims(self._mj_model.body_mass, -1)
        cvel = -data.cvel
        return (jp.sum(mass * cvel, 0) / jp.sum(mass))[3:5]

    def _get_height(self, data):
        """获取质心高度"""
        mass = jp.expand_dims(self._mj_model.body_mass, -1)
        com = data.xipos
        return (jp.sum(mass * com, 0) / jp.sum(mass))[2]

    def _get_rot_condition(self, data):
        """旋转条件判断"""
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
        """获取指定关节的角度"""
        return jp.array([self._mj_model.jnt_qposadr[mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, 'torso')]
                         for name in joint_names])

    # 访问器属性
    @property
    def xml_path(self) -> str:
        """模型XML文件路径"""
        return self._xml_path

    @property
    def action_size(self) -> int:
        """动作空间大小"""
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        """MuJoCo模型"""
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        """MJX模型"""
        return self._mjx_model