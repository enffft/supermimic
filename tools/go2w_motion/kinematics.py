"""Named Go2W model mapping, analytic leg IK and MuJoCo geometry."""

from itertools import combinations

import mujoco
import numpy as np
import yaml

from . import LEGS, MODEL, TRAIN

UP = np.array([0., 0., 1.])
IDENTITY = np.array([1., 0., 0., 0.])


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def hull(points):
    """Return a CCW convex hull without redundant points along straight edges."""
    points = sorted(set(map(tuple, points)))
    if len(points) <= 2:
        return np.asarray(points)
    halves = []
    for order in (points, points[::-1]):
        chain = []
        for point in map(np.asarray, order):
            while len(chain) >= 2 and cross2(chain[-1] - chain[-2], point - chain[-1]) <= 0:
                chain.pop()
            chain.append(point)
        halves.extend(chain[:-1])
    return np.array(halves)


def support_geometry(points, com):
    polygon = hull(points[:, :2])
    if len(polygon) < 3:
        return polygon, -1., np.zeros(2)
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.column_stack((-edges[:, 1], edges[:, 0])) / np.linalg.norm(edges, axis=1)[:, None]
    distances = np.sum((com[:2] - polygon) * normals, axis=1)
    nearest = distances.argmin()
    return polygon, distances[nearest], normals[nearest]


def wheel_rates(center_vel, carrier_omega, axis, arm):
    """Solve contact tangency along the rolling direction."""
    direction = np.cross(axis, UP)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    without_spin = center_vel + np.cross(carrier_omega, arm)
    spin = np.cross(axis, arm)
    return -np.sum(without_spin * direction, axis=-1) / np.sum(spin * direction, axis=-1)


class Robot:
    def __init__(self):
        self.model = m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = d = mujoco.MjData(m)
        env = yaml.safe_load(TRAIN.read_text())["env"]
        entity = env["scene"]["entities"]["robot"]
        control = env["actions"]["mixed"]["control"]
        self.joint_names = tuple(entity["joint_names"])
        self.actuator_names = tuple(entity["actuator_names"])
        self.joints = np.array([m.joint(name).id for name in self.joint_names])
        self.actuators = np.array([m.actuator(name).id for name in self.actuator_names])
        if not np.array_equal(m.actuator_trnid[self.actuators, 0], self.joints):
            raise ValueError("Go2W actuator-to-joint mapping disagrees with the control contract")
        self.qpos = m.jnt_qposadr[self.joints]
        self.qvel = m.jnt_dofadr[self.joints]
        self.base = m.body(entity["root_body_name"]).id
        free_joint = m.body_jntadr[self.base]
        if m.jnt_type[free_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Go2W base must have a free joint")
        address = m.jnt_qposadr[free_joint]
        self.root_qpos = slice(address, address + 7)
        velocity_address = m.jnt_dofadr[free_joint]
        self.root_qvel = slice(velocity_address, velocity_address + 6)

        self.wheel_bodies = np.array([m.body(f"robot/{leg}_wheel_link").id for leg in LEGS])
        self.wheel_geoms = np.array([m.geom(name).id for name in entity["geom_names"]])
        self.floor = m.geom("floor").id
        chain = m.jnt_bodyid[self.joints[:12]].reshape(4, 3)
        if not (np.allclose(m.jnt_axis[self.joints[:12]].reshape(4, 3, 3),
                            [[1, 0, 0], [0, 1, 0], [0, 1, 0]])
                and np.allclose(m.jnt_axis[self.joints[12:]], [0, 1, 0])):
            raise ValueError("Analytic IK requires the Go2W X/Y/Y leg axes and Y wheel axes")
        if not (np.allclose(m.jnt_pos[self.joints], 0)
                and np.allclose(m.body_quat[np.r_[chain.ravel(), self.wheel_bodies]], IDENTITY)
                and np.array_equal(m.body_parentid[chain[:, 0]], np.repeat(self.base, 4))
                and np.array_equal(m.body_parentid[chain[:, 1]], chain[:, 0])
                and np.array_equal(m.body_parentid[chain[:, 2]], chain[:, 1])
                and np.array_equal(m.body_parentid[self.wheel_bodies], chain[:, 2])
                and np.array_equal(m.jnt_bodyid[self.joints[12:]], self.wheel_bodies)
                and np.array_equal(m.geom_bodyid[self.wheel_geoms], self.wheel_bodies)
                and np.allclose(m.body_pos[chain[:, 1]][:, [0, 2]], 0)
                and np.allclose(m.body_pos[chain[:, 2]][:, :2], 0)
                and np.allclose(m.body_pos[self.wheel_bodies][:, :2], 0)):
            raise ValueError("Go2W analytic IK chain geometry changed")
        self.hip_origins = m.body_pos[chain[:, 0]].copy()
        self.hip_offsets = m.body_pos[chain[:, 1], 1].copy()
        self.upper = -m.body_pos[chain[:, 2], 2].copy()
        self.lower = -m.body_pos[self.wheel_bodies, 2].copy()
        self.hard = m.jnt_range[self.joints[:12]].copy()
        midpoint = self.hard.mean(axis=1)
        half = np.diff(self.hard, axis=1)[:, 0] * control["leg_limit_fraction"] / 2
        self.soft = np.column_stack((midpoint - half, midpoint + half))
        self.wheel_rate_limit = control["wheel_velocity_scale"]
        key = m.key(env["scene"]["default_keyframe_name"]).id
        mujoco.mj_resetDataKeyframe(m, d, key)
        self.default_q = d.qpos[self.qpos[:12]].copy()
        self.action_limits = self.soft.copy()
        if not control["full_leg_target_range"]:
            span = control["leg_position_scale"]
            self.action_limits[:, 0] = np.maximum(self.soft[:, 0], self.default_q - span)
            self.action_limits[:, 1] = np.minimum(self.soft[:, 1], self.default_q + span)
        mujoco.mj_kinematics(m, d)
        self.wheel_vertices = []
        profiles = []
        for geom, center, axis in zip(
                self.wheel_geoms, d.xpos[self.wheel_bodies], d.xaxis[self.joints[12:]]):
            mesh = m.geom_dataid[geom]
            start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            vertices = m.mesh_vert[start:start + count].astype(np.float64)
            # The convex hull and original mesh have the same support function.
            # Cache unique vertices, without relying on MuJoCo's mesh graph layout.
            vertices = np.unique(vertices, axis=0)
            self.wheel_vertices.append(vertices)
            relative = vertices @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom] - center
            axial = relative @ axis
            radial = np.linalg.norm(relative - axial[:, None] * axis, axis=1)
            # Remove roundoff-scale duplicates in the axial/radial cross section.
            profiles.append(hull(np.round(np.column_stack((axial, radial)), 7)))
        self.profiles = profiles
        self.wheel_radius = max(profile[:, 1].max() for profile in profiles)
        original_centers = d.xpos[self.wheel_bodies].copy()
        self.spawn_clearance = self.contact_points()[..., 2].copy()
        self.ground_z = d.geom_xpos[self.floor, 2]
        if not np.allclose(d.geom_xmat[self.floor].reshape(3, 3)[:, 2], UP) or not np.isclose(self.ground_z, 0):
            raise ValueError("The offline planner requires a level floor at z=0")
        self.stand_height = d.qpos[self.root_qpos][2] - self.spawn_clearance.mean()
        self.stand_xy = original_centers[:, :2] - d.qpos[self.root_qpos][:2]
        xy = self.stand_xy[None]
        self.stand_local, q, valid = self.targets(xy, np.zeros((1, 4)), self.stand_height)
        self.stand_local = self.stand_local[0]
        self.stand_q = q[0].ravel()
        if not valid.all():
            raise ValueError("Go2W calibrated standing pose is outside the workspace")
        self.set_pose(np.array([0, 0, self.stand_height]), IDENTITY, self.stand_q)
        self.stand_com = self.data.subtree_com[self.base].copy()

        self.collision_geoms = np.flatnonzero(m.geom_group == 3)
        self.ground_geoms = np.setdiff1d(self.collision_geoms, self.wheel_geoms)
        # Check all distinct nonadjacent link pairs, irrespective of contact masks.
        self.self_pairs = np.array([
            (i, j) for i, j in combinations(self.collision_geoms, 2)
            if (m.geom_bodyid[i] != m.geom_bodyid[j]
                and m.body_parentid[m.geom_bodyid[i]] != m.geom_bodyid[j]
                and m.body_parentid[m.geom_bodyid[j]] != m.geom_bodyid[i])
        ])

    def inverse(self, local):
        """Vectorized negative-knee, below-hip branch continuous from default q."""
        x, y, z = np.moveaxis(local - self.hip_origins, -1, 0)
        sagittal_z_sq = y*y + z*z - self.hip_offsets**2
        sagittal_z = -np.sqrt(np.maximum(sagittal_z_sq, 0))
        hip = np.arctan2(z, y) - np.arctan2(sagittal_z, self.hip_offsets)
        hip = (hip + np.pi) % (2 * np.pi) - np.pi
        cosine = (x*x + sagittal_z_sq - self.upper**2 - self.lower**2) / (2*self.upper*self.lower)
        calf = -np.arccos(np.clip(cosine, -1, 1))
        thigh = np.arctan2(-x, -sagittal_z) - np.arctan2(
            self.lower * np.sin(calf), self.upper + self.lower * np.cos(calf))
        q = np.stack((hip, thigh, calf), -1)
        lo, hi = self.hard.reshape(4, 3, 2).transpose(2, 0, 1)
        valid = ((sagittal_z_sq > 0) & (np.abs(cosine) <= 1)
                 & np.all((q >= lo) & (q <= hi), axis=-1))
        return q, valid

    def targets(self, xy, lift, height, compensate_camber=False):
        """Set wheel center height from a spin-invariant mesh envelope and solve IK."""
        height = np.asarray(height)
        local = np.concatenate((xy, (lift + self.wheel_radius - height[..., None])[..., None]), axis=-1)
        for _ in range(7 if compensate_camber else 5):
            q, _ = self.inverse(local)
            if compensate_camber:
                # A changing hip angle rotates the patch's vertical lever arm about
                # base x. Translate the axle laterally to counter that velocity.
                home_hip = self.stand_q.reshape(4, 3)[:, 0]
                local[..., 1] = xy[..., 1] - self.wheel_radius * (q[..., 0] - home_hip)
            for leg, profile in enumerate(self.profiles):
                hip = q[..., leg, 0, None]
                depth = np.max(np.cos(hip) * profile[:, 1] - np.sin(hip) * profile[:, 0], axis=-1)
                local[..., leg, 2] = lift[..., leg] + depth - height
        q, valid = self.inverse(local)
        return local, q, valid

    def set_pose(self, pos, quat, leg_q, wheel_q=None):
        self.data.qpos[self.root_qpos] = np.r_[pos, quat]
        self.data.qpos[self.qpos[:12]] = leg_q
        self.data.qpos[self.qpos[12:]] = 0 if wheel_q is None else wheel_q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def contact_points(self):
        """Lowest actual mesh height and centroid of its near-lowest contact patch."""
        points = []
        for geom, vertices in zip(self.wheel_geoms, self.wheel_vertices):
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            depth = vertices @ rotation[2]
            bottom = depth.min()
            point = vertices[depth <= bottom + 0.00025].mean(axis=0) @ rotation.T
            point[2] = bottom
            points.append(point + self.data.geom_xpos[geom])
        return np.asarray(points)

    def geometry(self):
        d = self.data
        return (d.xpos[self.wheel_bodies].copy(), self.contact_points(),
                d.xaxis[self.joints[12:]].copy(), d.xaxis[self.joints[:12]].reshape(4, 3, 3).copy(),
                d.subtree_com[self.base].copy())

    def collisions(self, threshold):
        """Count overlapping geometry pairs at the current pose."""
        m, d = self.model, self.data
        i, j = self.self_pairs.T
        near = np.linalg.norm(d.geom_xpos[i] - d.geom_xpos[j], axis=1) < m.geom_rbound[i] + m.geom_rbound[j]
        pairs = list(self.self_pairs[near])
        ground_near = self.ground_geoms[d.geom_xpos[self.ground_geoms, 2] < m.geom_rbound[self.ground_geoms]]
        pairs.extend((self.floor, geom) for geom in ground_near)
        hits = {}
        for first, second in pairs:
            distance = mujoco.mj_geomDistance(m, d, int(first), int(second), 0.005, None)
            if distance < -threshold:
                names = f"{m.geom(int(first)).name}/{m.geom(int(second)).name}"
                hits[names] = distance
        return hits
