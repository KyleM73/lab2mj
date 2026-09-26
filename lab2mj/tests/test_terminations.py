"""Tests for lab2mj.terminations (offline; synthetic contacts via mj_step)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from lab2mj.ir import TerminationIR
from lab2mj.quat import quat_from_euler_xyz
from lab2mj.terminations import (
    TerminationSet,
    map_size_from_generator,
    net_contact_forces_w,
    resolve_body_ids,
)

from .shared import make_state, parse_fixture

BOX_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 0.3">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05" mass="2.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def term(name, func, params, time_out=False, module="isaaclab.envs.mdp.terminations") -> TerminationIR:
    return TerminationIR(name=name, func=f"{module}:{func}", params=params, time_out=time_out)


def settled_box() -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    model = mujoco.MjModel.from_xml_string(BOX_XML)
    data = mujoco.MjData(model)
    for _ in range(400):  # 2 s: box lands and settles on the plane
        mujoco.mj_step(model, data)
    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    return model, data, box_id


# ---------------------------------------------------------------------------------------
# Contact-force aggregation
# ---------------------------------------------------------------------------------------


def test_net_contact_force_equals_weight_on_settled_box():
    model, data, box_id = settled_box()
    forces = net_contact_forces_w(model, data, [box_id])
    assert forces.shape == (1, 3)
    # Plane pushes the 2 kg box up with ~mg.
    np.testing.assert_allclose(forces[0], [0.0, 0.0, 2.0 * 9.81], atol=0.5)
    # The plane (worldbody) sees the equal-and-opposite force.
    floor_forces = net_contact_forces_w(model, data, [0])
    np.testing.assert_allclose(floor_forces[0], [0.0, 0.0, -2.0 * 9.81], atol=0.5)


def test_multimap_dedupes_welded_bodies():
    # Two Isaac names folded into the same mj body must count the force once, not twice.
    body_map = {"torso": [1], "welded_child": [1]}
    assert resolve_body_ids(["torso", "welded_child"], body_map) == [1]
    assert resolve_body_ids(None, body_map) == [1]
    assert resolve_body_ids("torso", {"torso": [1], "other": [2]}) == [1]

    model, data, box_id = settled_box()
    ts = TerminationSet(
        [
            term(
                "body_contact",
                "illegal_contact",
                {"sensor_cfg": {"body_names": ["torso", "welded_child"]}, "threshold": 1.0},
            )
        ],
        step_dt=0.02,
        episode_length_s=20.0,
        body_map={"torso": [box_id], "welded_child": [box_id]},
    )
    history = ts._contact_history(model, data)
    assert history is not None and history.shape == (1, 1, 3)  # deduped to one tracked body
    np.testing.assert_allclose(history[0, 0], [0.0, 0.0, 2.0 * 9.81], atol=0.5)


# ---------------------------------------------------------------------------------------
# Individual terms
# ---------------------------------------------------------------------------------------


def test_time_out():
    ts = TerminationSet(
        [term("time_out", "time_out", {}, time_out=True)],
        step_dt=0.2,
        episode_length_s=1.0,  # max_episode_length = 5 steps
        body_map={},
    )
    model = mujoco.MjModel.from_xml_string(BOX_XML)
    data = mujoco.MjData(model)
    state = make_state()
    assert ts.check(model, data, state, 0.8) == (False, False, [])
    terminated, timed_out, reasons = ts.check(model, data, state, 1.0)
    assert (terminated, timed_out, reasons) == (False, True, ["time_out"])


def test_illegal_contact_threshold():
    model, data, box_id = settled_box()
    state = make_state(pos=data.qpos[:3], quat=data.qpos[3:7])

    def build(threshold):
        return TerminationSet(
            [term("body_contact", "illegal_contact", {"sensor_cfg": {"body_names": "box"}, "threshold": threshold})],
            step_dt=0.02,
            episode_length_s=20.0,
            body_map={"box": [box_id]},
        )

    terminated, timed_out, reasons = build(1.0).check(model, data, state, 0.02)
    assert (terminated, timed_out, reasons) == (True, False, ["body_contact"])
    assert build(100.0).check(model, data, state, 0.02) == (False, False, [])


def test_sensor_cfg_resolves_against_sensor_tracked_bodies():
    # Isaac resolves sensor_cfg.body_names against ContactSensor.body_names — the
    # bodies matched by the sensor's prim-path leaf — not the full articulation.
    model, data, box_id = settled_box()
    sensors = [{"name": "feet", "prim_path": "/World/envs/env_.*/Robot/.*_foot", "history_length": 0}]
    body_map = {"base": [0], "fl_foot": [box_id]}
    ts = TerminationSet(
        [
            term(
                "foot_contact",
                "illegal_contact",
                {"sensor_cfg": {"name": "feet", "body_names": None}, "threshold": 1.0},
            )
        ],
        step_dt=0.02,
        episode_length_s=20.0,
        body_map=body_map,
        contact_sensors=sensors,
    )
    assert ts.tracked_body_ids == [box_id]  # body_names=None matches the sensor subset, not 'base'
    # A pattern reaching only untracked bodies raises (Isaac raises on no-match too).
    with pytest.raises(ValueError, match="match none"):
        TerminationSet(
            [term("bad", "illegal_contact", {"sensor_cfg": {"name": "feet", "body_names": "base"}, "threshold": 1.0})],
            step_dt=0.02,
            episode_length_s=20.0,
            body_map=body_map,
            contact_sensors=sensors,
        )
    # Naming a sensor the manifest does not carry raises.
    with pytest.raises(ValueError, match="contact sensor 'imu'"):
        TerminationSet(
            [term("bad", "illegal_contact", {"sensor_cfg": {"name": "imu", "body_names": None}, "threshold": 1.0})],
            step_dt=0.02,
            episode_length_s=20.0,
            body_map=body_map,
            contact_sensors=sensors,
        )


def test_illegal_contact_windows_are_per_sensor():
    # A feet sensor with history 4 must not stretch a base sensor's 1-step window:
    # Isaac maxes each term over its OWN sensor's history buffer.
    model, data, box_id = settled_box()
    sensors = [
        {"name": "feet_sensor", "prim_path": "/World/envs/env_.*/Robot/foot", "history_length": 4},
        {"name": "base_sensor", "prim_path": "/World/envs/env_.*/Robot/base", "history_length": 1},
    ]
    body_map = {"foot": [box_id], "base": [box_id]}
    ts = TerminationSet(
        [
            term("foot_contact", "illegal_contact", {"sensor_cfg": {"name": "feet_sensor"}, "threshold": 1.0}),
            term("base_contact", "illegal_contact", {"sensor_cfg": {"name": "base_sensor"}, "threshold": 1.0}),
        ],
        step_dt=0.02,
        episode_length_s=20.0,
        body_map=body_map,
        contact_sensors=sensors,
    )
    ts.push_contact_forces(model, data)  # in contact
    # Two contact-free steps: the transient is 2 steps old.
    data.qpos[0:3] = (0.0, 0.0, 1.0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    ts.push_contact_forces(model, data)
    ts.push_contact_forces(model, data)
    state = make_state(pos=data.qpos[:3])
    _, _, reasons = ts.check(model, data, state, 0.02)
    assert reasons == ["foot_contact"]  # 4-step window keeps the spike; 1-step window forgot it


def test_illegal_contact_history_window():
    model, data, box_id = settled_box()

    def build(history_length):
        return TerminationSet(
            [term("body_contact", "illegal_contact", {"sensor_cfg": {"body_names": "box"}, "threshold": 1.0})],
            step_dt=0.02,
            episode_length_s=20.0,
            body_map={"box": [box_id]},
            contact_history_length=history_length,
        )

    ts_h2, ts_h1 = build(2), build(1)
    ts_h2.push_contact_forces(model, data)  # in contact
    ts_h1.push_contact_forces(model, data)
    # Teleport the box into the air: no contacts on the current step.
    data.qpos[0:3] = (0.0, 0.0, 1.0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    ts_h2.push_contact_forces(model, data)
    ts_h1.push_contact_forces(model, data)
    state = make_state(pos=data.qpos[:3])
    assert ts_h2.check(model, data, state, 0.02)[0] is True  # spike kept in the window
    assert ts_h1.check(model, data, state, 0.02)[0] is False  # window of 1 forgot it
    ts_h2.reset()  # reset clears the history -> falls back to (empty) current contacts
    assert ts_h2.check(model, data, state, 0.02)[0] is False


def test_vertical_contact_z_component_only():
    model, data, box_id = settled_box()
    state = make_state(pos=data.qpos[:3])

    def build(threshold):
        return TerminationSet(
            [
                term(
                    "vert",
                    "vertical_contact",
                    {"sensor_cfg": {"body_names": "box"}, "threshold": threshold},
                    module="contact_lab.tasks.base_wrench.mdp.terminations",
                )
            ],
            step_dt=0.02,
            episode_length_s=20.0,
            body_map={"box": [box_id]},
        )

    assert build(1.0).check(model, data, state, 0.02)[0] is True  # |Fz| ~ 19.6 N > 1
    assert build(100.0).check(model, data, state, 0.02)[0] is False


def test_bad_orientation():
    model = mujoco.MjModel.from_xml_string(BOX_XML)
    data = mujoco.MjData(model)
    rolled = make_state(quat=quat_from_euler_xyz(0.5, 0.0, 0.0))

    def build(limit_angle):
        return TerminationSet(
            [term("tilt", "bad_orientation", {"limit_angle": limit_angle})],
            step_dt=0.02,
            episode_length_s=20.0,
            body_map={},
        )

    # Tilt angle acos(-projected_gravity_z) equals the 0.5 rad roll.
    assert build(0.4).check(model, data, rolled, 0.02)[0] is True
    assert build(0.6).check(model, data, rolled, 0.02)[0] is False
    assert build(0.4).check(model, data, make_state(), 0.02)[0] is False


def test_root_height_below_minimum():
    model = mujoco.MjModel.from_xml_string(BOX_XML)
    data = mujoco.MjData(model)
    ts = TerminationSet(
        [term("fell", "root_height_below_minimum", {"minimum_height": 0.2})],
        step_dt=0.02,
        episode_length_s=20.0,
        body_map={},
    )
    assert ts.check(model, data, make_state(pos=(0, 0, 0.15)), 0.02)[0] is True
    assert ts.check(model, data, make_state(pos=(0, 0, 0.25)), 0.02)[0] is False


def test_terrain_out_of_bounds():
    model = mujoco.MjModel.from_xml_string(BOX_XML)
    data = mujoco.MjData(model)

    def build(terrain_size):
        return TerminationSet(
            [
                term(
                    "oob",
                    "terrain_out_of_bounds",
                    {"distance_buffer": 0.5},
                    time_out=True,
                    module="contact_lab.tasks.base_wrench.mdp.terminations",
                )
            ],
            step_dt=0.02,
            episode_length_s=20.0,
            body_map={},
            terrain_size_xy=terrain_size,
        )

    ts = build((10.0, 8.0))
    # x bound at 0.5 * 10 - 0.5 = 4.5; y bound at 0.5 * 8 - 0.5 = 3.5.
    assert ts.check(model, data, make_state(pos=(4.6, 0, 0.5)), 0.02) == (False, True, ["oob"])
    assert ts.check(model, data, make_state(pos=(4.4, 0, 0.5)), 0.02) == (False, False, [])
    assert ts.check(model, data, make_state(pos=(0, -3.6, 0.5)), 0.02)[1] is True
    # Plane terrain (no size) is infinite.
    assert build(None).check(model, data, make_state(pos=(1e6, 1e6, 0.5)), 0.02) == (False, False, [])


def test_map_size_from_generator():
    gen = {"size": [8.0, 8.0], "num_rows": 10, "num_cols": 20, "border_width": 20.0}
    assert map_size_from_generator(gen) == (10 * 8.0 + 40.0, 20 * 8.0 + 40.0)


# ---------------------------------------------------------------------------------------
# Build validation + fixture coverage
# ---------------------------------------------------------------------------------------


SPOT_BODY_NAMES = [
    "body",
    "fl_uleg",
    "fr_uleg",
    "hl_uleg",
    "hr_uleg",
    "fl_lleg",
    "fr_lleg",
    "hl_lleg",
    "hr_lleg",
    "fl_foot",
    "fr_foot",
    "hl_foot",
    "hr_foot",
]
G1_BODY_NAMES = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]


@pytest.mark.parametrize(
    "fixture, body_names",
    [("spot_velocity_env.yaml", SPOT_BODY_NAMES), ("g1_flat_env.yaml", G1_BODY_NAMES)],
)
def test_every_fixture_termination_builds_and_checks(fixture, body_names):
    ir = parse_fixture(fixture)
    assert ir.terminations, "fixture should declare terminations"
    body_map = {name: [i + 1] for i, name in enumerate(body_names)}
    ts = TerminationSet(
        ir.terminations,
        step_dt=ir.timing.policy_dt,
        episode_length_s=ir.timing.episode_length_s,
        body_map=body_map,
    )
    # Contact terms resolved a non-empty body set from the fixture patterns.
    contact_terms = [t for t in ts._terms if t.func in ("illegal_contact", "vertical_contact")]
    assert contact_terms and all(t.body_ids for t in contact_terms)

    model = mujoco.MjModel.from_xml_string(BOX_XML)
    data = mujoco.MjData(model)
    assert ts.check(model, data, make_state(), 0.0) == (False, False, [])
    # Past the episode length only the time_out-flagged terms fire.
    terminated, timed_out, reasons = ts.check(model, data, make_state(), ir.timing.episode_length_s)
    assert (terminated, timed_out) == (False, True)
    assert "time_out" in reasons


def test_spot_fixture_illegal_contact_matches_body_and_legs():
    ir = parse_fixture("spot_velocity_env.yaml")
    body_map = {name: [i + 1] for i, name in enumerate(SPOT_BODY_NAMES)}
    ts = TerminationSet(
        ir.terminations,
        step_dt=ir.timing.policy_dt,
        episode_length_s=ir.timing.episode_length_s,
        body_map=body_map,
    )
    (contact_term,) = [t for t in ts._terms if t.func == "illegal_contact"]
    matched = {name for name, ids in body_map.items() if ids[0] in contact_term.body_ids}
    assert matched == {"body", "fl_uleg", "fr_uleg", "hl_uleg", "hr_uleg", "fl_lleg", "fr_lleg", "hl_lleg", "hr_lleg"}
