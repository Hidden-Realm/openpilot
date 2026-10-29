"""
Dash cam: when holding LKAS saves a clip, MADS toggles steering on the release of a short press,
and a held press leaves steering alone.
"""
import pytest
from pytest_mock import MockerFixture

from cereal import custom
from opendbc.car import structs
from openpilot.selfdrive.selfdrived.events import Events
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
from openpilot.sunnypilot.mads import mads as mads_module

ButtonType = structs.CarState.ButtonEvent.Type
EventNameSP = custom.OnroadEventSP.EventName


class FakeParams:
  def get_bool(self, key, *args, **kwargs):
    return key == "Mads"

  def get(self, key, *args, **kwargs):
    return 0


def car_state(*lkas_events):
  return structs.CarState.new_message(cruiseState={"available": True},
                                      buttonEvents=[{"type": ButtonType.lkas, "pressed": p} for p in lkas_events])


@pytest.fixture
def make_mads(mocker: MockerFixture):
  def make(wheel_button: str):
    mocker.patch.object(mads_module, "Params", FakeParams)
    mocker.patch.object(mads_module, "wheel_button", lambda root: wheel_button)
    mocker.patch.object(mads_module, "dashcam_root", lambda: "")
    clock = mocker.patch.object(mads_module.time, "monotonic", return_value=0.0)

    selfdrive = mocker.MagicMock()
    selfdrive.CP = structs.CarParams.new_message(brand="honda")
    selfdrive.CP_SP = structs.CarParamsSP()
    selfdrive.params = FakeParams()
    selfdrive.events = Events()
    selfdrive.events_sp = EventsSP()
    selfdrive.enabled = False
    selfdrive.CS_prev = structs.CarState.new_message()
    m = mads_module.ModularAssistiveDrivingSystem(selfdrive)
    return m, clock
  return make


def step(m, clock, t, *lkas_events):
  clock.return_value = t
  m.events_sp.clear()
  m.update_events(car_state(*lkas_events))
  return m.events_sp.has(EventNameSP.lkasEnable)


def test_tap_toggles_on_release(make_mads):
  m, clock = make_mads("lkas")
  assert not step(m, clock, 0.0, True)
  assert step(m, clock, 0.3, False)


def test_hold_leaves_steering_alone(make_mads):
  m, clock = make_mads("lkas")
  assert not step(m, clock, 0.0, True)
  assert not step(m, clock, 1.0)
  assert not step(m, clock, 1.4, False)


def test_press_and_release_in_one_frame(make_mads):
  m, clock = make_mads("lkas")
  assert step(m, clock, 0.0, True, False)


@pytest.mark.parametrize("wheel_button", ["distance", "off"])
def test_other_settings_keep_toggle_on_press(make_mads, wheel_button):
  m, clock = make_mads(wheel_button)
  assert step(m, clock, 0.0, True)
  assert not step(m, clock, 2.0, False)
