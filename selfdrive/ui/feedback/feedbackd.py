#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from cereal import car
from openpilot.system.micd import SAMPLE_RATE, SAMPLE_BUFFER
from openpilot.sunnypilot.dashcam.clips import dashcam_root
from openpilot.sunnypilot.dashcam.hold import ButtonHold, wheel_button

FEEDBACK_MAX_DURATION = 10.0
ButtonType = car.CarState.ButtonEvent.Type
SAVE_BUTTONS = {"lkas": ButtonType.lkas, "distance": ButtonType.gapAdjustCruise}


def distance_hold_allowed(params: Params) -> bool:
  # with openpilot longitudinal, holding distance already switches experimental mode
  cp = params.get("CarParams")
  return cp is not None and not messaging.log_from_bytes(cp, car.CarParams).openpilotLongitudinalControl


def main():
  params = Params()
  pm = messaging.PubMaster(['userBookmark', 'audioFeedback'])
  sm = messaging.SubMaster(['rawAudioData', 'bookmarkButton', 'carState', 'selfdriveStateSP'])
  should_record_audio = False
  block_num = 0
  waiting_for_release = False
  early_stop_triggered = False
  button_name = wheel_button(dashcam_root())
  save_button = SAVE_BUTTONS.get(button_name)
  button_hold = ButtonHold()

  while True:
    sm.update()
    should_send_bookmark = False

    # dash cam: holding the chosen steering wheel button saves a clip, same as the on-screen bookmark.
    # A short press still does its usual job (MADS steering for LKAS, following distance for distance).
    if save_button is not None and sm.updated['carState']:
      if sm['carState'].canValid:
        pressed = None
        for be in sm['carState'].buttonEvents:
          if be.type == save_button:
            pressed = be.pressed
        if button_hold.update(pressed, time.monotonic()):
          if save_button != ButtonType.gapAdjustCruise or distance_hold_allowed(params):
            cloudlog.info(f"{button_name} button held - saving dash cam clip")
            should_send_bookmark = True
      else:
        button_hold.reset()

    # TODO: https://github.com/commaai/openpilot/issues/36015
    # only allow the LKAS button to record feedback when MADS is disabled
    if False and sm.updated['carState'] and sm['carState'].canValid and not sm['selfdriveStateSP'].mads.available:
      for be in sm['carState'].buttonEvents:
        if be.type == ButtonType.lkas:
          if be.pressed:
            if not should_record_audio:
              if params.get_bool("RecordAudioFeedback"):  # Start recording on first press if toggle set
                should_record_audio = True
                block_num = 0
                waiting_for_release = False
                early_stop_triggered = False
                cloudlog.info("LKAS button pressed - starting 10-second audio feedback")
              else:
                should_send_bookmark = True  # immediately send bookmark if toggle false
                cloudlog.info("LKAS button pressed - bookmarking")
            elif should_record_audio and not waiting_for_release:  # Wait for release of second press to stop recording early
              waiting_for_release = True
          elif waiting_for_release:  # Second press released
            waiting_for_release = False
            early_stop_triggered = True
            cloudlog.info("LKAS button released - ending recording early")

    if should_record_audio and sm.updated['rawAudioData']:
      raw_audio = sm['rawAudioData']
      msg = messaging.new_message('audioFeedback', valid=True)
      msg.audioFeedback.audio.data = raw_audio.data
      msg.audioFeedback.audio.sampleRate = raw_audio.sampleRate
      msg.audioFeedback.blockNum = block_num
      block_num += 1
      if (block_num * SAMPLE_BUFFER / SAMPLE_RATE) >= FEEDBACK_MAX_DURATION or early_stop_triggered:  # Check for timeout or early stop
        should_send_bookmark = True  # send bookmark at end of audio segment
        should_record_audio = False
        early_stop_triggered = False
        cloudlog.info("10-second recording completed or second button press - stopping audio feedback")
      pm.send('audioFeedback', msg)

    if sm.updated['bookmarkButton']:
      cloudlog.info("Bookmark button pressed!")
      should_send_bookmark = True

    if should_send_bookmark:
      msg = messaging.new_message('userBookmark', valid=True)
      pm.send('userBookmark', msg)


if __name__ == '__main__':
  main()
