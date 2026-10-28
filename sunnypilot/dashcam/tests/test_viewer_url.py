from openpilot.sunnypilot.dashcam.viewer_url import TOKEN_ALPHABET, TOKEN_LENGTH, get_token, viewer_url, wifi_qr_payload


def test_token_is_easy_to_type_and_stable(tmp_path):
  token = get_token(str(tmp_path))
  assert len(token) == TOKEN_LENGTH
  assert set(token) <= set(TOKEN_ALPHABET)
  assert not set(token) & set("l1io0")
  assert get_token(str(tmp_path)) == token


def test_viewer_url():
  assert viewer_url("192.168.43.1", "abcd2345") == "http://192.168.43.1:8099/?k=abcd2345"


def test_wifi_qr_escapes_special_characters():
  assert wifi_qr_payload("weedle-1a2b", "swag;swag") == "WIFI:T:WPA;S:weedle-1a2b;P:swag\\;swag;;"
