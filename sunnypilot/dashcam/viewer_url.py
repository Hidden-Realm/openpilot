"""
Where the phone finds the dash cam viewer.

The viewer is protected by a short random access code that is baked into the
QR code on the device screen, so other people on the same Wi-Fi can't browse
the clips.
"""
import os
import secrets
import socket
import struct

PORT = 8099
TETHERING_IP = "192.168.43.1"

# easy to read off the device screen and type on a phone: no look-alikes like l/1 or o/0
TOKEN_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
TOKEN_LENGTH = 8


def get_token(root: str) -> str:
  path = os.path.join(root, "viewer_token")
  try:
    with open(path) as f:
      token = f.read().strip()
    if token:
      return token
  except OSError:
    pass
  token = "".join(secrets.choice(TOKEN_ALPHABET) for _ in range(TOKEN_LENGTH))
  os.makedirs(root, exist_ok=True)
  with open(path, "w") as f:
    f.write(token)
  return token


def wlan_ip(ifname: str = "wlan0") -> str | None:
  """IPv4 address of the Wi-Fi interface, either on a network or as the hotspot. None when Wi-Fi is down."""
  try:
    import fcntl
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
      packed = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", ifname[:15].encode()))  # SIOCGIFADDR
    return socket.inet_ntoa(packed[20:24])
  except (ImportError, OSError):
    return None


def viewer_url(ip: str, token: str) -> str:
  return f"http://{ip}:{PORT}/?k={token}"


def wifi_qr_payload(ssid: str, password: str) -> str:
  """The format the iPhone camera recognizes as 'join this Wi-Fi network'."""
  def esc(s: str) -> str:
    return "".join("\\" + c if c in '\\;,:"' else c for c in s)
  return f"WIFI:T:WPA;S:{esc(ssid)};P:{esc(password)};;"
