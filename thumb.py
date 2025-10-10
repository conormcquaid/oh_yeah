import os
import time

from library.lcd.lcd_comm_rev_a import LcdCommRevA, Orientation

# sourced from https://github.com/mathoudebine/turing-smart-screen-python.git

l=LcdCommRevA()
l.Reset()
l.InitializeComm()
back='../oh_yeah/thumbnail.png'
#n2='../oh_yeah/other.png'
l.SetOrientation(Orientation.LANDSCAPE)
l.DisplayBitmap(back)

last_mod = os.path.getmtime(back)

while True:
   curr_mod = os.path.getmtime(back)
   if(curr_mod != last_mod):
      last_mod = curr_mod
      print(f"File '{back}' has changed!")
      ##l.DisplayBitmap(n2)
      l.DisplayBitmap(back)
   time.sleep(1)



