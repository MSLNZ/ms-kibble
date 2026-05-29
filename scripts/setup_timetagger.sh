#!/bin/bash

# Create swabian group, if it does not already exist
if ! [ $(getent group swabian) ]; then
  sudo addgroup swabian
  echo "Created swabian group"
fi

# Create udev rule file
echo "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"151f\", ATTR{idProduct}==\"012b\", GROUP=\"swabian\", MODE=\"0664\"" | sudo tee /etc/udev/rules.d/99-swabian.rules > /dev/null

# Print instructions for user
echo "Next steps:"
echo "1. Add yourself to the 'swabian' group"
echo "     sudo adduser $USER swabian"
echo "2. Reboot computer (easiest thing to do to make sure you can successfully run the TimeTagger example)"
