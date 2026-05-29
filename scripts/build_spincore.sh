#!/bin/bash

# The version to download and build
VERSION="SpinAPI_linux-20250210-x86_64"

# Install build dependencies
sudo apt-get install wget build-essential cmake --yes

# Download source code
wget https://spincore.com/CD/Setup/linux/$VERSION.tar.gz

# Extract source code
tar -xzf $VERSION.tar.gz

# Build (only the API, not the examples)
cd $VERSION
mkdir build
cd build
cmake ..
make spinapi

# Move library
sudo mv src/libspinapi.so /usr/local/bin

# Clean up
cd ../..
rm -rf $VERSION/
rm $VERSION.tar.gz

# Create spincore group, if it does not already exist
if ! [ $(getent group spincore) ]; then
  sudo addgroup spincore
  echo "Created spincore group"
fi

# Create udev rule file
echo "SUBSYSTEM==\"usb\", ATTR{idVendor}==\"0403\", ATTR{idProduct}==\"c1ab\", ATTR{bcdDevice}==\"0001\", GROUP=\"spincore\", MODE=\"0664\"" | sudo tee /etc/udev/rules.d/99-spincore.rules > /dev/null

# Print instructions for user
echo "Next steps:"
echo "1. Add yourself to the 'spincore' group"
echo "     sudo adduser $USER spincore"
echo "2. Reboot computer (easiest thing to do to make sure you can successfully run the SpinCore example)"
