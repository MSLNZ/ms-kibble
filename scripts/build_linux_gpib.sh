#!/bin/bash

# The linux-gpib version to build
VERSION="4.3.7"

# Prerequisites to build linux-gpib
sudo apt-get install --yes linux-headers-$(uname -r) wget automake libtool flex bison

# Download and extract the specified linux-gpib version
echo "Downloading linux-gpib-$VERSION.tar.gz source from https://sourceforge.net ..."
wget -q -O linux-gpib-$VERSION.tar.gz https://sourceforge.net/projects/linux-gpib/files/linux-gpib%20for%203.x.x%20and%202.6.x%20kernels/$VERSION/linux-gpib-$VERSION.tar.gz/download
tar -xf linux-gpib-$VERSION.tar.gz
cd linux-gpib-$VERSION

# Build and install the kernel files
tar -xf linux-gpib-kernel-$VERSION.tar.gz
cd linux-gpib-kernel-$VERSION
make
sudo make install
cd ..

# Build and install the user files
tar -xf linux-gpib-user-$VERSION.tar.gz
cd linux-gpib-user-$VERSION
./bootstrap
./configure --sysconfdir=/etc
make
sudo make install
sudo ldconfig

# Create spincore group, if it does not already exist
if ! [ $(getent group gpib) ]; then
  sudo addgroup gpib
  echo "Created gpib group"
fi

# Clean up
cd ../..
rm -rf linux-gpib-$VERSION/
rm linux-gpib-$VERSION.tar.gz

# Print instructions for user
echo ""
echo "Installed, next steps:"
echo "1. Edit the value of 'board_type' in the 'interface' section of /etc/gpib.conf"
echo "   For the list of supported board types, see https://linux-gpib.sourceforge.io/doc_html/supported-hardware.html"
echo "   For example, for a NI GPIB-USB-HS+ controller set board_type = \"ni_usb_b\""
echo "     sudo nano /etc/gpib.conf"
echo "2. Plug in the GPIB-USB controller and make sure that it is listed as a USB device"
echo "     lsusb"
echo "3. Check the GPIB configuration, the following command should run without displaying an error"
echo "     sudo gpib_config"
echo "   If you get an error similar to:"
echo "     failed to configure boardtype: ni_pci"
echo "     failed to configure board"
echo "     main: Invalid argument"
echo "   that means you specified the wrong board_type in Step 1"
echo "4. Add yourself to the 'gpib' group"
echo "     sudo adduser $USER gpib"
echo "5. Reboot computer (easiest thing to do to make sure you can successfully run the examples)"
