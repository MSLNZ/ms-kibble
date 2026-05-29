## Helpful scripts

* `build_linux_gpib.sh` &mdash; Build the linux-gpib library. Re-running this script is also required if the Linux *kernel* gets updated. Each `$USER` must add themselves to the *gpib* group to be able to access GPIB devices without running scripts as sudo, e.g., `sudo adduser $USER gpib` and then reboot the computer to have the changes take effect.

* `build_spincore.sh` &mdash; Builds the SpinCore API library and creates the *udev* rule file. Each `$USER` must add themselves to the *spincore* group to be able to access the SpinCore device without running scripts as sudo, e.g., `sudo adduser $USER spincore` and then reboot the computer to have the changes take effect.

* `setup_timetagger.sh` &mdash; Creates the *swabian* group and the *udev* rule file. Each `$USER` must add themselves to the *swabian* group to be able to access the TimeTagger device without running scripts as sudo, e.g., `sudo adduser $USER swabian` and then reboot the computer to have the changes take effect.
