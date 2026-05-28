## Helpful scripts

* `build_spincore.sh` &mdash; Builds the SpinCore API library and creates the *udev* rule file. Each `$USER` must add themselves to the *spincore* group to be able to access the SpinCore device without running scripts as sudo, e.g., `sudo adduser $USER spincore` and then reboot the computer to have the changes take effect.
