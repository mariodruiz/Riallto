# Riallto Ubuntu 24.04 setup

Currently there is support for bringing up Riallto on Ubuntu 24.04 with docker.
To use Riallto on Linux requires the use of the [xdna-driver](https://github.com/amd/xdna-driver) which is installed as part of the setup.
This driver requires version 6.10+ of the Linux kernel, these scripts will upgrade a standard Ubuntu 24.04 installation to this kernel version. __Using this kernel version will require disabling secure boot on your device.__

## Install steps

On an NPU enabled laptop running Ubuntu 24.04.

1. __Setup Docker.__

You can follow the steps [here](https://docs.docker.com/engine/install/ubuntu/).

2. __Add your user to the docker user group and then relogin.__

   ```sh
   sudo usermod -aG docker $USER ; exit
   ```

3. __Obtain a license file for Riallto.__

Please follow the [guide here](https://riallto.ai/prerequisites-aie-license.html#prerequisites-aie-license)

4. __Disable secure boot from your BIOS settings.__ For now we are using an unsigned kernel version requiring that secure boot is disabled before it can be used. To disable secure boot there is a [guide](https://learn.microsoft.com/en-us/windows-hardware/manufacture/desktop/disabling-secure-boot?view=windows-11) from Microsoft here, but often the steps depend on your hardware manufacturer.

5. Clone this the Riallto repository and then navigate to the installation script `git clone https://github.com/AMDResearch/Riallto.git && cd Riallto/scripts/linux/`.

6. __Run `./setup_riallto_linux.sh <your license file>`.__
This command will check the kernel version and if the xdna-driver has been installed. If the kernel is not 6.10 or the NPU device drivers are missing it will build them within a docker and install them on the host machine. This takes about 10 minutes to run and after completing successfully the user will be asked to restart.

7. __Reboot the machine.__
To finish upgrading the kernel to `6.10`.

8. __Re run the `./setup_riallto_linux.sh <your license file>` script.__
This will build the Riallto docker and will take about 20 minutes.

## Running Riallto / Running Tests
Inside this directory there are a few scripts.

* `launch_jupyter.sh <notebooks directory>` - will launch a jupyterlab server from a docker container allowing you to use Riallto notebooks in the directory provided (usually Riallto/notebooks).
* `run_pytest.sh` - will run a suit of pytests to test the operation of your NPU device and the Riallto installation.

### FAQ

### If you get the following error on boot, `error: bad shim signature`

This means that secure boot has not been disabled from the machine and it cannot run the necessary kernel version.

### If the Docker container does not start with `launch_jupyter.sh` or `run_pytest.sh` because of the `/dev/accel/accel0` does not exist

Check `dmesg`

```sh
sudo dmesg | grep "xdna"
```

If you see `loading /lib/firmware/amdnpu/1502_00/npu.sbin failed with error -22`, it means that the driver did not install properly.

Delete the NPU firmware:

```sh
sudo rm -rf /lib/firmware/amdnpu/1502_00/npu.sbin
```

Then, start the installation from scratch.
