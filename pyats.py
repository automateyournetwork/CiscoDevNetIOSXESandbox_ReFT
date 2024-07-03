import os
import json
import logging
from pyats import aetest
from pyats.log.utils import banner

# ----------------
# Get logger for script
# ----------------

log = logging.getLogger(__name__)

# ----------------
# AE Test Setup
# ----------------
class common_setup(aetest.CommonSetup):
    """Common Setup section"""
# ----------------
# Connected to devices
# ----------------
    @aetest.subsection
    def connect_to_devices(self, testbed):
        """Connect to all the devices"""
        testbed.connect()
# ----------------
# Mark the loop for Learn Interfaces
# ----------------
    @aetest.subsection
    def loop_mark(self, testbed):
        aetest.loop.mark(Show_Run, device_name=testbed.devices)

# ----------------
# Test Case #1
# ----------------
class Show_Run(aetest.Testcase):
    """pyATS Get and Save Show Run"""

    @aetest.test
    def setup(self, testbed, device_name):
        """ Testcase Setup section """
        # Set current device in loop as self.device
        self.device = testbed.devices[device_name]

    @aetest.test
    def capture_show_run_raw(self):
        self.parsed_command = self.device.execute("show run brief")

        with open('show_run.txt', 'w') as f:
            f.write(self.parsed_command)

    @aetest.test
    def capture_parsed_show_ip_interface_brief(self):
        self.parsed_command = self.device.parse("show ip interface brief")
        
        # Convert the parsed command to JSON format
        parsed_command_json = json.dumps(self.parsed_command, indent=4)

        # Write the JSON data to a file
        with open('show_ip_interface_brief.json', 'w') as f:
            f.write(parsed_command_json)

    @aetest.test
    def capture_parsed_show_interface(self):
        self.parsed_command = self.device.parse("show interfaces")
        
        # Convert the parsed command to JSON format
        parsed_command_json = json.dumps(self.parsed_command, indent=4)

        # Write the JSON data to a file
        with open('show_interfaces.json', 'w') as f:
            f.write(parsed_command_json)

    @aetest.test
    def capture_parsed_show_access_lists(self):
        self.parsed_command = self.device.parse("show access-lists")
        
        # Convert the parsed command to JSON format
        parsed_command_json = json.dumps(self.parsed_command, indent=4)

        # Write the JSON data to a file
        with open('show_access_lists.json', 'w') as f:
            f.write(parsed_command_json)

    @aetest.test
    def capture_parsed_show_ip_route_lists(self):
        self.parsed_command = self.device.parse("show ip route")
        
        # Convert the parsed command to JSON format
        parsed_command_json = json.dumps(self.parsed_command, indent=4)

        # Write the JSON data to a file
        with open('show_ip_route.json', 'w') as f:
            f.write(parsed_command_json)

class CommonCleanup(aetest.CommonCleanup):
    @aetest.subsection
    def disconnect_from_devices(self, testbed):
        testbed.disconnect()

# for running as its own executable
if __name__ == '__main__':
    aetest.main()