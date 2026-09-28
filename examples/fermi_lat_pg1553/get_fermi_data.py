"""Configure the Fermi-LAT event filter for PG 1553+113."""

import os

import gt_apps


def configure_filter(data_path="~/fermi-lat"):
    """Configure and return the Fermi-LAT event-filter application."""
    data_path = os.path.expanduser(data_path)
    event_list_path = os.path.join(data_path, "PG1553_events.list")
    output_path = os.path.join(data_path, "PG1553_filtered.fits")

    event_filter = gt_apps.filter
    event_filter["evclass"] = 128
    event_filter["evtype"] = 3
    event_filter["ra"] = 238.929  # [degree]
    event_filter["dec"] = 11.1901  # [degree]
    event_filter["rad"] = 10.0  # [degree]
    event_filter["emin"] = 100.0  # [MeV]
    event_filter["emax"] = 300000.0  # [MeV]
    event_filter["zmax"] = 90.0  # [degree]
    event_filter["tmin"] = 239557417.0  # [s MET]
    event_filter["tmax"] = 256970880.0  # [s MET]
    event_filter["infile"] = "@" + event_list_path
    event_filter["outfile"] = output_path
    return event_filter


if __name__ == "__main__":
    configure_filter()
