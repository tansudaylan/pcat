"""Collect manually selected PCAT analysis commands for batch submission."""

import os

listcmnd = [ \
            'python $TDGU_PATH/fermi_lat_galactic_analysis.py pcat_lens_mock', \
            'python $TDGU_PATH/fermi_lat_galactic_analysis.py pcat_lens_mock_grid', \
            'python $PCAT_PATH/cnfg.py test_info', \
            'python $PCAT_PATH/cnfg.py test_prio', \
            'python $PCAT_PATH/cnfg.py test_atcr', \
            'python $PCAT_PATH/cnfg.py test_spmr', \
            'python $PCAT_PATH/cnfg.py test_time', \
            'python $PCAT_PATH/cnfg.py test_lowr', \
            'python $PCAT_PATH/cnfg.py test_errr', \
            'python $PCAT_PATH/cnfg.py test_nomi', \
            'python $PCAT_PATH/cnfg.py test_uppr', \
            'python $PCAT_PATH/cnfg.py test_post', \
            'python $PCAT_PATH/cnfg.py test_popl', \
            'python $TDGU_PATH/xray_back.py pcat_chan_inpt', \
            'python $TDGU_PATH/xray_back.py pcat_chan_mock', \
            'python $TDGU_PATH/fermi_lat_galactic_analysis.py pcat_ferm_mock_igal', \
            'python $TDGU_PATH/fermi_lat_galactic_analysis.py pcat_ferm_mock_igal_syst', \
            'python $TDGU_PATH/fermi_lat_galactic_analysis.py pcat_ferm_mock_igal_brok', \
            'python $TDGU_PATH/fermi_lat_galactic_analysis.py pcat_ferm_inpt_igal', \
            'python $TDGU_PATH/fermi_lat_galactic_analysis.py pcat_ferm_inpt_ptch', \
            'python $PCAT_PATH/cnfg.py test_psfn', \
             ]
for k in range(len(listcmnd)):
    try:
        os.system(listcmnd[k])
    except:
        pass
