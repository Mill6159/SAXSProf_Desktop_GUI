'''
Created on Jul 5, 2010

@author: Soren S. Nielsen

#******************************************************************************
# This file is part of RAW.
#
#    RAW is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RAW is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RAW.  If not, see <http://www.gnu.org/licenses/>.
#
#******************************************************************************
'''
import os
import copy
import threading
from math import pi, sin

import numpy as np
# import SASExceptions


class SASM:
    '''
        Small Angle Scattering Measurement (SASM) Object.
        Contains all information extracted from a SAS data file.
    '''

    def __init__(self, i, q, err, parameters):
        ''' Constructor

            parameters contains at least {'filename': filename_with_no_path}
            other reserved keys are:

            'counters' : [(countername, value),...] Info from counterfiles
            'fileHeader' : [(label, value),...] Info from the header in the loaded file
        '''

        #Raw intensity variables
        self._i_raw = np.array(i)
        self._q_raw = np.array(q)
        self._err_raw = np.array(err)
        self._parameters = parameters

        # Make an entry for analysis parameters i.e. Rg, I(0) etc:
        if 'analysis' not in self._parameters:
            self._parameters['analysis'] = {}
        if 'history' not in self._parameters:
            self._parameters['history'] = {}

        #Binned intensity variables
        self._i_binned = self._i_raw.copy()
        self._q_binned = self._q_raw.copy()
        self._err_binned = self._err_raw.copy()

        #Modified intensity variables
        self.i = self._i_raw.copy()
        self.q = self._q_raw.copy()
        self.err = self._err_raw.copy()

        self._scale_factor = 1
        self._offset_value = 0
        self._norm_factor = 1
        self._q_scale_factor = 1
        self._bin_size = 1

        #variables used for plot management
        self.item_panel = None
        self.plot_panel = None
        self.line = None
        self.err_line = None
        self.axes = None
        self.is_plotted = False
        self._selected_q_range = (0, len(self._q_binned))


    def setQrange(self, qrange):
        if qrange[0] < 0 or qrange[1] > (len(self._q_binned)):
            raise SASExceptions.InvalidQrange('Qrange: ' + str(qrange) + ' is not a valid q-range for a q-vector of length ' + str(len(self._q_binned)-1))
        else:
            self._selected_q_range = map(int, qrange)

    def getQrange(self):
        return self._selected_q_range

    def setAllParameters(self, new_parameters):
        self._parameters = new_parameters

    def getAllParameters(self):
        return self._parameters

    def getParameter(self, key):
        ''' Get parameter from parameters dict '''

        if self._parameters.has_key(key):
            return self._parameters[key]
        else:
            return None

    def setParameter(self, key, value):
        ''' insert key,value pair into parameters dict '''
        self._parameters[key] = value

    