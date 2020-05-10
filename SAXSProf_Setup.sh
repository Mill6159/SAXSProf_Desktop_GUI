## Set up Script for SAXSProf Desktop GUI ##
# Must have python3 and 'pip' installed on local comp.
# Note if python3 was properly installed, the packaged 'tkinter' should already be present
# If NOT. Uninstall and reinstall python3 and it should be there.


if command -v  python >/dev/null 2>&1 && echo Python 3 is installed BUT aliased to 'python' 2>/dev/null; then 
  python -m pip install numpy
  python -m pip install scipy
  python -m pip install matplotlib
else
  echo Python 3 is NOT aliased. Call python3 each time. 
  python3 -m pip install numpy
  python3 -m pip install scipy
  python3 -m pip install matplotlib; 
fi

