"""
Invoked when python -m opmsim, runs the GUI-based opmsim
"""
from opmsim.gui import tk_gui

def main():
    tk_gui.main()

if __name__ == "__main__":
    main()
