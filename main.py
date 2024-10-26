# geo-clip/main.py

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geoclip.runner import main

if __name__ == "__main__":
    main()