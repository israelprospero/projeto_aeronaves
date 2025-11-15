import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import modules.designTool as dt

with open("airplane_opt.json", "r") as file:
    airplane_opt = json.load(file)

dt.plot3d(airplane_opt)