import os
import shutil
from pathlib import Path

from blip.run_blip import run_pipeline
from blip.config import parse_config

blip_root_dir = Path(os.path.dirname(os.path.dirname(__file__))).absolute()

# This runs BLIP on params_test.ini and fails if any exception is raised.
def test_end_to_end(tmp_path):
    source = blip_root_dir / "params_test.ini"
    target = tmp_path.absolute() / "params.ini"
    shutil.copy(source, target)

    # change output directory to be inside tmp_path
    params, inj, misc = parse_config(target, resume=False)
    params["out_dir"] = str(tmp_path / "run_result")
    config = (params, inj, misc)

    run_pipeline(config, resume=False)
