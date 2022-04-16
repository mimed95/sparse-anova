from dynaconf import Dynaconf
from pathlib import Path

config_folder = Path(__file__).parent.resolve()
settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[config_folder / 'settings.toml', '.secrets.toml'],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
