import os
import logging
import subprocess

log = logging.getLogger()

try:
    env_dir = os.environ["VIRTUAL_ENV"]
except KeyError:
    raise RuntimeError("Please activate your virtual environment.")

# Set directory paths
src_dir = os.path.join(env_dir, "src")
firedrake_dir = os.path.join(src_dir, "firedrake")
ufl_dir = os.path.join(src_dir, "ufl")
pyadjoint_dir = os.path.join(src_dir, "pyadjoint")


def check_call(arguments):
    try:
        log.debug("Running command '%s'", " ".join(arguments))
        log.debug(subprocess.check_output(arguments, stderr=subprocess.STDOUT, env=os.environ).decode())
    except subprocess.CalledProcessError as e:
        log.debug(e.output.decode())
        raise


# Check out UFL branch
def set_branch(package_dir, branch):
    os.chdir(package_dir)
    try:
        check_call(["git", "pull"])
        log.info("Checking out branch %s" % branch)
        check_call(["git", "checkout", "-q", branch])
        log.info("Successfully checked out branch %s" % branch)
    except subprocess.CalledProcessError:
        log.error("Failed to check out branch %s" % branch)
        raise

set_branch(ufl_dir, "external-operator_dualspace")
set_branch(firedrake_dir, "pointwise-adjoint-operator_dualspace")
set_branch(pyadjoint_dir, "dualspace")
