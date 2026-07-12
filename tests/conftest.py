import os

from hypothesis import settings

# CI runs the derandomized profile (HYPOTHESIS_PROFILE=ci) so property tests
# cannot flake; locally the default profile keeps exploration on.
settings.register_profile("ci", derandomize=True, deadline=None, max_examples=100)
settings.register_profile("dev", deadline=None, max_examples=50)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
