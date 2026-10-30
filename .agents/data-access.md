# Data access

Some functionality needs ONE/Alyx, the Allen DocDB or large raw ephys files.

- Do not assume network access, Alyx credentials or raw data are available.
- Mock ONE/Alyx and the file system at the boundary; tests never hit the network.
- Optional stacks (`ibl`, `allen`, `ephysatlas` extras) are imported lazily at the
  point of use. Check for them with `utils/optional.py` (`has_ibllib`,
  `IBL_EXTRA_HINT`) so offline startup works without extras.
- Raw data is large: read snippets or chunks, never whole files, and cache derived
  results next to the data.
- Don't work around bugs in upstream IBL packages (ONE, ibllib, ibl-neuropixel,
  iblatlas) unless the user asks; report them instead (file/line, effect, suggested
  fix) so the user can raise them upstream.
