(how-to)=
# How-to guides

These guides are here for when you have something specific to get done with your own data. Each
one starts from a goal, such as "convert my Kilosort output" or "open a probe whose shanks were
sorted separately", and takes you straight to the steps that get you there.

They assume you are already familiar with the GUI and how it works. If this is your first time,
we'd recommend starting with the {doc}`../tutorials/index`: they'll give you a feel for how
everything fits together, and these guides will make much more sense afterwards.

The guides are grouped by what you're trying to do:

{doc}`install/index`
: Add the optional packages for converting Kilosort output, the IBL online mode and the Channel
  Prediction plugin.

{doc}`prepare-data/index`
: Turn your spike sorting, raw data and histology into the files the GUI reads.

{doc}`configure/index`
: Tell the GUI where your data lives, from a single folder to multi-shank probes and side-by-side
  configurations.

{doc}`align/index`
: Pick up earlier work, recover a lost alignment, and adjust what the plots show while you align.

{doc}`channel-prediction/index`
: Run the region prediction and auto-alignment models on your own recordings.

{doc}`ibl-online/index`
: Align IBL probe insertions directly from the Alyx database.

{doc}`extend/index`
: Add your own plots and unit filters with a plugin.

Can't find what you're looking for? The {doc}`../reference/index` lists every setting, file and
shortcut, and the {doc}`../explanation/index` pages explain why the GUI works the way it does.

## All guides

```{toctree}
:maxdepth: 2

install/index
prepare-data/index
configure/index
align/index
channel-prediction/index
ibl-online/index
extend/index
```
